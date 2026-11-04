#!/usr/bin/env python3
"""Run E0--E4 on fixed T3 references and the existing APRA correction budget."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import DATASETS, evaluation_frame, signal_names
from experiments.bimodal_reference.run_amr_feature_scoring import load_caches, read_score, restore, write_score
from experiments.temporal_matching.runner import publish_json_no_replace, publish_npz_no_replace
from src.models.apra_trem import APRATREM, b3_token_evidence
from src.models.bimodal_reference_vit4ts import _fitted_standardized_windows, _visual_costs
from src.models.fixed_reference_numeric_evidence import positive_median
from src.models.numeric_temporal_matching import match_paths
from src.models.vfg_paired_vit4ts import _patch_tokens, select_common_visual_windows


BASELINES = ("E0_B3", "E1_APRA_only")


def output_variants(config):
    return list(BASELINES) + list(config["variants"])


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = APRATREM(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def signed_calibration(local, base, full_length, device):
    point = restore(local, full_length, device)
    magnitude = np.abs(point[np.isfinite(point)])
    magnitude = magnitude[magnitude > 1e-12]
    if not len(magnitude):
        return np.zeros_like(point), 1.0
    scale = float(np.clip(
        positive_median(base) / max(float(np.median(magnitude)), 1e-12), 1e-4, 1e4,
    ))
    return point * scale, scale


def reference_raw(raw, ids, columns):
    patches = raw[ids].reshape(len(ids), 14, 16)
    return patches.gather(1, columns[..., None].expand(-1, -1, 16))


@torch.inference_mode()
def model_evidence(model, windows, patch, reference_ids, reference_columns, batch_size=256):
    numeric = []; visual = []
    for start in range(0, len(windows), batch_size):
        numeric.append(model.numeric(windows[start:start + batch_size]))
        visual.append(model.visual(patch[start:start + batch_size]))
    numeric = torch.cat(numeric); visual = torch.cat(visual)
    combined = []; apra = []; trem = []; rows = []
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        ids = reference_ids[start:stop]
        columns = reference_columns[start:stop]
        expanded = columns[..., None]
        nr_full = numeric[ids]; vr_full = visual[ids]
        nr = nr_full.gather(1, expanded.expand(-1, -1, nr_full.shape[-1]))
        vr = vr_full.gather(1, expanded.expand(-1, -1, vr_full.shape[-1]))
        nq = numeric[start:stop]; vq = visual[start:stop]
        rq = windows[start:stop].reshape(stop - start, 14, 16)
        rr = reference_raw(windows, ids, columns)
        base = b3_token_evidence(nq, vq, nr, vr, rq, rr)
        tick = time.perf_counter()
        output = model.forward_tokens(nq, vq, nr, vr, base)
        elapsed = time.perf_counter() - tick
        alignment = output["alignment"]
        combined.append(output["delta"]); apra.append(output["apra_delta"]); trem.append(output["trem_delta"])
        weights = output["fusion_weights"]
        rows.append({
            "inference_seconds": elapsed,
            "rejection": float(alignment["rejection"].mean().cpu()),
            "reliability": float(alignment["reliability"].mean().cpu()),
            "structural_numeric": float(alignment["structural_numeric"].norm(dim=-1).mean().cpu()),
            "structural_visual": float(alignment["structural_visual"].norm(dim=-1).mean().cpu()),
            "unmatched_numeric": float(alignment["unmatched_numeric"].norm(dim=-1).mean().cpu()),
            "unmatched_visual": float(alignment["unmatched_visual"].norm(dim=-1).mean().cpu()),
            "b3_token_evidence": float(base.mean().cpu()),
            "apra_delta_mean": float(output["apra_delta"].mean().cpu()),
            "trem_delta_mean": float(output["trem_delta"].mean().cpu()),
            "trem_delta_abs": float(output["trem_delta"].abs().mean().cpu()),
            "local_dynamics_norm": float(output["local_dynamics"].norm(dim=-1).mean().cpu()),
            "multiscale_context_norm": float(output["multiscale_context"].norm(dim=-1).mean().cpu()),
            "local_fusion_weight": float(weights[..., 0].mean().cpu()) if weights.shape[-1] == 2 else float(model.trem.use_local),
            "context_fusion_weight": float(weights[..., -1].mean().cpu()) if weights.shape[-1] == 2 else float(model.trem.use_multiscale),
        })
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    return torch.cat(combined), torch.cat(apra), torch.cat(trem), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--signals", nargs="+")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rcca_config = json.loads(Path(config["rcca_config"]).read_text())
    variants = output_variants(config)
    device = torch.device(args.device)
    data_root = Path(config["data_root"])
    feature = Path(config["feature_scoring_run"])
    apra_run = Path(config["apra_run"])
    stats = json.loads(Path(config["normalization_stats"]).read_text())
    datasets = args.datasets or list(DATASETS)
    tasks = [(dataset, signal) for dataset in datasets for signal in signal_names(data_root, dataset)]
    if args.signals:
        tasks = [task for task in tasks if task[1] in set(args.signals)]
    tasks = tasks[args.worker_index::args.num_workers]
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    gamma = float(config["loss_weights"]["gamma"])
    records = []; diagnostics = []; failures = 0; current = None; models = {}
    for ordinal, (dataset, signal) in enumerate(tasks, 1):
        tick = time.perf_counter()
        expected = [args.output_root / "point_scores" / dataset / signal / f"{variant}.npz"
                    for variant in variants]
        diagnostic_path = args.output_root / "feature_diagnostics" / dataset / signal / "evidence.npz"
        if args.skip_complete and diagnostic_path.is_file() and all(path.is_file() for path in expected):
            records.append({"dataset": dataset, "signal": signal, "status": "skipped_complete"})
            continue
        try:
            frame = evaluation_frame(data_root, dataset, signal)
            raw = frame.value.to_numpy(np.float64); timestamps = frame.timestamp.to_numpy()
            final, block, _, _, digest = load_caches(
                raw, {"dataset": dataset, "signal": signal}, {"cache": rcca_config["cache"]},
            )
            patch = _patch_tokens(block["intermediate_patch"]).float().to(device)
            pool = select_common_visual_windows(_patch_tokens(final["patch"]).to(device))
            visual_costs = _visual_costs(patch.reshape(len(patch), 14, 14, -1), pool)
            t3 = match_paths(visual_costs, pool, band=2, penalty_scale=.1)
            with np.load(feature / "feature_diagnostics" / dataset / signal / "evidence.npz") as stored:
                if not np.array_equal(t3.reference_ids.cpu().numpy(), stored["t3_reference_window_ids"]):
                    raise RuntimeError("T3 reference mismatch")
                if not np.array_equal(t3.reference_columns.cpu().numpy(), stored["t3_reference_columns"]):
                    raise RuntimeError("T3 path mismatch")
            ids = t3.reference_ids.long().to(device)
            columns = t3.reference_columns.round().long().to(device)
            fitted = stats[dataset][signal]
            start = int(fitted["fit_points"]) if dataset in {"MSL", "SMAP"} else 0
            windows = _fitted_standardized_windows(raw, fitted, start).to(device)
            scores = {}
            source = {
                "E0_B3": feature / "point_scores" / dataset / signal / "B3_raw_numeric_l010.npz",
                "E1_APRA_only": apra_run / "point_scores" / dataset / signal / "APRA_only.npz",
            }
            for name, path in source.items():
                scores[name], source_time = read_score(path)
                if not np.array_equal(source_time, timestamps):
                    raise RuntimeError(f"timestamp mismatch {name}")
            if current != dataset:
                models.clear()
                for variant in config["variants"]:
                    models[variant], _ = load_model(
                        args.output_root / "checkpoints" / variant / f"{dataset}.pt", device,
                    )
                current = dataset
            evidence = {}
            for variant, model in models.items():
                model_tick = time.perf_counter()
                local, local_apra, local_trem, diagnostic = model_evidence(
                    model, windows, patch, ids, columns,
                )
                point, scale = signed_calibration(local, scores["E0_B3"], len(raw), device)
                scores[variant] = scores["E0_B3"] + gamma * point
                evidence[f"point_combined_correction__{variant}"] = scores[variant] - scores["E0_B3"]
                evidence[f"point_trem_change_vs_E1__{variant}"] = scores[variant] - scores["E1_APRA_only"]
                diagnostic.update({
                    "calibration_scale": scale,
                    "total_inference_seconds": time.perf_counter() - model_tick,
                    "score_change_vs_E1_mean": float((scores[variant] - scores["E1_APRA_only"]).mean()),
                    "score_change_vs_E1_abs": float(np.abs(scores[variant] - scores["E1_APRA_only"]).mean()),
                })
                diagnostics.append({"dataset": dataset, "signal": signal,
                                    "variant": variant, **diagnostic})
            for variant in variants:
                metadata = {
                    "variant": variant, "labels_opened": False,
                    "fixed_reference": "T3", "fixed_dtw_anchor": "T3",
                    "code_commit": commit, "input_digest": digest[:20],
                    "base": "E1_APRA_only" if variant not in BASELINES else None,
                    "shared_apra_correction_budget": gamma if variant not in BASELINES else None,
                }
                write_score(args.output_root / "point_scores" / dataset / signal / f"{variant}.npz",
                            scores[variant], timestamps, metadata, time.perf_counter() - tick)
            publish_npz_no_replace(diagnostic_path, **evidence)
            records.append({"dataset": dataset, "signal": signal, "status": "passed",
                            "elapsed_seconds": time.perf_counter() - tick, "labels_opened": False,
                            "t3_reference_error": 0, "t3_path_error": 0})
            print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s", flush=True)
        except Exception as error:
            failures += 1
            records.append({"dataset": dataset, "signal": signal, "status": "failed",
                            "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {dataset}/{signal}: {error}", flush=True)
    publish_json_no_replace(
        args.output_root / "inference_manifests" / f"worker_{args.worker_index:02d}_of_{args.num_workers:02d}.json",
        {"status": "passed" if not failures else "failed", "records": records,
         "diagnostics": diagnostics, "failures": failures,
         "labels_opened": False, "code_commit": commit},
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

