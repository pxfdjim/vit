#!/usr/bin/env python3
"""Run fixed-T3-path RCCA/CADM scoring for one inference shard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import DATASETS, evaluation_frame, signal_names
from experiments.bimodal_reference.run_amr_feature_scoring import (
    load_caches, read_score, restore_aligned, write_score,
)
from experiments.temporal_matching.runner import publish_json_no_replace, publish_npz_no_replace
from src.models.bimodal_reference_vit4ts import (
    _calibrate, _fitted_standardized_windows, _token_costs, _visual_costs,
)
from src.models.fixed_reference_numeric_evidence import score_on_fixed_paths
from src.models.numeric_temporal_matching import match_paths
from src.models.rcca_cadm import RCCACADM
from src.models.vfg_paired_vit4ts import _patch_tokens, select_common_visual_windows


BASELINES = ("T3", "E2", "B3", "AMR_native", "C2_simplified")
PRIMARY_MODELS = ("M1_full", "M2_no_rcca", "M3_same_window", "M4_no_synthetic_objective")
CONTROLS = ("M5_shuffled_reference", "M6_no_cross_score")


def output_variants(config):
    return list(BASELINES) + list(PRIMARY_MODELS) + list(CONTROLS) + [
        f"M1_seed_{seed}" for seed in config["stability_seeds"] if seed != config["primary_seed"]]


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = RCCACADM(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def entropy(weights: torch.Tensor) -> tuple[float, float]:
    p = weights.float().clamp_min(1e-12)
    normalized = -(p * p.log()).sum(-1) / np.log(p.shape[-1])
    diagonal = p.diagonal(dim1=-2, dim2=-1).mean()
    return float(normalized.mean().cpu()), float(diagonal.cpu())


@torch.inference_mode()
def model_evidence(model: RCCACADM, windows: torch.Tensor, patch: torch.Tensor,
                   reference_ids: torch.Tensor, reference_columns: torch.Tensor,
                   *, shuffle_reference: bool = False, batch_size: int = 256):
    numeric, visual = [], []
    for start in range(0, len(windows), batch_size):
        numeric.append(model.numeric(windows[start:start + batch_size]))
        visual.append(model.visual(patch[start:start + batch_size]))
    numeric = torch.cat(numeric); visual = torch.cat(visual)
    used_ids = reference_ids
    logits, conditional = [], []
    attention_entropy, attention_diagonal = [], []
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        ids = used_ids[start:stop]
        columns = reference_columns[start:stop][..., None]
        nr_full, vr_full = numeric[ids], visual[ids]
        nr = nr_full.gather(1, columns.expand(-1, -1, nr_full.shape[-1]))
        vr = vr_full.gather(1, columns.expand(-1, -1, vr_full.shape[-1]))
        if shuffle_reference:
            # Shuffle the aligned reference Tokens themselves. Rolling only the
            # reference IDs is a no-op when T3 selects one common reference for
            # every query in a short signal.
            nr = nr.roll(1, dims=0).flip(1)
            vr = vr.roll(1, dims=0).flip(1)
        output = model.forward_tokens(numeric[start:stop], visual[start:stop], nr, vr)
        logits.append(output["logits"]); conditional.append(output["conditional_difference"])
        for weights in output["attention"].values():
            ent, diag = entropy(weights)
            attention_entropy.append(ent); attention_diagonal.append(diag)
    diagnostics = {
        "attention_entropy": float(np.mean(attention_entropy)) if attention_entropy else np.nan,
        "attention_diagonal_mass": float(np.mean(attention_diagonal)) if attention_diagonal else np.nan,
        "cadm_logit_mean": float(torch.cat(logits).mean().cpu()),
        "cadm_logit_std": float(torch.cat(logits).std(unbiased=False).cpu()),
        "conditional_difference_norm": float(torch.cat(conditional).norm(dim=-1).mean().cpu()),
    }
    return numeric, torch.sigmoid(torch.cat(logits)), diagnostics


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
    config = json.loads(args.config.read_text()); variants = output_variants(config)
    data_root = Path(config["data_root"]); feature_root = Path(config["feature_scoring_run"])
    ablation_root = Path(config["training_ablation_run"])
    stats = json.loads(Path(config["normalization_stats"]).read_text())
    device = torch.device(args.device)
    datasets = args.datasets or list(DATASETS)
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    tasks = [(dataset, signal) for dataset in datasets for signal in signal_names(data_root, dataset)]
    if args.signals:
        tasks = [(dataset, signal) for dataset, signal in tasks if signal in set(args.signals)]
        missing = sorted(set(args.signals) - {signal for _, signal in tasks})
        if missing:
            raise ValueError(f"unknown signals for selected datasets: {missing}")
    tasks = tasks[args.worker_index::args.num_workers]
    records, diagnostic_rows, failures = [], [], 0
    current_dataset = None; models = {}; payloads = {}
    code_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=ROOT, text=True).strip()
    for ordinal, (dataset, signal) in enumerate(tasks, 1):
        started = time.perf_counter()
        expected = [args.output_root / "point_scores" / dataset / signal / f"{v}.npz"
                    for v in variants]
        diag_path = args.output_root / "feature_diagnostics" / dataset / signal / "evidence.npz"
        if args.skip_complete and diag_path.is_file() and all(path.is_file() for path in expected):
            records.append({"dataset": dataset, "signal": signal, "status": "skipped_complete"})
            continue
        try:
            frame = evaluation_frame(data_root, dataset, signal)
            raw = frame.value.to_numpy(np.float64); timestamps = frame.timestamp.to_numpy()
            context = {"dataset": dataset, "signal": signal}
            final, block, _, _, digest = load_caches(raw, context, {"cache": config["cache"]})
            patch = _patch_tokens(block["intermediate_patch"]).float().to(device)
            pool = select_common_visual_windows(_patch_tokens(final["patch"]).to(device))
            visual_costs = _visual_costs(patch.reshape(len(patch), 14, 14, -1), pool)
            t3_match = match_paths(visual_costs, pool, band=2, penalty_scale=0.1)
            evidence_path = feature_root / "feature_diagnostics" / dataset / signal / "evidence.npz"
            with np.load(evidence_path, allow_pickle=False) as stored:
                expected_ids = stored["t3_reference_window_ids"]
                expected_columns = stored["t3_reference_columns"]
                raw_point = stored["point_raw_t3"].astype(np.float64)
            if not np.array_equal(t3_match.reference_ids.cpu().numpy(), expected_ids):
                raise RuntimeError("T3 reference mismatch")
            if not np.array_equal(t3_match.reference_columns.cpu().numpy(), expected_columns):
                raise RuntimeError("T3 path mismatch")
            reference_ids = t3_match.reference_ids.long().to(device)
            reference_columns = t3_match.reference_columns.round().long().to(device)
            fitted = stats[dataset][signal]
            start_index = int(fitted["fit_points"]) if dataset in {"MSL", "SMAP"} else 0
            windows = _fitted_standardized_windows(raw, fitted, start_index).to(device)

            source_map = {
                "T3": feature_root / "point_scores" / dataset / signal / "T3.npz",
                "E2": feature_root / "point_scores" / dataset / signal / "E2_T3_raw_l010.npz",
                "B3": feature_root / "point_scores" / dataset / signal / "B3_raw_numeric_l010.npz",
                "AMR_native": feature_root / "point_scores" / dataset / signal / "AMR_native.npz",
                "C2_simplified": ablation_root / "point_scores" / dataset / signal / "C2_no_cross.npz",
            }
            scores = {}
            for name, path in source_map.items():
                scores[name], source_time = read_score(path)
                if not np.array_equal(source_time, timestamps):
                    raise RuntimeError(f"timestamp mismatch: {name}")

            if current_dataset != dataset:
                models.clear(); payloads.clear()
                model_specs = [(name, config["primary_seed"]) for name in PRIMARY_MODELS]
                model_specs += [("M1_full", seed) for seed in config["stability_seeds"]
                                if seed != config["primary_seed"]]
                for name, seed in model_specs:
                    key = (name, seed)
                    path = args.output_root / "checkpoints" / name / f"seed_{seed}" / f"{dataset}.pt"
                    models[key], payloads[key] = load_model(path, device)
                current_dataset = dataset

            point_numeric, point_cross, local_cross, model_diagnostics = {}, {}, {}, {}
            tokens_by_key = {}
            for name in PRIMARY_MODELS:
                key = (name, config["primary_seed"]); model = models[key]
                tick = time.perf_counter()
                numeric, cross_local, diag = model_evidence(
                    model, windows, patch, reference_ids, reference_columns)
                diag["inference_seconds"] = time.perf_counter() - tick
                numeric_costs, token_scale = _calibrate(visual_costs, _token_costs(numeric, pool))
                numeric_local = score_on_fixed_paths(numeric_costs, t3_match)
                point_numeric[name], numeric_point_scale = restore_aligned(
                    numeric_local, scores["T3"], len(raw), device)
                point_cross[name], cross_point_scale = restore_aligned(
                    cross_local, scores["T3"], len(raw), device)
                local_cross[name] = cross_local.cpu().numpy(); tokens_by_key[key] = numeric
                diag.update({"token_cost_scale": token_scale,
                             "numeric_point_scale": numeric_point_scale,
                             "cross_point_scale": cross_point_scale})
                model_diagnostics[name] = diag
                scores[name] = (0.90 * scores["T3"] + 0.05 * raw_point
                                + 0.025 * point_numeric[name] + 0.025 * point_cross[name])

            primary_model = models[("M1_full", config["primary_seed"])]
            _, shuffled_local, shuffled_diag = model_evidence(
                primary_model, windows, patch, reference_ids, reference_columns,
                shuffle_reference=True)
            point_cross["M5_shuffled_reference"], shuffled_scale = restore_aligned(
                shuffled_local, scores["T3"], len(raw), device)
            scores["M5_shuffled_reference"] = (
                0.90 * scores["T3"] + 0.05 * raw_point
                + 0.025 * point_numeric["M1_full"]
                + 0.025 * point_cross["M5_shuffled_reference"])
            scores["M6_no_cross_score"] = (
                0.90 * scores["T3"] + 0.05 * raw_point + 0.05 * point_numeric["M1_full"])
            shuffled_diag.update({"cross_point_scale": shuffled_scale,
                                  "mean_abs_cross_local_change_vs_M1": float(
                                      np.mean(np.abs(shuffled_local.cpu().numpy()-local_cross["M1_full"])))})
            model_diagnostics["M5_shuffled_reference"] = shuffled_diag
            model_diagnostics["M6_no_cross_score"] = {"cross_weight": 0.0}

            for seed in config["stability_seeds"]:
                if seed == config["primary_seed"]:
                    continue
                name = f"M1_seed_{seed}"; key = ("M1_full", seed); model = models[key]
                tick = time.perf_counter()
                numeric, cross_local, diag = model_evidence(
                    model, windows, patch, reference_ids, reference_columns)
                diag["inference_seconds"] = time.perf_counter() - tick
                numeric_costs, token_scale = _calibrate(visual_costs, _token_costs(numeric, pool))
                numeric_local = score_on_fixed_paths(numeric_costs, t3_match)
                point_numeric[name], numeric_scale = restore_aligned(
                    numeric_local, scores["T3"], len(raw), device)
                point_cross[name], cross_scale = restore_aligned(
                    cross_local, scores["T3"], len(raw), device)
                scores[name] = (0.90 * scores["T3"] + 0.05 * raw_point
                                + 0.025 * point_numeric[name] + 0.025 * point_cross[name])
                diag.update({"token_cost_scale": token_scale, "numeric_point_scale": numeric_scale,
                             "cross_point_scale": cross_scale})
                model_diagnostics[name] = diag

            m5_change = model_diagnostics["M5_shuffled_reference"][
                "mean_abs_cross_local_change_vs_M1"]
            if m5_change <= 1e-8:
                raise RuntimeError("M5 reference replacement did not change CADM evidence")
            m6_expected = 0.90*scores["T3"] + 0.05*raw_point + 0.05*point_numeric["M1_full"]
            m6_error = float(np.max(np.abs(scores["M6_no_cross_score"] - m6_expected)))
            if m6_error != 0:
                raise RuntimeError("M6 formula mismatch")

            for variant in variants:
                metadata = {"variant": variant, "labels_opened": False,
                            "fixed_reference": "T3", "fixed_dtw_path": "T3",
                            "code_commit": code_commit, "input_digest": digest[:20],
                            "visual_weight": 0.9, "raw_weight": 0.05,
                            "numeric_weight": 0.05 if variant == "M6_no_cross_score" else (
                                0.025 if variant not in BASELINES else None),
                            "cross_weight": 0.0 if variant == "M6_no_cross_score" else (
                                0.025 if variant not in BASELINES else None)}
                write_score(args.output_root / "point_scores" / dataset / signal / f"{variant}.npz",
                            scores[variant], timestamps, metadata, time.perf_counter()-started)
            diagnostic_payload = {"point_raw_t3": raw_point,
                                  "point_numeric__M1_full": point_numeric["M1_full"],
                                  "point_cross__M1_full": point_cross["M1_full"],
                                  "point_cross__M5_shuffled_reference": point_cross["M5_shuffled_reference"]}
            for name in PRIMARY_MODELS:
                diagnostic_payload[f"point_numeric__{name}"] = point_numeric[name]
                diagnostic_payload[f"point_cross__{name}"] = point_cross[name]
            publish_npz_no_replace(diag_path, **diagnostic_payload)
            for name, row in model_diagnostics.items():
                diagnostic_rows.append({"dataset": dataset, "signal": signal,
                                        "variant": name, **row})
            records.append({"dataset": dataset, "signal": signal, "status": "passed",
                            "elapsed_seconds": time.perf_counter()-started,
                            "labels_opened": False, "t3_reference_error": 0,
                            "t3_path_error": 0, "m5_reference_effect": m5_change,
                            "m6_formula_error": m6_error})
            print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s", flush=True)
        except Exception as error:
            failures += 1
            records.append({"dataset": dataset, "signal": signal, "status": "failed",
                            "error": f"{type(error).__name__}: {error}", "labels_opened": False})
            print(f"FAILED {dataset}/{signal}: {error}", flush=True)
    manifest = {"status": "passed" if failures == 0 else "failed",
                "worker_index": args.worker_index, "num_workers": args.num_workers,
                "records": records, "diagnostics": diagnostic_rows,
                "failures": failures, "labels_opened": False, "code_commit": code_commit}
    publish_json_no_replace(args.output_root / "inference_manifests" /
                            f"worker_{args.worker_index:02d}_of_{args.num_workers:02d}.json", manifest)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
