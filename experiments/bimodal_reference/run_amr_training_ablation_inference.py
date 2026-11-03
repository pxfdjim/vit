#!/usr/bin/env python3
"""Fixed-T3-path B3 scoring for AMR training ablation checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import DATASETS, evaluation_frame, signal_names
from experiments.bimodal_reference.run_amr_feature_scoring import (
    load_caches, read_score, restore_aligned, write_score,
)
from experiments.temporal_matching.runner import publish_json_no_replace, publish_npz_no_replace
from src.models.amr_feature_scoring import fuse_pair
from src.models.amr_training_ablation import AblationReferenceModel
from src.models.bimodal_reference_vit4ts import (
    _calibrate, _fitted_standardized_windows, _token_costs, _visual_costs,
)
from src.models.fixed_reference_numeric_evidence import score_on_fixed_paths
from src.models.numeric_temporal_matching import match_paths
from src.models.vfg_paired_vit4ts import _patch_tokens, select_common_visual_windows


MODEL_VARIANTS = (
    "T1_random", "T2_numeric_ssl", "Full", "C1_full", "C2_no_cross",
    "C3_late_joint", "C4_numeric_only", "L1_no_patch",
    "L2_no_temporal_relation", "L3_no_transition_geometry",
    "L4_no_reference", "L5_no_alignment",
)


def checkpoint_path(config: dict, output_root: Path, variant: str, dataset: str) -> Path:
    if variant in {"Full", "C1_full"}:
        return Path(config["full_checkpoint_root"]) / f"{dataset}.pt"
    if variant == "C4_numeric_only":
        variant = "T2_numeric_ssl"
    return output_root / "checkpoints" / variant / f"{dataset}.pt"


def load_ablation(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = payload["model_config"]
    architecture = cfg.get("architecture", "cross")
    model = AblationReferenceModel(architecture, hidden=int(cfg["hidden"]),
                                   heads=int(cfg["heads"]), layers=int(cfg.get("layers", 2)))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.reshape(-1, left.shape[-1]).float()
    y = right.reshape(-1, right.shape[-1]).float()
    x = x - x.mean(0, keepdim=True); y = y - y.mean(0, keepdim=True)
    cross = (x.T @ y).square().sum()
    denom = ((x.T @ x).square().sum() * (y.T @ y).square().sum()).sqrt().clamp_min(1e-12)
    return float((cross / denom).cpu())


def fused_tokens(model: AblationReferenceModel, numeric: torch.Tensor,
                 patch: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if model.architecture == "numeric":
        return None, None
    visual = model.visual(patch)
    if model.architecture == "cross":
        nv = model.n_to_v(numeric, visual, visual, need_weights=False)[0]
        vn = model.v_to_n(visual, numeric, numeric, need_weights=False)[0]
        fused = F.normalize(model.fuse(torch.cat((numeric, visual, nv, vn), dim=-1)), dim=-1)
    elif model.architecture == "late":
        fused = F.normalize(model.fuse(torch.cat((numeric, visual), dim=-1)), dim=-1)
    else:
        fused = None
    return visual, fused


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--stage", choices=("initial", "all"), required=True)
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_root = Path(config["data_root"])
    feature_root = Path(config["feature_scoring_run"])
    stats = json.loads(Path(config["normalization_stats"]).read_text())
    device = torch.device(args.device)
    variants = config["initial_variants"] if args.stage == "initial" else config["all_variants"]
    learned = [name for name in variants if name in MODEL_VARIANTS]
    tasks = [(dataset, signal) for dataset in DATASETS for signal in signal_names(data_root, dataset)]
    tasks = tasks[args.worker_index::args.num_workers]
    current_dataset = None; models = {}; payloads = {}
    records, similarities, failures = [], [], 0
    code_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=ROOT, text=True).strip()
    for ordinal, (dataset, signal) in enumerate(tasks, 1):
        started = time.perf_counter()
        expected = [args.output_root / "point_scores" / dataset / signal / f"{name}.npz"
                    for name in variants]
        diagnostic = args.output_root / "feature_diagnostics" / dataset / signal / "numeric_scores.npz"
        if args.skip_complete and diagnostic.is_file() and all(path.is_file() for path in expected):
            records.append({"dataset": dataset, "signal": signal, "status": "skipped_complete"})
            continue
        try:
            frame = evaluation_frame(data_root, dataset, signal)
            raw = frame.value.to_numpy(np.float64); timestamps = frame.timestamp.to_numpy()
            context = {"dataset": dataset, "signal": signal}
            cache_cfg = {"cache": config["cache"]}
            final, block, final_path, block_path, digest = load_caches(raw, context, cache_cfg)
            patch = _patch_tokens(block["intermediate_patch"]).to(device)
            pool = select_common_visual_windows(_patch_tokens(final["patch"]).to(device))
            visual_costs = _visual_costs(patch.reshape(len(patch), 14, 14, -1), pool)
            t3_match = match_paths(visual_costs, pool, band=2, penalty_scale=0.1)
            source_diag = feature_root / "feature_diagnostics" / dataset / signal / "evidence.npz"
            with np.load(source_diag, allow_pickle=False) as stored:
                expected_ids = stored["t3_reference_window_ids"]
                expected_columns = stored["t3_reference_columns"]
                raw_point = stored["point_raw_t3"].astype(np.float64)
            if not np.array_equal(t3_match.reference_ids.cpu().numpy(), expected_ids):
                raise RuntimeError("T3 reference reconstruction mismatch")
            if not np.array_equal(t3_match.reference_columns.cpu().numpy(), expected_columns):
                raise RuntimeError("T3 path reconstruction mismatch")
            fitted = stats[dataset][signal]
            start_index = int(fitted["fit_points"]) if dataset in {"MSL", "SMAP"} else 0
            windows = _fitted_standardized_windows(raw, fitted, start_index).to(device)
            source_t3, source_time = read_score(feature_root / "point_scores" / dataset / signal / "T3.npz")
            source_e2, e2_time = read_score(feature_root / "point_scores" / dataset / signal /
                                            "E2_T3_raw_l010.npz")
            source_b3, b3_time = read_score(feature_root / "point_scores" / dataset / signal /
                                            "B3_raw_numeric_l010.npz")
            if not (np.array_equal(source_time, timestamps) and np.array_equal(e2_time, timestamps)
                    and np.array_equal(b3_time, timestamps)):
                raise RuntimeError("source timestamp mismatch")
            if current_dataset != dataset:
                models.clear(); payloads.clear()
                for variant in learned:
                    canonical = "Full" if variant == "C1_full" else (
                        "T2_numeric_ssl" if variant == "C4_numeric_only" else variant)
                    if canonical not in models:
                        models[canonical], payloads[canonical] = load_ablation(
                            checkpoint_path(config, args.output_root, variant, dataset), device)
                current_dataset = dataset
            numeric_tokens = {}; learned_points = {}; scales = {}; timings = {}
            for variant in learned:
                canonical = "Full" if variant == "C1_full" else (
                    "T2_numeric_ssl" if variant == "C4_numeric_only" else variant)
                if canonical in numeric_tokens:
                    numeric_tokens[variant] = numeric_tokens[canonical]
                    learned_points[variant] = learned_points[canonical]
                    scales[variant] = scales[canonical]; timings[variant] = timings[canonical]
                    continue
                model = models[canonical]
                synchronize(device); tick = time.perf_counter()
                with torch.inference_mode():
                    tokens = model.numeric(windows)
                synchronize(device); encode_seconds = time.perf_counter() - tick
                costs, scale = _calibrate(visual_costs, _token_costs(tokens, pool))
                local = score_on_fixed_paths(costs, t3_match)
                point, point_scale = restore_aligned(local, source_t3, len(raw), device)
                numeric_tokens[canonical] = tokens
                learned_points[canonical] = point
                scales[canonical] = {"token_cost_scale": scale, "point_scale": point_scale}
                timings[canonical] = encode_seconds
                numeric_tokens[variant] = tokens; learned_points[variant] = point
                scales[variant] = scales[canonical]; timings[variant] = encode_seconds
            full_tokens = numeric_tokens["Full"]
            similarity_rows = []
            for variant in learned:
                canonical = "Full" if variant == "C1_full" else (
                    "T2_numeric_ssl" if variant == "C4_numeric_only" else variant)
                model = models[canonical]; tokens = numeric_tokens[variant]
                with torch.inference_mode():
                    visual, fused = fused_tokens(model, tokens, patch)
                row = {"dataset": dataset, "signal": signal, "variant": variant,
                       "numeric_cka_with_full": linear_cka(tokens, full_tokens),
                       "numeric_visual_cosine": np.nan, "numeric_fused_cosine": np.nan,
                       "fused_visual_cosine": np.nan}
                if visual is not None:
                    row["numeric_visual_cosine"] = float((tokens * visual).sum(-1).mean().cpu())
                if fused is not None:
                    row["numeric_fused_cosine"] = float((tokens * fused).sum(-1).mean().cpu())
                    row["fused_visual_cosine"] = float((fused * visual).sum(-1).mean().cpu())
                similarity_rows.append(row)
            output_scores = {"T3": source_t3, "E2": source_e2, "B3": source_b3}
            for variant in learned:
                output_scores[variant] = fuse_pair(source_t3, raw_point, learned_points[variant], 0.10)
            full_reproduction = float(np.max(np.abs(output_scores["Full"] - source_b3)))
            if full_reproduction > 1e-12:
                raise RuntimeError(f"Full B3 reproduction error: {full_reproduction}")
            for variant in variants:
                metadata = {"variant": variant, "labels_opened": False,
                            "fixed_reference": "T3", "fixed_dtw_path": "T3",
                            "visual_weight": 0.9, "raw_numeric_weight": 0.05,
                            "learned_numeric_weight": 0.05, "code_commit": code_commit,
                            "input_digest": digest[:20], "full_b3_reproduction_error": full_reproduction}
                if variant in learned:
                    metadata.update({"checkpoint": str(checkpoint_path(config, args.output_root,
                                                                        variant, dataset)),
                                     "numeric_encoding_seconds": timings[variant],
                                     **scales[variant]})
                write_score(args.output_root / "point_scores" / dataset / signal / f"{variant}.npz",
                            output_scores[variant], timestamps, metadata,
                            time.perf_counter() - started)
            publish_npz_no_replace(
                diagnostic,
                point_raw_t3=raw_point,
                **{f"point_numeric__{name}": learned_points[name] for name in learned},
            )
            similarities.extend(similarity_rows)
            records.append({"dataset": dataset, "signal": signal, "status": "passed",
                            "points": len(raw), "windows": len(windows),
                            "elapsed_seconds": time.perf_counter() - started,
                            "full_b3_reproduction_error": full_reproduction,
                            "labels_opened": False,
                            "numeric_encoding_seconds": timings})
            print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s", flush=True)
        except Exception as error:
            failures += 1
            records.append({"dataset": dataset, "signal": signal, "status": "failed",
                            "error": f"{type(error).__name__}: {error}", "labels_opened": False})
            print(f"FAILED {dataset}/{signal}: {error}", flush=True)
    manifest = {"status": "passed" if failures == 0 else "failed", "stage": args.stage,
                "worker_index": args.worker_index, "num_workers": args.num_workers,
                "records": records, "feature_similarity": similarities,
                "failures": failures, "labels_opened": False, "code_commit": code_commit}
    publish_json_no_replace(args.output_root / "inference_manifests" /
                            f"worker_{args.worker_index:02d}_of_{args.num_workers:02d}_{args.stage}.json",
                            manifest)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
