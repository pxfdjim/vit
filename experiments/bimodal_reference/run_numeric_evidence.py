#!/usr/bin/env python3
"""Generate fixed-reference numeric evidence without opening anomaly labels."""
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

from experiments.bimodal_reference.orionbench_protocol import (
    DATASETS, evaluation_frame, signal_names,
)
from experiments.temporal_matching.runner import publish_json_no_replace, publish_npz_no_replace
from src.models.bimodal_reference import load_model
from src.models.bimodal_reference_vit4ts import (
    _calibrate, _fitted_standardized_windows, _token_costs, _visual_costs,
)
from src.models.curve_aware_clip import (
    expected_window_count, input_digest, intermediate_cache_path,
    load_final_tokens, validate_intermediate_cache,
)
from src.models.fixed_reference_numeric_evidence import (
    align_point_score_to_t3, build_numeric_evidence, candidate_reliability,
    score_on_fixed_paths,
)
from src.models.numeric_temporal_matching import match_paths
from src.models.temporal_vit4ts import _column_map
from src.models.vfg_paired_vit4ts import (
    _patch_tokens, point_scores_from_maps, select_common_visual_windows,
)


def load_caches(raw: np.ndarray, context: dict, config: dict):
    digest = input_digest(raw)
    windows = expected_window_count(len(raw))
    _, final_path = load_final_tokens([Path(x) for x in config["cache"]["final_cache_roots"]],
                                     context, digest, windows)
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    candidates = [intermediate_cache_path(Path(root), context, digest)
                  for root in config["cache"]["block_cache_roots"]]
    block_path = next((path for path in candidates if path.is_file()), None)
    if block_path is None:
        raise FileNotFoundError(f"missing Block-6 cache: {candidates}")
    block = torch.load(block_path, map_location="cpu", weights_only=True)
    validate_intermediate_cache(block, digest=digest, expected_windows=windows, path=block_path)
    return final, block, final_path, block_path, digest


def restore(local: torch.Tensor, full_length: int, device: torch.device) -> np.ndarray:
    return np.asarray(point_scores_from_maps(_column_map(local.double()), full_length=full_length,
                                             device=device), np.float64)


def point_gate(window_gate: np.ndarray, full_length: int, device: torch.device) -> np.ndarray:
    columns = torch.from_numpy(window_gate.astype(np.float64))[:, None].expand(-1, 14)
    return np.clip(restore(columns, full_length, device), 0.0, 1.0)


def write_score(path: Path, scores: np.ndarray, timestamps: np.ndarray, metadata: dict, elapsed: float):
    publish_npz_no_replace(
        path,
        scores=np.asarray(scores, np.float64),
        timestamps=np.asarray(timestamps),
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
        timing_json=np.asarray([json.dumps({"inference_seconds": elapsed}, sort_keys=True)]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_root = Path(config["source_run"])
    stats = json.loads(Path(config["normalization_stats"]).read_text())
    protocol = config["fixed_protocol"]
    tasks = [(dataset, signal) for dataset in DATASETS for signal in signal_names(args.data_root, dataset)]
    tasks = tasks[args.worker_index::args.num_workers]
    device = torch.device(args.device)
    current_dataset, model = None, None
    records, failures = [], 0
    code_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=ROOT, text=True).strip()
    for ordinal, (dataset, signal) in enumerate(tasks, 1):
        started = time.perf_counter()
        try:
            frame = evaluation_frame(args.data_root, dataset, signal)
            raw = frame.value.to_numpy(np.float64)
            timestamps = frame.timestamp.to_numpy()
            context = {"dataset": dataset, "signal": signal}
            final, block, final_path, block_path, digest = load_caches(raw, context, config)
            patch = _patch_tokens(block["intermediate_patch"]).to(device)
            pool = select_common_visual_windows(_patch_tokens(final["patch"]).to(device))
            visual_costs = _visual_costs(patch.reshape(len(patch), 14, 14, -1), pool)
            t3_match = match_paths(visual_costs, pool, band=int(protocol["dtw_band"]),
                                   penalty_scale=float(protocol["dtw_penalty_scale"]))
            source_diag_path = source_root / "path_diagnostics" / dataset / signal / "T3.npz"
            with np.load(source_diag_path, allow_pickle=False) as source_diag:
                source_refs = source_diag["reference_window_ids"]
                source_columns = source_diag["reference_columns"]
                source_local = source_diag["visual_local_costs"]
            reference_match = np.array_equal(t3_match.reference_ids.cpu().numpy(), source_refs)
            column_error = float(np.max(np.abs(t3_match.reference_columns.cpu().numpy() - source_columns)))
            local_error = float(np.max(np.abs(0.5 * t3_match.scores.cpu().numpy() - source_local)))
            if not reference_match or column_error > 1e-6 or local_error > 1e-6:
                raise RuntimeError(f"T3 path audit failed refs={reference_match} columns={column_error} local={local_error}")

            fitted = stats[dataset][signal]
            start_index = int(fitted["fit_points"]) if dataset in {"MSL", "SMAP"} else 0
            windows = _fitted_standardized_windows(raw, fitted, start_index).to(device)
            evidence = build_numeric_evidence(windows, pool)
            numeric_local = {
                "level": score_on_fixed_paths(evidence.level, t3_match),
                "dynamic": score_on_fixed_paths(evidence.dynamic, t3_match),
                "combined": score_on_fixed_paths(evidence.combined, t3_match),
            }
            source_t3_path = source_root / "point_scores" / dataset / signal / "T3.npz"
            source_amr_path = source_root / "point_scores" / dataset / signal / "AMR_native.npz"
            with np.load(source_t3_path, allow_pickle=False) as stored:
                t3_score = stored["scores"].astype(np.float64)
            with np.load(source_amr_path, allow_pickle=False) as stored:
                amr_score = stored["scores"].astype(np.float64)
            aligned, point_scales = {}, {}
            for name, local in numeric_local.items():
                numeric_point = restore(local, len(raw), device)
                aligned[name], point_scales[name] = align_point_score_to_t3(numeric_point, t3_score)

            if current_dataset != dataset:
                if model is not None:
                    del model
                    torch.cuda.empty_cache()
                model, _ = load_model(config["checkpoints"][dataset], device)
                current_dataset = dataset
            output = model(windows, patch)
            learned_numeric, numeric_token_scale = _calibrate(visual_costs, _token_costs(output["numeric"], pool))
            fused, fused_token_scale = _calibrate(visual_costs, _token_costs(output["fused"], pool))
            gate_stat, gate_cross, gate_late, gate_details = candidate_reliability(
                visual_costs, evidence.combined, learned_numeric, fused, pool,
                top_k=int(protocol["top_k"]),
                penalty_scale=float(protocol["dtw_penalty_scale"]),
                late_weight=float(protocol["late_joint_weight"]),
            )
            point_gates = {
                "stat": point_gate(gate_stat, len(raw), device),
                "cross": point_gate(gate_cross, len(raw), device),
                "late": point_gate(gate_late, len(raw), device),
            }
            lam = float(protocol["e3_lambda"])
            variants = {
                "T3": t3_score,
                "AMR_native": amr_score,
                "E1_A_level": aligned["level"],
                "E1_B_dynamic": aligned["dynamic"],
                "E1_C_combined": aligned["combined"],
                "E2_A_level_l010": 0.9 * t3_score + 0.1 * aligned["level"],
                "E2_B_dynamic_l010": 0.9 * t3_score + 0.1 * aligned["dynamic"],
                "E2_C_combined_l005": 0.95 * t3_score + 0.05 * aligned["combined"],
                "E2_C_combined_l010": 0.9 * t3_score + 0.1 * aligned["combined"],
                "E2_C_combined_l020": 0.8 * t3_score + 0.2 * aligned["combined"],
                "E3_A_stat_l010": (1.0 - lam * point_gates["stat"]) * t3_score +
                                   lam * point_gates["stat"] * aligned["combined"],
                "E3_B_cross_l010": (1.0 - lam * point_gates["cross"]) * t3_score +
                                    lam * point_gates["cross"] * aligned["combined"],
                "E3_C_late_l010": (1.0 - lam * point_gates["late"]) * t3_score +
                                   lam * point_gates["late"] * aligned["combined"],
            }
            if list(variants) != config["variants_order"]:
                raise RuntimeError("variant order differs from preregistered configuration")
            shared_elapsed = time.perf_counter() - started
            metadata_common = {
                "fixed_t3_reference_and_path": True,
                "labels_opened": False,
                "checkpoint_protocol": "dataset_native",
                "training_dataset": dataset,
                "numeric_component_scales": evidence.scales,
                "numeric_point_to_t3_scales": point_scales,
                "numeric_token_to_visual_scale": numeric_token_scale,
                "fused_token_to_visual_scale": fused_token_scale,
                "source_run": str(source_root),
                "input_digest": digest[:20],
                "final_cache_path": str(final_path),
                "block_cache_path": str(block_path),
                "code_commit": code_commit,
            }
            for variant, scores in variants.items():
                write_score(args.output_root / "point_scores" / dataset / signal / f"{variant}.npz",
                            scores, timestamps, metadata_common | {"variant": variant}, shared_elapsed)
            publish_npz_no_replace(
                args.output_root / "evidence_diagnostics" / dataset / signal / "fixed_t3_evidence.npz",
                reference_window_ids=t3_match.reference_ids.cpu().numpy(),
                reference_columns=t3_match.reference_columns.cpu().numpy(),
                path_masks=t3_match.path_masks.cpu().numpy().astype(np.uint8),
                visual_local_costs=(0.5 * t3_match.scores).cpu().numpy(),
                numeric_level_local=numeric_local["level"].cpu().numpy(),
                numeric_dynamic_local=numeric_local["dynamic"].cpu().numpy(),
                numeric_combined_local=numeric_local["combined"].cpu().numpy(),
                gate_stat_window=gate_stat, gate_cross_window=gate_cross, gate_late_window=gate_late,
                gate_stat_point=point_gates["stat"], gate_cross_point=point_gates["cross"],
                gate_late_point=point_gates["late"],
                **gate_details,
            )
            records.append({"dataset": dataset, "signal": signal, "status": "passed",
                            "points": len(raw), "windows": len(windows),
                            "elapsed_seconds": time.perf_counter() - started,
                            "path_reference_match": reference_match,
                            "path_column_max_error": column_error,
                            "visual_local_max_error": local_error,
                            "mean_gate_stat": float(gate_stat.mean()),
                            "mean_gate_cross": float(gate_cross.mean()),
                            "mean_gate_late": float(gate_late.mean()),
                            "labels_opened": False})
            print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s", flush=True)
        except Exception as error:
            failures += 1
            records.append({"dataset": dataset, "signal": signal, "status": "failed",
                            "error": f"{type(error).__name__}: {error}", "labels_opened": False})
            print(f"[{ordinal}/{len(tasks)}] FAILED {dataset}/{signal}: {error}", flush=True)
    manifest = {"status": "passed" if failures == 0 else "failed",
                "worker_index": args.worker_index, "num_workers": args.num_workers,
                "records": records, "failures": failures, "labels_opened": False,
                "code_commit": code_commit}
    publish_json_no_replace(args.output_root / "inference_manifests" /
                            f"worker_{args.worker_index:02d}_of_{args.num_workers:02d}.json", manifest)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
