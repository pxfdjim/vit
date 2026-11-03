#!/usr/bin/env python3
"""Generate fixed-weight AMR feature anomaly scores without opening labels."""
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
from experiments.temporal_matching.runner import publish_json_no_replace, publish_npz_no_replace
from src.models.amr_feature_scoring import (
    fuse_pair, fuse_single, reference_volatility_ratio, volatility_correct,
)
from src.models.bimodal_reference import load_model
from src.models.bimodal_reference_vit4ts import (
    _calibrate, _fitted_standardized_windows, _token_costs, _visual_costs,
)
from src.models.curve_aware_clip import (
    expected_window_count, input_digest, intermediate_cache_path,
    load_final_tokens, validate_intermediate_cache,
)
from src.models.fixed_reference_numeric_evidence import (
    align_point_score_to_t3, build_numeric_evidence, score_on_fixed_paths,
)
from src.models.numeric_temporal_matching import match_paths, mix_costs
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
    values = point_scores_from_maps(_column_map(local.double()), full_length=full_length, device=device)
    return np.asarray(values, np.float64)


def restore_aligned(local: torch.Tensor, base: np.ndarray, full_length: int,
                    device: torch.device) -> tuple[np.ndarray, float]:
    return align_point_score_to_t3(restore(local, full_length, device), base)


def read_score(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return stored["scores"].astype(np.float64), stored["timestamps"]


def write_score(path: Path, scores: np.ndarray, timestamps: np.ndarray, metadata: dict, elapsed: float):
    publish_npz_no_replace(
        path,
        scores=np.asarray(scores, np.float64), timestamps=np.asarray(timestamps),
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
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--manifest-tag", default="initial")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    amr_source = Path(config["amr_source_run"])
    numeric_source = Path(config["numeric_source_run"])
    stats = json.loads(Path(config["normalization_stats"]).read_text())
    protocol = config["fixed_protocol"]
    tasks = [(dataset, signal) for dataset in DATASETS for signal in signal_names(args.data_root, dataset)]
    tasks = tasks[args.worker_index::args.num_workers]
    device = torch.device(args.device)
    current_dataset, model = None, None
    records, failures = [], 0
    code_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=ROOT, text=True).strip()
    weights = [0.05, 0.10, 0.20]

    for ordinal, (dataset, signal) in enumerate(tasks, 1):
        started = time.perf_counter()
        try:
            expected = [args.output_root / "point_scores" / dataset / signal / f"{variant}.npz"
                        for variant in config["variants_order"]]
            diagnostic = args.output_root / "feature_diagnostics" / dataset / signal / "evidence.npz"
            if args.skip_complete and diagnostic.is_file() and all(path.is_file() for path in expected):
                records.append({"dataset": dataset, "signal": signal, "status": "skipped_complete",
                                "labels_opened": False})
                continue

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

            fitted = stats[dataset][signal]
            start_index = int(fitted["fit_points"]) if dataset in {"MSL", "SMAP"} else 0
            windows = _fitted_standardized_windows(raw, fitted, start_index).to(device)
            raw_evidence = build_numeric_evidence(windows, pool)
            if current_dataset != dataset:
                if model is not None:
                    del model
                    torch.cuda.empty_cache()
                model, _ = load_model(config["checkpoints"][dataset], device)
                current_dataset = dataset
            output = model(windows, patch)
            numeric_costs, numeric_cost_scale = _calibrate(
                visual_costs, _token_costs(output["numeric"], pool))
            cross_costs, cross_cost_scale = _calibrate(
                visual_costs, _token_costs(output["fused"], pool))
            selector_weight = float(protocol["amr_selector_weight"])
            late_costs = mix_costs(visual_costs, numeric_costs, selector_weight)
            amr_selector_costs = mix_costs(visual_costs, cross_costs, selector_weight)
            amr_match = match_paths(amr_selector_costs, pool, band=int(protocol["dtw_band"]),
                                    penalty_scale=float(protocol["dtw_penalty_scale"]))

            source_t3, source_time = read_score(amr_source / "point_scores" / dataset / signal / "T3.npz")
            source_amr, amr_time = read_score(amr_source / "point_scores" / dataset / signal / "AMR_native.npz")
            if not np.array_equal(source_time, timestamps) or not np.array_equal(amr_time, timestamps):
                raise RuntimeError("source timestamp mismatch")
            source_amr_diag = amr_source / "path_diagnostics" / dataset / signal / "AMR_native.npz"
            with np.load(source_amr_diag, allow_pickle=False) as stored:
                amr_reference_ids = stored["reference_window_ids"]
                amr_reference_columns = stored["reference_columns"]
                amr_visual_local = stored["visual_local_costs"]
            amr_reference_error = float(not np.array_equal(
                amr_match.reference_ids.cpu().numpy(), amr_reference_ids))
            amr_column_error = float(np.max(np.abs(
                amr_match.reference_columns.cpu().numpy() - amr_reference_columns)))
            computed_amr_visual = 0.5 * score_on_fixed_paths(visual_costs, amr_match).cpu().numpy()
            amr_visual_error = float(np.max(np.abs(computed_amr_visual - amr_visual_local)))
            if amr_reference_error or amr_column_error > 1e-6 or amr_visual_error > 1e-6:
                raise RuntimeError("AMR path reconstruction failed: "
                                   f"refs={amr_reference_error} cols={amr_column_error} visual={amr_visual_error}")

            t3_raw_local = {
                "level": score_on_fixed_paths(raw_evidence.level, t3_match),
                "dynamic": score_on_fixed_paths(raw_evidence.dynamic, t3_match),
                "combined": score_on_fixed_paths(raw_evidence.combined, t3_match),
            }
            amr_raw_local = {
                "level": score_on_fixed_paths(raw_evidence.level, amr_match),
                "dynamic": score_on_fixed_paths(raw_evidence.dynamic, amr_match),
                "combined": score_on_fixed_paths(raw_evidence.combined, amr_match),
            }
            numeric_local = score_on_fixed_paths(numeric_costs, t3_match)
            cross_local = score_on_fixed_paths(cross_costs, t3_match)
            late_local = score_on_fixed_paths(late_costs, t3_match)
            volatility_ratio, volatility_diagnostics = reference_volatility_ratio(
                windows, t3_match,
                minimum=float(protocol["reference_volatility_ratio_clip"][0]),
                maximum=float(protocol["reference_volatility_ratio_clip"][1]),
            )
            corrected_local = {
                "raw": volatility_correct(t3_raw_local["combined"], volatility_ratio),
                "numeric": volatility_correct(numeric_local, volatility_ratio),
                "cross": volatility_correct(cross_local, volatility_ratio),
                "late": volatility_correct(late_local, volatility_ratio),
            }

            point, point_scales = {}, {}
            for name, local in t3_raw_local.items():
                point[f"t3_raw_{name}"], point_scales[f"t3_raw_{name}"] = restore_aligned(
                    local, source_t3, len(raw), device)
            for name, local in amr_raw_local.items():
                point[f"amr_raw_{name}"], point_scales[f"amr_raw_{name}"] = restore_aligned(
                    local, source_amr, len(raw), device)
            for name, local in (("numeric", numeric_local), ("cross", cross_local), ("late", late_local)):
                point[name], point_scales[name] = restore_aligned(local, source_t3, len(raw), device)
            for name, local in corrected_local.items():
                point[f"corrected_{name}"], point_scales[f"corrected_{name}"] = restore_aligned(
                    local, source_t3, len(raw), device)

            source_e2 = {}
            for suffix in ("005", "010", "020"):
                source_e2[suffix], source_e2_time = read_score(
                    numeric_source / "point_scores" / dataset / signal / f"E2_C_combined_l{suffix}.npz")
                if not np.array_equal(source_e2_time, timestamps):
                    raise RuntimeError("numeric-source timestamp mismatch")
            e2_formula_error = max(float(np.max(np.abs(
                source_e2[f"{int(weight * 100):03d}"]
                - fuse_single(source_t3, point["t3_raw_combined"], weight)))) for weight in weights)
            if e2_formula_error > 1e-12:
                raise RuntimeError(f"E2 source formula mismatch: {e2_formula_error}")

            variants: dict[str, np.ndarray] = {
                "T3": source_t3,
                "AMR_native": source_amr,
                "E2_T3_raw_l005": source_e2["005"],
                "E2_T3_raw_l010": source_e2["010"],
                "E2_T3_raw_l020": source_e2["020"],
                "A_level_only": point["amr_raw_level"],
                "A_dynamic_only": point["amr_raw_dynamic"],
                "A_combined_only": point["amr_raw_combined"],
                "A_AMR_level_l010": fuse_single(source_amr, point["amr_raw_level"], 0.10),
                "A_AMR_dynamic_l010": fuse_single(source_amr, point["amr_raw_dynamic"], 0.10),
                "A_AMR_combined_l005": fuse_single(source_amr, point["amr_raw_combined"], 0.05),
                "A_AMR_combined_l010": fuse_single(source_amr, point["amr_raw_combined"], 0.10),
                "A_AMR_combined_l020": fuse_single(source_amr, point["amr_raw_combined"], 0.20),
                "B1_numeric_feature": point["numeric"],
            }
            for weight, suffix in zip(weights, ("005", "010", "020")):
                variants[f"B2_numeric_l{suffix}"] = fuse_single(source_t3, point["numeric"], weight)
                variants[f"B3_raw_numeric_l{suffix}"] = fuse_pair(
                    source_t3, point["t3_raw_combined"], point["numeric"], weight)
                variants[f"C1_cross_l{suffix}"] = fuse_single(source_t3, point["cross"], weight)
                variants[f"C2_raw_cross_l{suffix}"] = fuse_pair(
                    source_t3, point["t3_raw_combined"], point["cross"], weight)
                variants[f"C3_raw_late_l{suffix}"] = fuse_pair(
                    source_t3, point["t3_raw_combined"], point["late"], weight)
                variants[f"C4_raw_numeric_l{suffix}"] = fuse_pair(
                    source_t3, point["t3_raw_combined"], point["numeric"], weight)
                variants[f"D_E2_rawvol_l{suffix}"] = fuse_single(
                    source_t3, point["corrected_raw"], weight)
                variants[f"D_B2_numericvol_l{suffix}"] = fuse_single(
                    source_t3, point["corrected_numeric"], weight)
                variants[f"D_C2_raw_crossvol_l{suffix}"] = fuse_pair(
                    source_t3, point["corrected_raw"], point["corrected_cross"], weight)
                variants[f"D_C3_raw_latevol_l{suffix}"] = fuse_pair(
                    source_t3, point["corrected_raw"], point["corrected_late"], weight)
                variants[f"D_C4_raw_numericvol_l{suffix}"] = fuse_pair(
                    source_t3, point["corrected_raw"], point["corrected_numeric"], weight)
            variants = {name: variants[name] for name in config["variants_order"]}
            duplicate_error = max(float(np.max(np.abs(
                variants[f"B3_raw_numeric_l{suffix}"] - variants[f"C4_raw_numeric_l{suffix}"])))
                                  for suffix in ("005", "010", "020"))
            if duplicate_error != 0:
                raise RuntimeError(f"B3/C4 controlled duplicate mismatch: {duplicate_error}")

            elapsed = time.perf_counter() - started
            metadata_common = {
                "labels_opened": False, "checkpoint_protocol": "dataset_native",
                "training_dataset": dataset, "source_amr_run": str(amr_source),
                "source_numeric_run": str(numeric_source), "input_digest": digest[:20],
                "final_cache_path": str(final_path), "block_cache_path": str(block_path),
                "numeric_token_cost_scale": numeric_cost_scale,
                "cross_token_cost_scale": cross_cost_scale,
                "point_scales": point_scales, "volatility_diagnostics": volatility_diagnostics,
                "code_commit": code_commit,
            }
            for variant, scores in variants.items():
                uses_amr_path = variant.startswith("A_")
                write_score(args.output_root / "point_scores" / dataset / signal / f"{variant}.npz",
                            scores, timestamps,
                            metadata_common | {"variant": variant,
                                               "reference_path": "AMR_native" if uses_amr_path else "T3"},
                            elapsed)
            publish_npz_no_replace(
                diagnostic,
                t3_reference_window_ids=t3_match.reference_ids.cpu().numpy(),
                t3_reference_columns=t3_match.reference_columns.cpu().numpy(),
                t3_path_masks=t3_match.path_masks.cpu().numpy().astype(np.uint8),
                amr_reference_window_ids=amr_match.reference_ids.cpu().numpy(),
                amr_reference_columns=amr_match.reference_columns.cpu().numpy(),
                amr_path_masks=amr_match.path_masks.cpu().numpy().astype(np.uint8),
                t3_raw_level_local=t3_raw_local["level"].cpu().numpy(),
                t3_raw_dynamic_local=t3_raw_local["dynamic"].cpu().numpy(),
                t3_raw_combined_local=t3_raw_local["combined"].cpu().numpy(),
                amr_raw_combined_local=amr_raw_local["combined"].cpu().numpy(),
                numeric_feature_local=numeric_local.cpu().numpy(),
                cross_feature_local=cross_local.cpu().numpy(),
                late_feature_local=late_local.cpu().numpy(),
                reference_volatility_ratio=volatility_ratio.cpu().numpy(),
                corrected_raw_local=corrected_local["raw"].cpu().numpy(),
                corrected_numeric_local=corrected_local["numeric"].cpu().numpy(),
                corrected_cross_local=corrected_local["cross"].cpu().numpy(),
                corrected_late_local=corrected_local["late"].cpu().numpy(),
                point_raw_t3=point["t3_raw_combined"],
                point_numeric_feature=point["numeric"], point_cross_feature=point["cross"],
                point_late_feature=point["late"], point_corrected_raw=point["corrected_raw"],
                point_corrected_numeric=point["corrected_numeric"],
                point_corrected_cross=point["corrected_cross"],
                point_corrected_late=point["corrected_late"],
            )
            records.append({
                "dataset": dataset, "signal": signal, "status": "passed",
                "points": len(raw), "windows": len(windows),
                "elapsed_seconds": time.perf_counter() - started,
                "amr_reference_error": amr_reference_error,
                "amr_column_max_error": amr_column_error,
                "amr_visual_local_max_error": amr_visual_error,
                "e2_formula_max_error": e2_formula_error,
                "b3_c4_duplicate_max_error": duplicate_error,
                "mean_reference_volatility_ratio": volatility_diagnostics["mean_ratio"],
                "labels_opened": False,
            })
            print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s", flush=True)
        except Exception as error:
            failures += 1
            records.append({"dataset": dataset, "signal": signal, "status": "failed",
                            "error": f"{type(error).__name__}: {error}", "labels_opened": False})
            print(f"[{ordinal}/{len(tasks)}] FAILED {dataset}/{signal}: {error}", flush=True)

    manifest = {
        "status": "passed" if failures == 0 else "failed",
        "worker_index": args.worker_index, "num_workers": args.num_workers,
        "records": records, "failures": failures, "labels_opened": False,
        "code_commit": code_commit,
    }
    publish_json_no_replace(
        args.output_root / "inference_manifests" /
        f"worker_{args.worker_index:02d}_of_{args.num_workers:02d}_{args.manifest_tag}.json", manifest)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
