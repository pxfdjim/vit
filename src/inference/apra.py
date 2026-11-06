#!/usr/bin/env python3
"""Run Detect-only APRA correction on frozen B3 scores and T3 paths."""
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

from src.evaluation.orionbench_protocol import DATASETS, evaluation_frame, signal_names
from src.inference.artifacts import (publish_json_no_replace, publish_npz_no_replace,
                                     read_score, write_score)
from src.apra.cache import (expected_window_count, input_digest, intermediate_cache_path,
                            load_final_tokens, validate_intermediate_cache)
from src.apra.reference import (column_map, fitted_standardized_windows, match_paths,
                                patch_tokens, point_scores_from_maps, positive_median,
                                select_reference_windows, visual_costs)
from src.models.apra import load_apra_checkpoint


VARIANTS = ("B3", "E6_Detect_only")


def load_caches(raw: np.ndarray, context: dict, config: dict):
    digest = input_digest(raw)
    windows = expected_window_count(len(raw))
    _, final_path = load_final_tokens([Path(path) for path in config["cache"]["final_cache_roots"]],
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
    values = point_scores_from_maps(column_map(local.double()), full_length=full_length,
                                    device=device)
    return np.asarray(values, np.float64)


def signed_calibration(local, base, full_length, device):
    point = restore(local, full_length, device)
    magnitude = np.abs(point[np.isfinite(point)])
    magnitude = magnitude[magnitude > 1e-12]
    if not len(magnitude):
        return np.zeros_like(point), 1.0
    scale = float(np.clip(positive_median(base) /
                         max(float(np.median(magnitude)), 1e-12), 1e-4, 1e4))
    return point * scale, scale


@torch.inference_mode()
def model_evidence(model, windows, patch, reference_ids, reference_columns, batch_size=256):
    numeric, visual = [], []
    for start in range(0, len(windows), batch_size):
        numeric.append(model.numeric(windows[start:start + batch_size]))
        visual.append(model.visual(patch[start:start + batch_size]))
    numeric = torch.cat(numeric)
    visual = torch.cat(visual)
    deltas, diagnostics = [], []
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        ids = reference_ids[start:stop]
        columns = reference_columns[start:stop][..., None]
        nr_full, vr_full = numeric[ids], visual[ids]
        nr = nr_full.gather(1, columns.expand(-1, -1, nr_full.shape[-1]))
        vr = vr_full.gather(1, columns.expand(-1, -1, vr_full.shape[-1]))
        output = model.forward_tokens(numeric[start:stop], visual[start:stop], nr, vr)
        alignment = output["alignment"]
        deltas.append(output["delta"])
        match = alignment["soft_match"]
        normalized = match / match.sum(-1, keepdim=True).clamp_min(1e-8)
        entropy = -(normalized.clamp_min(1e-8) * normalized.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / np.log(14)
        index = torch.arange(14, device=match.device, dtype=match.dtype)
        displacement = (normalized * index[None, None]).sum(-1) - index[None]
        diagnostics.append({
            "path_entropy": float(entropy.mean().cpu()),
            "match_probability": float(match.sum(-1).mean().cpu()),
            "rejection": float(alignment["rejection"].mean().cpu()),
            "reference_skip": float(alignment["reference_skip"].mean().cpu()),
            "reliability": float(alignment["reliability"].mean().cpu()),
            "alignment_displacement": float(displacement.abs().mean().cpu()),
            "unmatched_numeric": float(alignment["unmatched_numeric"].norm(dim=-1).mean().cpu()),
            "unmatched_visual": float(alignment["unmatched_visual"].norm(dim=-1).mean().cpu()),
            "structural_numeric": float(alignment["structural_numeric"].norm(dim=-1).mean().cpu()),
            "structural_visual": float(alignment["structural_visual"].norm(dim=-1).mean().cpu()),
            "delta_mean": float(output["delta"].mean().cpu()),
            "delta_std": float(output["delta"].std(unbiased=False).cpu()),
        })
    summary = {key: float(np.mean([row[key] for row in diagnostics])) for key in diagnostics[0]}
    return torch.cat(deltas), summary


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
    device = torch.device(args.device)
    data_root = Path(config["data_root"])
    feature_root = Path(config["feature_scoring_run"])
    stats = json.loads(Path(config["normalization_stats"]).read_text())
    datasets = args.datasets or list(DATASETS)
    tasks = [(dataset, signal) for dataset in datasets for signal in signal_names(data_root, dataset)]
    if args.signals:
        tasks = [task for task in tasks if task[1] in set(args.signals)]
    tasks = tasks[args.worker_index::args.num_workers]
    gamma = float(config["loss_weights"]["gamma"])
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    records, diagnostic_rows = [], []
    failures = 0
    loaded_dataset = None
    model = None

    for ordinal, (dataset, signal) in enumerate(tasks, 1):
        tick = time.perf_counter()
        expected = [args.output_root / "point_scores" / dataset / signal / f"{name}.npz"
                    for name in VARIANTS]
        diagnostic_path = args.output_root / "feature_diagnostics" / dataset / signal / "evidence.npz"
        if args.skip_complete and diagnostic_path.is_file() and all(path.is_file() for path in expected):
            records.append({"dataset": dataset, "signal": signal, "status": "skipped_complete"})
            continue
        try:
            frame = evaluation_frame(data_root, dataset, signal)
            raw = frame.value.to_numpy(np.float64)
            timestamps = frame.timestamp.to_numpy()
            final, block, _, _, digest = load_caches(
                raw, {"dataset": dataset, "signal": signal}, config)
            patch = patch_tokens(block["intermediate_patch"]).float().to(device)
            pool = select_reference_windows(patch_tokens(final["patch"]).to(device))
            costs = visual_costs(patch.reshape(len(patch), 14, 14, -1), pool)
            t3 = match_paths(costs, pool, band=2, penalty_scale=.1)
            with np.load(feature_root / "feature_diagnostics" / dataset / signal / "evidence.npz") as stored:
                if not np.array_equal(t3.reference_ids.cpu().numpy(), stored["t3_reference_window_ids"]):
                    raise RuntimeError("T3 reference mismatch")
                if not np.array_equal(t3.reference_columns.cpu().numpy(), stored["t3_reference_columns"]):
                    raise RuntimeError("T3 path mismatch")
            reference_ids = t3.reference_ids.long().to(device)
            reference_columns = t3.reference_columns.round().long().to(device)
            fitted = stats[dataset][signal]
            start = int(fitted["fit_points"]) if dataset in {"MSL", "SMAP"} else 0
            windows = fitted_standardized_windows(raw, fitted, start).to(device)
            b3, source_timestamps = read_score(
                feature_root / "point_scores" / dataset / signal / "B3_raw_numeric_l010.npz")
            if not np.array_equal(source_timestamps, timestamps):
                raise RuntimeError("B3 timestamp mismatch")
            if loaded_dataset != dataset:
                model, _ = load_apra_checkpoint(
                    Path(config["checkpoint_root"]) / f"{dataset}.pt", device)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                loaded_dataset = dataset
            model_tick = time.perf_counter()
            local, diagnostics = model_evidence(model, windows, patch, reference_ids, reference_columns)
            point_delta, scale = signed_calibration(local, b3, len(raw), device)
            final_score = b3 + gamma * point_delta
            diagnostics.update({"delta_calibration_scale": scale,
                                "inference_seconds": time.perf_counter() - model_tick})
            diagnostic_rows.append({"dataset": dataset, "signal": signal, **diagnostics})
            for name, score in (("B3", b3), ("E6_Detect_only", final_score)):
                write_score(args.output_root / "point_scores" / dataset / signal / f"{name}.npz",
                            score, timestamps,
                            {"variant": name, "labels_opened": False,
                             "fixed_reference": "T3", "fixed_dtw_anchor": "T3",
                             "code_commit": commit, "input_digest": digest[:20],
                             "base": "B3" if name == "E6_Detect_only" else None,
                             "gamma": gamma if name == "E6_Detect_only" else None},
                            time.perf_counter() - tick)
            publish_npz_no_replace(diagnostic_path, point_delta=point_delta)
            records.append({"dataset": dataset, "signal": signal, "status": "passed",
                            "elapsed_seconds": time.perf_counter() - tick,
                            "labels_opened": False, "t3_reference_error": 0, "t3_path_error": 0})
            print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s",
                  flush=True)
        except Exception as error:
            failures += 1
            records.append({"dataset": dataset, "signal": signal, "status": "failed",
                            "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {dataset}/{signal}: {error}", flush=True)
    publish_json_no_replace(
        args.output_root / "inference_manifests" /
        f"worker_{args.worker_index:02d}_of_{args.num_workers:02d}.json",
        {"status": "passed" if not failures else "failed", "records": records,
         "diagnostics": diagnostic_rows, "failures": failures,
         "labels_opened": False, "code_commit": commit})
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
