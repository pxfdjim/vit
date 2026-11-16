#!/usr/bin/env python3
"""Run reference-prior correction using frozen B3 and stored T3 paths."""
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
                                     write_score)
from src.core.visual_features import (expected_window_count, input_digest,
                                      intermediate_cache_path,
                                      validate_intermediate_cache)
from src.core.reference import fitted_standardized_windows, patch_tokens
from src.core.scoring import calibrated_delta
from src.models.model import load_model_checkpoint


VARIANTS = ("B3", "Model")


def load_visual_features(raw: np.ndarray, context: dict, config: dict):
    digest = input_digest(raw)
    windows = expected_window_count(len(raw))
    candidates = [intermediate_cache_path(Path(root), context, digest)
                  for root in config["visual_features"]["block_cache_roots"]]
    block_path = next((path for path in candidates if path.is_file()), None)
    if block_path is None:
        raise FileNotFoundError(f"missing Block-6 cache: {candidates}")
    block = torch.load(block_path, map_location="cpu", weights_only=True)
    validate_intermediate_cache(block, digest=digest, expected_windows=windows, path=block_path)
    return block, digest


def signed_calibration(local, base, full_length, device):
    if len(base) != full_length:
        raise ValueError("base score length does not match the input")
    correction, scale = calibrated_delta(local, torch.as_tensor(base, device=device))
    return correction.detach().cpu().numpy(), scale


def stored_t3_reference(path: Path, windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the immutable T3 selector output without repeating its DTW search."""
    with np.load(path, allow_pickle=False) as stored:
        ids = stored["t3_reference_window_ids"].astype(np.int64)
        columns = stored["t3_reference_columns"].astype(np.float32)
        path_masks = stored["t3_path_masks"].astype(bool)
    if ids.shape != (windows,) or columns.shape != (windows, 14) or \
            path_masks.shape != (windows, 14, 14):
        raise ValueError("stored T3 path shape differs from visual windows")
    if np.any(ids < 0) or np.any(ids >= windows) or \
            np.any(np.abs(ids - np.arange(windows)) < 4):
        raise ValueError("stored T3 reference is missing or overlaps its query")
    if not np.isfinite(columns).all() or np.any(columns < 0) or np.any(columns > 13) or \
            np.any(np.diff(columns, axis=1) < -1e-5):
        raise ValueError("stored T3 column centers are invalid")
    if not np.all(path_masks[:, 0, 0]) or not np.all(path_masks[:, -1, -1]):
        raise ValueError("stored T3 paths do not have fixed endpoints")
    return torch.from_numpy(ids), torch.from_numpy(np.rint(columns).astype(np.float32))


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
        anchor = reference_columns[start:stop]
        output = model.forward_tokens(numeric[start:stop], visual[start:stop],
                                      numeric[ids], visual[ids], anchor)
        alignment = output["alignment"]
        deltas.append(output["delta"])
        match = alignment["soft_match"]
        normalized = match / match.sum(-1, keepdim=True).clamp_min(1e-8)
        entropy = -(normalized.clamp_min(1e-8) * normalized.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / np.log(14)
        index = torch.arange(14, device=match.device, dtype=match.dtype)
        displacement = (normalized * index[None, None]).sum(-1) - anchor
        evidence = alignment["evidence"]
        diagnostics.append({
            "path_entropy": float(entropy.mean().cpu()),
            "match_probability": float(match.sum(-1).mean().cpu()),
            "rejection": float(alignment["rejection"].mean().cpu()),
            "reference_skip": float(alignment["reference_skip"].mean().cpu()),
            "reliability": float(alignment["reliability"].mean().cpu()),
            "alignment_displacement": float(displacement.abs().mean().cpu()),
            "numeric_matched_difference": float(evidence[..., 0].mean().cpu()),
            "visual_matched_difference": float(evidence[..., 1].mean().cpu()),
            "numeric_unmatched_difference": float(evidence[..., 2].mean().cpu()),
            "visual_unmatched_difference": float(evidence[..., 3].mean().cpu()),
            "numeric_structure": float(evidence[..., 4].mean().cpu()),
            "visual_structure": float(evidence[..., 5].mean().cpu()),
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
            block, digest = load_visual_features(
                raw, {"dataset": dataset, "signal": signal}, config)
            # The unlabeled synthetic training cache stores Block-6 patches in
            # float16; use the same quantization before the frozen projection.
            patch = patch_tokens(block["intermediate_patch"]).half().float().to(device)
            reference_ids, reference_columns = stored_t3_reference(
                feature_root / "feature_diagnostics" / dataset / signal / "evidence.npz",
                len(patch))
            reference_ids = reference_ids.to(device)
            reference_columns = reference_columns.to(device)
            fitted = stats[dataset][signal]
            start = int(fitted["fit_points"]) if dataset in {"MSL", "SMAP"} else 0
            windows = fitted_standardized_windows(raw, fitted, start).to(device)
            with np.load(feature_root / "point_scores" / dataset / signal /
                         "B3_raw_numeric_l010.npz", allow_pickle=False) as stored:
                b3 = stored["scores"].astype(np.float64)
                source_timestamps = stored["timestamps"]
                metadata = json.loads(str(stored["metadata_json"][0]))
            if metadata.get("input_digest") != digest[:20]:
                raise RuntimeError("frozen B3 input digest mismatch")
            if not np.array_equal(source_timestamps, timestamps):
                raise RuntimeError("B3 timestamp mismatch")
            if loaded_dataset != dataset:
                model, _ = load_model_checkpoint(
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
            for name, score in (("B3", b3), ("Model", final_score)):
                write_score(args.output_root / "point_scores" / dataset / signal / f"{name}.npz",
                            score, timestamps,
                            {"variant": name, "labels_opened": False,
                             "fixed_reference": "T3", "dtw_prior": "T3",
                             "code_commit": commit, "input_digest": digest[:20],
                             "base": "B3" if name == "Model" else None,
                             "gamma": gamma if name == "Model" else None},
                            time.perf_counter() - tick)
            publish_npz_no_replace(diagnostic_path, point_delta=point_delta)
            records.append({"dataset": dataset, "signal": signal, "status": "passed",
                            "elapsed_seconds": time.perf_counter() - tick,
                            "labels_opened": False, "stored_t3_valid": True})
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
