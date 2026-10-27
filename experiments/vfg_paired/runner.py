#!/usr/bin/env python3
"""Resumable three-worker runner for the paired-reference VFG experiment."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_DATA_ROOT = Path("/home/pengxf/jim/VLM4TS/data")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/runs/vfg_paired_20261027"
DEFAULT_BACKEND = "src.models.vfg_paired_vit4ts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def publish_bytes_no_replace(path: Path, payload: bytes) -> bool:
    """Atomically publish on this filesystem and never replace a target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def publish_json_no_replace(path: Path, value: Mapping[str, Any]) -> bool:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return publish_bytes_no_replace(path, text.encode("utf-8"))


def publish_npz_no_replace(path: Path, **arrays: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def discover_tasks(config: Mapping[str, Any], data_root: Path) -> list[tuple[str, str, Path]]:
    tasks = []
    for dataset in config["datasets"]:
        paths = sorted((data_root / dataset).glob("*.csv"))
        expected = int(config["expected_signal_counts"][dataset])
        if len(paths) != expected:
            raise RuntimeError(f"{dataset}: expected {expected} signals, found {len(paths)}")
        tasks.extend((dataset, path.stem, path) for path in paths)
    return tasks


def load_signal(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"timestamp", "value"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain timestamp and value")
    frame = frame[["timestamp", "value"]].sort_values("timestamp", kind="stable").reset_index(drop=True)
    if frame.empty or frame.isna().any().any() or not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"Empty or non-finite input: {path}")
    return frame


def load_ground_truth(data_root: Path) -> dict[str, list[list[Any]]]:
    result = {}
    with (data_root / "anomalies.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["signal"]] = ast.literal_eval(row["events"])
    return result


def load_backend(name: str) -> Any:
    return importlib.import_module(name)


def call_backend(
    module: Any,
    state: dict[str, Any],
    config: Mapping[str, Any],
    device: str,
    cache_root: Path,
    data: pd.DataFrame,
    variants: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """The only adapter to the evolving paired backend API."""
    if "instance" not in state:
        factory = getattr(module, "create_backend", None)
        if factory is None:
            raise AttributeError(f"{module.__name__} does not export create_backend")
        state["instance"] = factory(config=config, device=device, cache_dir=cache_root)
    return state["instance"].predict_variants(data=data, variants=variants, context=context)


def normalize_prediction(raw: Any, fallback_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    if not isinstance(raw, Mapping) or "scores" not in raw:
        raise TypeError("Backend prediction must map scores, optional timestamps/metadata/timing")
    scores = np.asarray(raw["scores"], dtype=np.float64).reshape(-1)
    timestamps = np.asarray(raw.get("timestamps", fallback_timestamps)).reshape(-1)
    if len(scores) != len(fallback_timestamps) or len(timestamps) != len(scores):
        raise ValueError("Backend point scores do not align with the input")
    if not np.isfinite(scores).all():
        raise ValueError("Backend returned non-finite scores")
    metadata = dict(raw.get("metadata", {}))
    timing = dict(raw.get("timing", {}))
    for key, value in metadata.items():
        if key.endswith("_seconds") and isinstance(value, (int, float)):
            timing.setdefault(key, float(value))
    if raw.get("inference_seconds") is not None:
        timing.setdefault("inference_seconds", float(raw["inference_seconds"]))
    for key in list(timing):
        timing[key] = float(timing[key])
    return scores, timestamps, metadata, timing


def load_score(path: Path, expected_length: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float], int | None]:
    with np.load(path, allow_pickle=False) as stored:
        scores = stored["scores"].astype(np.float64)
        timestamps = stored["timestamps"]
        metadata = json.loads(str(stored["metadata_json"][0]))
        timing = json.loads(str(stored["timing_json"][0]))
        peak = int(stored["peak_cuda_memory_bytes"][0])
    if len(scores) != expected_length or len(timestamps) != expected_length or not np.isfinite(scores).all():
        raise ValueError(f"Invalid existing artifact retained at {path}")
    return scores, timestamps, metadata, timing, None if peak < 0 else peak


def overlap(left: Iterable[Any], right: Iterable[Any]) -> bool:
    a, b = left
    c, d = right
    return not (b < c or d < a)


def evaluate(scores: np.ndarray, timestamps: np.ndarray, truth: list[list[Any]], evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    from src.models.model_utils import compute_detection_intervals

    rows = []
    for alpha in evaluation["alphas"]:
        indices, threshold, _ = compute_detection_intervals(
            score_vector=scores,
            alpha=float(alpha),
            method=evaluation["threshold_method"],
            smoothing=bool(evaluation["smoothing"]),
            sliding=bool(evaluation["sliding"]),
            anomaly_padding=int(evaluation["anomaly_padding"]),
        )
        detected = [[timestamps[start].item(), timestamps[end].item()] for start, end in indices]
        tp = sum(any(overlap(item, event) for event in truth) for item in detected)
        fp = sum(not any(overlap(item, event) for event in truth) for item in detected)
        fn = sum(not any(overlap(event, item) for item in detected) for event in truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"alpha": float(alpha), "threshold": float(threshold), "TP": int(tp), "FP": int(fp), "FN": int(fn), "precision": precision, "recall": recall, "f1": f1, "detected_intervals": detected})
    return rows


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    variants = config["variants_order"] if not args.variants else [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(config["variants"]))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    tasks_all = discover_tasks(config, args.data_root)
    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        raise ValueError("Require num_workers >= 1 and 0 <= worker_id < num_workers")
    tasks = tasks_all[args.worker_id::args.num_workers]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    truth = load_ground_truth(args.data_root)
    if any(signal not in truth for _, signal, _ in tasks_all):
        raise RuntimeError("One or more configured signals lack ground truth")
    random.seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    plan = {"total_signals": len(tasks_all), "worker_signals": len(tasks), "worker_id": args.worker_id, "num_workers": args.num_workers, "device": args.device, "cache_root": str(args.cache_root), "output_root": str(args.output_root), "variants": variants}
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        cache_files = 0
        for dataset, signal, path in tasks:
            length = len(load_signal(path))
            window_size = int(config["model"]["window_size"])
            step_size = int(config["model"]["step_size"])
            if length < window_size:
                raise ValueError(f"Signal shorter than one window: {path}")
            n_windows = 1 + (length - window_size) // step_size
            letters = {name.split("_", 1)[0].upper() for name in variants}
            required = 12 if "D" in letters else 8 if letters - {"A"} else 1
            if n_windows < required:
                raise ValueError(
                    f"{dataset}/{signal} has {n_windows} windows; requested variants "
                    f"require at least {required}"
                )
            matches = list((args.cache_root / dataset / signal).glob("visual_*.pt"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one existing visual cache for {dataset}/{signal}, found {len(matches)}")
            cache_files += 1
        print(f"Dry-run validated {len(tasks)} signals and {cache_files} cache paths without importing the backend.")
        return 0

    publish_bytes_no_replace(args.output_root / "config.json", (json.dumps(config, indent=2, sort_keys=True) + "\n").encode())
    backend = None
    state: dict[str, Any] = {}
    for ordinal, (dataset, signal, path) in enumerate(tasks, 1):
        data = load_signal(path)
        score_dir = args.output_root / "point_scores" / dataset / signal
        metric_dir = args.output_root / "metrics" / dataset / signal
        missing = [name for name in variants if not (score_dir / f"{name}.npz").exists()]
        predictions: Mapping[str, Any] = {}
        peak: int | None = None
        wall = 0.0
        if missing:
            backend = backend or load_backend(args.backend_module)
            cuda = None
            if args.device.startswith("cuda"):
                import torch
                cuda = torch.cuda
                cuda.set_device(torch.device(args.device)); cuda.reset_peak_memory_stats(torch.device(args.device))
            started = time.perf_counter()
            predictions = call_backend(backend, state, config, args.device, args.cache_root, data, {name: config["variants"][name] for name in missing}, {"dataset": dataset, "signal": signal, "source_path": str(path), "seed": int(config["seed"])})
            if cuda is not None:
                cuda.synchronize(torch.device(args.device)); peak = int(cuda.max_memory_allocated(torch.device(args.device)))
            wall = time.perf_counter() - started
            omitted = sorted(set(missing) - set(predictions))
            if omitted:
                raise RuntimeError(f"Backend omitted variants {omitted}")
        for variant in variants:
            score_path = score_dir / f"{variant}.npz"
            metric_path = metric_dir / f"{variant}.json"
            if metric_path.exists():
                document = load_json(metric_path)
                if (document.get("dataset"), document.get("signal"), document.get("variant")) != (dataset, signal, variant):
                    raise ValueError(f"Invalid existing metric retained at {metric_path}")
                load_score(score_path, len(data))
                print(f"[{ordinal}/{len(tasks)}] skip {dataset}/{signal}/{variant}")
                continue
            if score_path.exists():
                scores, timestamps, metadata, timing, peak_variant = load_score(score_path, len(data))
            else:
                scores, timestamps, metadata, timing = normalize_prediction(predictions[variant], data["timestamp"].to_numpy())
                timing.setdefault("inference_seconds", wall)
                peak_variant = int(metadata.get("peak_cuda_memory_bytes", peak)) if metadata.get("peak_cuda_memory_bytes", peak) is not None else None
                published = publish_npz_no_replace(score_path, scores=scores, timestamps=timestamps, metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]), timing_json=np.asarray([json.dumps(timing, sort_keys=True)]), peak_cuda_memory_bytes=np.asarray([-1 if peak_variant is None else peak_variant], dtype=np.int64))
                if not published:
                    scores, timestamps, metadata, timing, peak_variant = load_score(score_path, len(data))
            document = {"schema_version": 1, "status": "success", "dataset": dataset, "signal": signal, "variant": variant, "n_points": len(scores), "ground_truth_intervals": truth[signal], "alpha_metrics": evaluate(scores, timestamps, truth[signal], config["evaluation"]), "timing": timing, "peak_cuda_memory_bytes": peak_variant, "backend_metadata": metadata}
            action = "wrote" if publish_json_no_replace(metric_path, document) else "kept"
            print(f"[{ordinal}/{len(tasks)}] {action} {dataset}/{signal}/{variant}")
    publish_json_no_replace(args.output_root / "manifests" / f"worker_{args.worker_id:03d}.json", {**plan, "status": "complete", "schema_version": 1})
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "feature_cache", help="Read/reuse visual cache root; files are validated by the backend and never replaced")
    result.add_argument("--backend-module", default=DEFAULT_BACKEND)
    result.add_argument("--variants")
    result.add_argument("--worker-id", type=int, default=0)
    result.add_argument("--num-workers", type=int, default=1)
    result.add_argument("--device", default="cpu")
    result.add_argument("--limit", type=int)
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
