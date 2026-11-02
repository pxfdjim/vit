#!/usr/bin/env python3
"""Resumable three-worker runner for the temporal ViT4TS experiment."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/runs/temporal_matching_20261028"
DEFAULT_BACKEND = "src.models.temporal_vit4ts"
YAHOO_DIRS = {
    "YAHOOA1": "A1Benchmark",
    "YAHOOA2": "A2Benchmark",
    "YAHOOA3": "A3Benchmark",
    "YAHOOA4": "A4Benchmark",
}
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


def load_dataset_index(data_root: Path) -> dict[str, tuple[str, ...]]:
    """Read the authoritative signal names used by the original protocol."""
    with (data_root / "datasets.csv").open(newline="", encoding="utf-8") as handle:
        return {name: tuple(ast.literal_eval(signals)) for name, signals in csv.reader(handle)}


def signal_path(data_root: Path, dataset: str, signal: str) -> Path:
    if dataset in YAHOO_DIRS:
        return data_root / "Yahoo_S5" / YAHOO_DIRS[dataset] / f"{signal}.csv"
    return data_root / dataset / f"{signal}.csv"


def discover_tasks(config: Mapping[str, Any], data_root: Path) -> list[tuple[str, str, Path]]:
    dataset_index = load_dataset_index(data_root) if any(dataset in YAHOO_DIRS for dataset in config["datasets"]) else {}
    tasks = []
    for dataset in config["datasets"]:
        expected = int(config["expected_signal_counts"][dataset])
        if dataset in YAHOO_DIRS:
            if dataset not in dataset_index:
                raise RuntimeError(f"{dataset}: missing from {data_root / 'datasets.csv'}")
            signals = dataset_index[dataset]
            if len(signals) != len(set(signals)):
                raise RuntimeError(f"{dataset}: duplicate signal names in datasets.csv")
            entries = [(signal, signal_path(data_root, dataset, signal)) for signal in signals]
        else:
            entries = [(path.stem, path) for path in sorted((data_root / dataset).glob("*.csv"))]
        missing = [str(path) for _, path in entries if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset}: indexed signal file missing: {missing[0]}")
        if len(entries) != expected:
            raise RuntimeError(f"{dataset}: expected {expected} signals, found {len(entries)}")
        tasks.extend((dataset, signal, path) for signal, path in entries)
    return tasks


def load_signal(path: Path, dataset: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "value" not in frame:
        raise ValueError(f"{path} must contain value")
    if dataset in YAHOO_DIRS:
        # Orion's anomalies.csv stores one-based row positions for every Yahoo
        # benchmark. A1/A2 use ``timestamp`` and ``is_anomaly`` while A3/A4
        # use ``timestamps`` and ``anomaly``; neither source label column is an
        # input to this unsupervised detector.
        frame = pd.DataFrame({
            "timestamp": np.arange(1, len(frame) + 1, dtype=np.int64),
            "value": frame["value"].to_numpy(),
        })
    else:
        if "timestamp" not in frame:
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


def configure_final_cache_roots(config: dict[str, Any], roots: Iterable[Path]) -> list[str]:
    """Expose ordered, read-only final-layer roots to the backend."""
    normalized = [str(Path(root).resolve()) for root in roots]
    config.setdefault("cache", {})["final_cache_roots"] = normalized
    return normalized


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


def normalize_prediction(raw: Any, fallback_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float], dict[str, np.ndarray]]:
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
    diagnostics = {}
    for key, value in dict(raw.get("diagnostics", {})).items():
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"Backend diagnostic {key!r} must be a finite numeric array")
        diagnostics[str(key)] = array
    return scores, timestamps, metadata, timing, diagnostics


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
    final_cache_roots = configure_final_cache_roots(config, args.final_cache_root)
    if not final_cache_roots:
        raise ValueError("provide at least one read-only --final-cache-root")
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
    plan = {"total_signals": len(tasks_all), "worker_signals": len(tasks), "worker_id": args.worker_id, "num_workers": args.num_workers, "device": args.device, "cache_root": str(args.cache_root), "final_cache_roots": final_cache_roots, "output_root": str(args.output_root), "variants": variants}
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        from src.models.curve_aware_clip import final_cache_path, input_digest, intermediate_cache_path
        final_files = 0
        curve_files = 0
        missing: list[str] = []
        for dataset, signal, path in tasks:
            length = len(load_signal(path, dataset))
            window_size = int(config["model"]["window_size"])
            step_size = int(config["model"]["step_size"])
            if length < window_size:
                raise ValueError(f"Signal shorter than one window: {path}")
            n_windows = 1 + (length - window_size) // step_size
            families = {str(config["variants"][name]["family"]).upper() for name in variants}
            # Four neighboring windows on either side are excluded. Eight
            # windows are therefore the minimum that leaves a real reference
            # for every paired query; A alone only needs one rendered window.
            required = 8 if families - {"A"} else 1
            if n_windows < required:
                raise ValueError(
                    f"{dataset}/{signal} has {n_windows} windows; requested variants "
                    f"require at least {required}"
                )
            context = {"dataset": dataset, "signal": signal}
            digest = input_digest(load_signal(path, dataset)["value"].to_numpy(dtype=np.float64))
            finals = [final_cache_path(root, context, digest) for root in final_cache_roots]
            curve = intermediate_cache_path(args.cache_root, context, digest)
            has_final = any(item.is_file() for item in finals)
            has_curve = curve.is_file()
            final_files += int(has_final); curve_files += int(has_curve)
            if not has_final or not has_curve:
                missing.append(
                    f"{dataset}/{signal}: final={'ok' if has_final else 'missing'}, "
                    f"curve={'ok' if has_curve else 'missing'}"
                )
        if missing:
            raise RuntimeError(
                f"cache preflight failed for {len(missing)}/{len(tasks)} worker signals; "
                + "; ".join(missing[:8])
            )
        print(
            f"Dry-run validated {len(tasks)} signals without importing the backend; "
            f"found {final_files} digest-matched final caches and {curve_files} curve caches."
        )
        return 0

    config_payload = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    config_path = args.output_root / "config.json"
    if not publish_bytes_no_replace(config_path, config_payload) and config_path.read_bytes() != config_payload:
        raise RuntimeError(f"existing output config differs from this run: {config_path}")
    backend = None
    state: dict[str, Any] = {}
    for ordinal, (dataset, signal, path) in enumerate(tasks, 1):
        data = load_signal(path, dataset)
        score_dir = args.output_root / "point_scores" / dataset / signal
        diagnostic_dir = args.output_root / "path_diagnostics" / dataset / signal
        metric_dir = args.output_root / "metrics" / dataset / signal
        diagnostic_families = {"T1", "T2", "T3", "T4", "T5"}
        def needs_backend(name: str) -> bool:
            score_missing = not (score_dir / f"{name}.npz").exists()
            family = str(config["variants"][name]["family"]).upper()
            diagnostic_missing = family in diagnostic_families and not (diagnostic_dir / f"{name}.npz").exists()
            return score_missing or diagnostic_missing
        missing = [name for name in variants if needs_backend(name)]
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
            diagnostic_path = diagnostic_dir / f"{variant}.npz"
            family = str(config["variants"][variant]["family"]).upper()
            if metric_path.exists() and not (family in diagnostic_families and not diagnostic_path.exists()):
                document = load_json(metric_path)
                if (document.get("dataset"), document.get("signal"), document.get("variant")) != (dataset, signal, variant):
                    raise ValueError(f"Invalid existing metric retained at {metric_path}")
                load_score(score_path, len(data))
                print(f"[{ordinal}/{len(tasks)}] skip {dataset}/{signal}/{variant}")
                continue
            diagnostics: dict[str, np.ndarray] = {}
            if variant in predictions:
                predicted_scores, predicted_timestamps, predicted_metadata, predicted_timing, diagnostics = normalize_prediction(predictions[variant], data["timestamp"].to_numpy())
            if score_path.exists():
                scores, timestamps, metadata, timing, peak_variant = load_score(score_path, len(data))
            else:
                scores, timestamps, metadata, timing = predicted_scores, predicted_timestamps, predicted_metadata, predicted_timing
                timing.setdefault("inference_seconds", wall)
                peak_variant = int(metadata.get("peak_cuda_memory_bytes", peak)) if metadata.get("peak_cuda_memory_bytes", peak) is not None else None
                published = publish_npz_no_replace(score_path, scores=scores, timestamps=timestamps, metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]), timing_json=np.asarray([json.dumps(timing, sort_keys=True)]), peak_cuda_memory_bytes=np.asarray([-1 if peak_variant is None else peak_variant], dtype=np.int64))
                if not published:
                    scores, timestamps, metadata, timing, peak_variant = load_score(score_path, len(data))
            if diagnostics:
                publish_npz_no_replace(diagnostic_path, **diagnostics)
            document = {"schema_version": 1, "status": "success", "dataset": dataset, "signal": signal, "variant": variant, "n_points": len(scores), "ground_truth_intervals": truth[signal], "alpha_metrics": evaluate(scores, timestamps, truth[signal], config["evaluation"]), "timing": timing, "peak_cuda_memory_bytes": peak_variant, "backend_metadata": metadata}
            action = "wrote" if publish_json_no_replace(metric_path, document) else "kept"
            print(f"[{ordinal}/{len(tasks)}] {action} {dataset}/{signal}/{variant}")
    phase = hashlib.sha256("\0".join(variants).encode()).hexdigest()[:10]
    publish_json_no_replace(args.output_root / "manifests" / f"worker_{args.worker_id:03d}_{phase}.json", {**plan, "status": "complete", "schema_version": 1, "variant_set_digest": phase})
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "feature_cache", help="Read-only Block-6 cache root; experiment artifacts are never written here")
    result.add_argument("--final-cache-root", type=Path, action="append", default=[], help="Read-only final-layer visual cache root; repeat to search multiple roots in order")
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
