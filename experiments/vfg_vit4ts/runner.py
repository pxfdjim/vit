#!/usr/bin/env python3
"""Safe, resumable single-variable experiment runner for VFG-ViT4TS.

The runner owns orchestration and evaluation only. Model implementation stays in
``src.models.vfg_vit4ts`` and is reached through the two adapter functions
``load_backend`` and ``call_backend`` below.
"""

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
DEFAULT_CONFIG = HERE / "configs.json"
DEFAULT_DATA_ROOT = Path("/home/pengxf/jim/VLM4TS/data")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "runs" / "vfg_vit4ts_20261027"
DEFAULT_BACKEND = "src.models.vfg_vit4ts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def publish_bytes_no_replace(path: Path, payload: bytes) -> bool:
    """Publish complete bytes atomically, refusing to replace an existing path."""
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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_json_no_replace(path: Path, value: Mapping[str, Any]) -> bool:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    return publish_bytes_no_replace(path, (payload + "\n").encode("utf-8"))


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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_ground_truth(data_root: Path) -> dict[str, list[list[Any]]]:
    anomalies_path = data_root / "anomalies.csv"
    result: dict[str, list[list[Any]]] = {}
    with anomalies_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            events = ast.literal_eval(row["events"])
            result[row["signal"]] = events
    return result


def discover_tasks(
    config: Mapping[str, Any], data_root: Path
) -> list[tuple[str, str, Path]]:
    tasks: list[tuple[str, str, Path]] = []
    for dataset in config["datasets"]:
        dataset_dir = data_root / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        paths = sorted(dataset_dir.glob("*.csv"))
        expected = int(config["expected_signal_counts"][dataset])
        if len(paths) != expected:
            raise RuntimeError(
                f"{dataset}: expected {expected} CSV signals, found {len(paths)}"
            )
        tasks.extend((dataset, path.stem, path) for path in paths)
    return tasks


def load_signal(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"timestamp", "value"}
    if not required.issubset(data.columns):
        raise ValueError(f"{path} lacks columns {sorted(required - set(data.columns))}")
    data = data.loc[:, ["timestamp", "value"]].copy()
    if data.empty:
        raise ValueError(f"Empty time series: {path}")
    if data["timestamp"].isna().any() or data["value"].isna().any():
        raise ValueError(f"NaN timestamp/value in {path}")
    data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return data


def choose_variants(
    config: Mapping[str, Any], suite: str | None, variants: str | None
) -> list[str]:
    if variants:
        selected = [item.strip() for item in variants.split(",") if item.strip()]
    else:
        selected = list(config["suites"][suite or "base"])
    unknown = [name for name in selected if name not in config["variants"]]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return list(dict.fromkeys(selected))


def resolve_device(device: str, gpu_id: int | None) -> str:
    if gpu_id is None:
        return device
    if device not in {"auto", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("--gpu-id can only be combined with --device auto/cuda/cuda:N")
    return f"cuda:{gpu_id}"


# Backend adapter: keep every dependency on the evolving core API in this block.
def load_backend(module_name: str = DEFAULT_BACKEND) -> Any:
    """Import the model backend lazily, after dry-run and input validation."""
    return importlib.import_module(module_name)


def call_backend(
    module: Any,
    data: pd.DataFrame,
    variant_configs: Mapping[str, Mapping[str, Any]],
    experiment_config: Mapping[str, Any],
    device: str,
    cache_dir: Path,
    context: Mapping[str, Any],
    backend_state: dict[str, Any],
) -> Mapping[str, Any]:
    """Call the core backend through one narrow, replaceable interface.

    Preferred API (module level)::

        predict_variants(data=..., variants=..., config=..., device=...,
                         cache_dir=..., context=...) -> mapping

    A stateful backend may instead export ``create_backend``; the returned object
    must expose the same ``predict_variants`` call. Construction happens once per
    worker. No concrete model class name is assumed here.
    """
    if hasattr(module, "predict_variants"):
        return module.predict_variants(
            data=data,
            variants=variant_configs,
            config=experiment_config,
            device=device,
            cache_dir=cache_dir,
            context=context,
        )
    if "instance" not in backend_state:
        factory = getattr(module, "create_backend", None)
        if factory is None:
            raise AttributeError(
                f"{module.__name__} must export predict_variants or create_backend"
            )
        backend_state["instance"] = factory(
            config=experiment_config, device=device, cache_dir=cache_dir
        )
    instance = backend_state["instance"]
    return instance.predict_variants(
        data=data, variants=variant_configs, context=context
    )


def normalize_prediction(
    raw: Any, fallback_timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float | None, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    elapsed: float | None = None
    if isinstance(raw, Mapping):
        scores = raw.get("scores")
        timestamps = raw.get("timestamps", fallback_timestamps)
        elapsed_value = raw.get("inference_seconds")
        if elapsed_value is not None:
            elapsed = float(elapsed_value)
        metadata = dict(raw.get("metadata", {}))
    elif isinstance(raw, tuple) and len(raw) == 2:
        scores, timestamps = raw
    else:
        scores, timestamps = raw, fallback_timestamps
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    timestamps_array = np.asarray(timestamps).reshape(-1)
    if len(scores_array) != len(fallback_timestamps):
        raise ValueError(
            f"Backend returned {len(scores_array)} scores for "
            f"{len(fallback_timestamps)} input points"
        )
    if len(timestamps_array) != len(scores_array):
        raise ValueError("Score and timestamp lengths differ")
    if not np.all(np.isfinite(scores_array)):
        bad = int(np.size(scores_array) - np.isfinite(scores_array).sum())
        raise ValueError(f"Backend returned {bad} non-finite anomaly scores")
    return scores_array, timestamps_array, elapsed, metadata


def load_score_artifact(
    path: Path, expected_length: int
) -> tuple[np.ndarray, np.ndarray, float | None, int | None, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as stored:
        scores = np.asarray(stored["scores"], dtype=np.float64).reshape(-1)
        timestamps = np.asarray(stored["timestamps"]).reshape(-1)
        inference_seconds = None
        peak_cuda_memory_bytes = None
        metadata: dict[str, Any] = {}
        if "inference_seconds" in stored:
            value = float(stored["inference_seconds"][0])
            inference_seconds = None if np.isnan(value) else value
        if "peak_cuda_memory_bytes" in stored:
            value = int(stored["peak_cuda_memory_bytes"][0])
            peak_cuda_memory_bytes = None if value < 0 else value
        if "metadata_json" in stored:
            metadata = json.loads(str(stored["metadata_json"][0]))
    if len(scores) != expected_length or len(timestamps) != expected_length:
        raise ValueError(
            f"Existing score artifact {path} has length {len(scores)}/{len(timestamps)}; "
            f"expected {expected_length}. It was not modified."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Existing score artifact has non-finite values: {path}")
    return scores, timestamps, inference_seconds, peak_cuda_memory_bytes, metadata


def validate_metric_artifact(path: Path, dataset: str, signal: str, variant: str) -> None:
    document = load_json(path)
    identity = (document.get("dataset"), document.get("signal"), document.get("variant"))
    if identity != (dataset, signal, variant) or document.get("status") != "success":
        raise ValueError(f"Existing metric artifact has unexpected identity/status: {path}")
    if not isinstance(document.get("alpha_metrics"), list):
        raise ValueError(f"Existing metric artifact lacks alpha_metrics: {path}")


def intervals_overlap(left: Iterable[Any], right: Iterable[Any]) -> bool:
    left_start, left_end = left
    right_start, right_end = right
    return not (left_end < right_start or right_end < left_start)


def contextual_counts(
    ground_truth: list[list[Any]], detected: list[list[Any]]
) -> tuple[int, int, int]:
    # This exactly follows src/evaluation/evaluate.py.
    tp = sum(any(intervals_overlap(item, truth) for truth in ground_truth) for item in detected)
    fp = sum(not any(intervals_overlap(item, truth) for truth in ground_truth) for item in detected)
    fn = sum(not any(intervals_overlap(truth, item) for item in detected) for truth in ground_truth)
    return int(tp), int(fp), int(fn)


def score_to_intervals(
    scores: np.ndarray,
    timestamps: np.ndarray,
    alpha: float,
    evaluation: Mapping[str, Any],
) -> tuple[list[list[Any]], float, np.ndarray]:
    # Reuse the original thresholding and timestamp-alignment implementation.
    from src.models.model_utils import compute_detection_intervals

    index_intervals, threshold, smoothed = compute_detection_intervals(
        score_vector=scores,
        alpha=alpha,
        method=evaluation["threshold_method"],
        smoothing=bool(evaluation["smoothing"]),
        sliding=bool(evaluation["sliding"]),
        anomaly_padding=int(evaluation["anomaly_padding"]),
    )
    detected = [
        [timestamps[max(0, start)].item(), timestamps[min(len(timestamps) - 1, end)].item()]
        for start, end in index_intervals
    ]
    return detected, float(threshold), np.asarray(smoothed, dtype=np.float64)


def evaluate_variant(
    dataset: str,
    signal: str,
    variant: str,
    scores: np.ndarray,
    timestamps: np.ndarray,
    ground_truth: list[list[Any]],
    evaluation: Mapping[str, Any],
    inference_seconds: float | None,
    peak_cuda_memory_bytes: int | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    alpha_metrics = []
    for alpha in evaluation["alphas"]:
        detected, threshold, _ = score_to_intervals(scores, timestamps, alpha, evaluation)
        tp, fp, fn = contextual_counts(ground_truth, detected)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        alpha_metrics.append(
            {
                "alpha": float(alpha),
                "threshold": threshold,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_detected": len(detected),
                "n_ground_truth": len(ground_truth),
                "detected_intervals": detected,
            }
        )
    return {
        "schema_version": 1,
        "dataset": dataset,
        "signal": signal,
        "variant": variant,
        "status": "success",
        "n_points": len(scores),
        "inference_seconds": inference_seconds,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "ground_truth_intervals": ground_truth,
        "alpha_metrics": alpha_metrics,
        "backend_metadata": dict(metadata),
    }


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    variants = choose_variants(config, args.suite, args.variants)
    device = resolve_device(args.device, args.gpu_id)
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    all_tasks = discover_tasks(config, args.data_root)
    ground_truth = load_ground_truth(args.data_root)
    missing_ground_truth = [signal for _, signal, _ in all_tasks if signal not in ground_truth]
    if missing_ground_truth:
        raise RuntimeError(f"Ground truth missing for signals: {missing_ground_truth}")
    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        raise ValueError("Require num_workers >= 1 and 0 <= worker_id < num_workers")
    tasks = all_tasks[args.worker_id :: args.num_workers]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    plan = {
        "total_signals": len(all_tasks),
        "worker_signals": len(tasks),
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "device": device,
        "variants": variants,
        "datasets": {name: config["expected_signal_counts"][name] for name in config["datasets"]},
        "output_root": str(args.output_root),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    # Dry-run intentionally stops before importing torch or the CLIP backend.
    if args.dry_run:
        for _, _, path in tasks:
            data = load_signal(path)
            if len(data) < int(config["model"]["window_size"]):
                raise ValueError(f"Signal shorter than one window: {path}")
        print(f"Dry-run validated {len(tasks)} input signals; no output was written.")
        return 0

    publish_bytes_no_replace(
        args.output_root / "config.json",
        (json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    backend_module = None
    backend_state: dict[str, Any] = {}
    cache_dir = args.output_root / "feature_cache"
    for task_number, (dataset, signal, path) in enumerate(tasks, start=1):
        data = load_signal(path)
        fallback_timestamps = data["timestamp"].to_numpy()
        score_dir = args.output_root / "point_scores" / dataset / signal
        metric_dir = args.output_root / "metrics" / dataset / signal
        missing_scores = [name for name in variants if not (score_dir / f"{name}.npz").exists()]

        predictions: Mapping[str, Any] = {}
        outer_elapsed = 0.0
        measured_peak_cuda_bytes: int | None = None
        if missing_scores:
            if backend_module is None:
                backend_module = load_backend(args.backend_module)
            requested_configs = {name: config["variants"][name] for name in missing_scores}
            context = {
                "dataset": dataset,
                "signal": signal,
                "source_path": str(path),
                "seed": int(config["seed"]),
            }
            cuda_module = None
            cuda_device = None
            if device.startswith("cuda"):
                import torch

                cuda_module = torch.cuda
                cuda_device = torch.device(device)
                # This environment's torch build does not accept a bare CUDA
                # index here before the runtime has been initialized.  Setting
                # and passing the explicit device keeps accounting on the
                # worker's assigned GPU.
                cuda_module.set_device(cuda_device)
                cuda_module.reset_peak_memory_stats(cuda_device)
            start = time.perf_counter()
            predictions = call_backend(
                backend_module,
                data,
                requested_configs,
                config,
                device,
                cache_dir,
                context,
                backend_state,
            )
            if cuda_module is not None:
                cuda_module.synchronize(cuda_device)
                measured_peak_cuda_bytes = int(
                    cuda_module.max_memory_allocated(cuda_device)
                )
            outer_elapsed = time.perf_counter() - start
            absent = sorted(set(missing_scores) - set(predictions))
            if absent:
                raise RuntimeError(f"Backend omitted requested variants: {absent}")

        for variant in variants:
            score_path = score_dir / f"{variant}.npz"
            metric_path = metric_dir / f"{variant}.json"
            if metric_path.exists():
                validate_metric_artifact(metric_path, dataset, signal, variant)
                load_score_artifact(score_path, len(data))
                print(f"[{task_number}/{len(tasks)}] skip existing {dataset}/{signal}/{variant}")
                continue

            inference_seconds: float | None = None
            peak_cuda_memory_bytes: int | None = None
            metadata: dict[str, Any] = {}
            if score_path.exists():
                (
                    scores,
                    timestamps,
                    inference_seconds,
                    peak_cuda_memory_bytes,
                    metadata,
                ) = load_score_artifact(score_path, len(data))
            else:
                scores, timestamps, backend_elapsed, metadata = normalize_prediction(
                    predictions[variant], fallback_timestamps
                )
                inference_seconds = backend_elapsed
                if inference_seconds is None:
                    # The backend call can share feature extraction among variants.
                    # Its full wall time is retained for every variant unless the
                    # backend reports a variant-specific duration.
                    inference_seconds = outer_elapsed
                    metadata.setdefault("timing_scope", "shared_backend_call")
                peak_cuda_memory_bytes = metadata.get(
                    "peak_cuda_memory_bytes", measured_peak_cuda_bytes
                )
                if peak_cuda_memory_bytes is not None:
                    peak_cuda_memory_bytes = int(peak_cuda_memory_bytes)
                published = publish_npz_no_replace(
                    score_path,
                    scores=scores,
                    timestamps=timestamps,
                    inference_seconds=np.asarray([inference_seconds], dtype=np.float64),
                    peak_cuda_memory_bytes=np.asarray(
                        [-1 if peak_cuda_memory_bytes is None else peak_cuda_memory_bytes],
                        dtype=np.int64,
                    ),
                    metadata_json=np.asarray(
                        [json.dumps(metadata, sort_keys=True, ensure_ascii=False)]
                    ),
                )
                if not published:
                    (
                        scores,
                        timestamps,
                        inference_seconds,
                        peak_cuda_memory_bytes,
                        metadata,
                    ) = load_score_artifact(score_path, len(data))

            metric = evaluate_variant(
                dataset,
                signal,
                variant,
                np.asarray(scores, dtype=np.float64),
                np.asarray(timestamps),
                ground_truth[signal],
                config["evaluation"],
                inference_seconds,
                peak_cuda_memory_bytes,
                metadata,
            )
            if publish_json_no_replace(metric_path, metric):
                print(f"[{task_number}/{len(tasks)}] wrote {dataset}/{signal}/{variant}")
            else:
                print(f"[{task_number}/{len(tasks)}] kept existing {dataset}/{signal}/{variant}")

    worker_manifest = {
        **plan,
        "schema_version": 1,
        "status": "complete",
        "config_path": str(args.config),
        "backend_module": args.backend_module,
    }
    manifest_path = args.output_root / "manifests" / f"worker_{args.worker_id:03d}.json"
    if not publish_json_no_replace(manifest_path, worker_manifest):
        print(f"Kept existing worker manifest: {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--suite", choices=["base", "core_ablations", "extended_ablations", "all"], default="base")
    parser.add_argument("--variants", help="Comma-separated variant names; overrides --suite")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu, auto, cuda, or cuda:N")
    parser.add_argument("--gpu-id", type=int, help="Explicit CUDA index; does not alter CUDA_VISIBLE_DEVICES")
    parser.add_argument("--backend-module", default=DEFAULT_BACKEND)
    parser.add_argument("--limit", type=int, help="Limit this worker after deterministic sharding")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without importing the model backend or writing output")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
