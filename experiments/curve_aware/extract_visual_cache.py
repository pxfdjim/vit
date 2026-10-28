#!/usr/bin/env python3
"""Extract immutable CLIP block-6 caches for all 11 indexed datasets.

Existing ``visual_*.pt`` artifacts provide final-layer features and are opened
read-only.  New block-6 tokens and curve coordinates are written to a separate
root using atomic no-replace publication.  Dry-run never imports or loads CLIP.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.signal import detrend


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIGS = [
    REPO_ROOT / "experiments/vfg_paired/config.json",
    REPO_ROOT / "experiments/vfg_paired/config_yahoo.json",
]
DEFAULT_DATA_ROOT = Path("/home/pengxf/jim/VLM4TS/data")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vfg_paired.runner import publish_json_no_replace  # noqa: E402
from src.models.curve_aware_clip import (  # noqa: E402
    FrozenCLIPBlock6Encoder,
    build_intermediate_payload,
    curve_coordinates_from_plot_transform,
    create_openai_block6_encoder,
    encode_block6,
    expected_window_count,
    input_digest,
    intermediate_cache_path,
    load_final_tokens,
    publish_torch_no_replace,
    validate_intermediate_cache,
    visual_windows,
)


YAHOO_DIRS = {
    "YAHOOA1": "A1Benchmark",
    "YAHOOA2": "A2Benchmark",
    "YAHOOA3": "A3Benchmark",
    "YAHOOA4": "A4Benchmark",
}


@dataclass(frozen=True)
class CacheTask:
    dataset: str
    signal: str
    path: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must be an object: {path}")
    return value


def load_dataset_index(data_root: Path) -> dict[str, list[str]]:
    """Read the authoritative signal index for every dataset."""

    result: dict[str, list[str]] = {}
    path = data_root / "datasets.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if len(row) != 2:
                raise ValueError(f"{path}:{line_number}: expected two columns")
            dataset, encoded = row
            signals = ast.literal_eval(encoded)
            if not isinstance(signals, (list, tuple)) or not all(
                isinstance(signal, str) and signal for signal in signals
            ):
                raise ValueError(f"{path}:{line_number}: invalid signal index")
            if dataset in result or len(set(signals)) != len(signals):
                raise ValueError(f"{path}:{line_number}: duplicate dataset/signal")
            result[dataset] = list(signals)
    return result


def signal_path(data_root: Path, dataset: str, signal: str) -> Path:
    if dataset in YAHOO_DIRS:
        return data_root / "Yahoo_S5" / YAHOO_DIRS[dataset] / f"{signal}.csv"
    return data_root / dataset / f"{signal}.csv"


def discover_tasks(
    configs: Sequence[Mapping[str, Any]], data_root: Path
) -> tuple[list[CacheTask], dict[str, list[str]]]:
    """Resolve only ``datasets.csv`` entries, including the seven old sets."""

    index = load_dataset_index(data_root)
    expected: dict[str, int] = {}
    datasets: list[str] = []
    for config in configs:
        for dataset in config.get("datasets", []):
            count = int(config.get("expected_signal_counts", {}).get(dataset, -1))
            if dataset in expected and expected[dataset] != count:
                raise ValueError(f"conflicting expected count for {dataset}")
            if dataset not in expected:
                datasets.append(dataset)
            expected[dataset] = count
    if not datasets:
        raise ValueError("configs contain no datasets")

    tasks: list[CacheTask] = []
    ignored: dict[str, list[str]] = {}
    for dataset in datasets:
        if dataset not in index:
            raise KeyError(f"{dataset} is absent from {data_root / 'datasets.csv'}")
        signals = index[dataset]
        if expected[dataset] >= 0 and len(signals) != expected[dataset]:
            raise ValueError(
                f"{dataset}: config expects {expected[dataset]}, index has {len(signals)}"
            )
        paths = [signal_path(data_root, dataset, signal) for signal in signals]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset}: indexed file missing: {missing[0]}")
        directory = paths[0].parent
        indexed_names = {path.name for path in paths}
        ignored[dataset] = sorted(
            path.name for path in directory.glob("*.csv") if path.name not in indexed_names
        )
        tasks.extend(
            CacheTask(dataset=dataset, signal=signal, path=path)
            for signal, path in zip(signals, paths)
        )
    return tasks, ignored


def shard_tasks(
    tasks: Sequence[CacheTask], worker_index: int, num_workers: int
) -> list[CacheTask]:
    if num_workers < 1 or not 0 <= worker_index < num_workers:
        raise ValueError("Require num_workers >= 1 and 0 <= worker_index < num_workers")
    return list(tasks[worker_index::num_workers])


def load_signal(task: CacheTask) -> pd.DataFrame:
    source = pd.read_csv(task.path)
    if "value" not in source:
        raise ValueError(f"{task.path}: missing value column")
    values = pd.to_numeric(source["value"], errors="coerce").to_numpy(dtype=np.float64)
    if values.size < 224 or not np.isfinite(values).all():
        raise ValueError(f"{task.path}: invalid values")
    if task.dataset in YAHOO_DIRS:
        timestamps = np.arange(1, values.size + 1, dtype=np.int64)
    else:
        if "timestamp" not in source:
            raise ValueError(f"{task.path}: missing timestamp column")
        order = np.argsort(source["timestamp"].to_numpy(), kind="stable")
        values = values[order]
        timestamps = source["timestamp"].to_numpy()[order]
    return pd.DataFrame({"timestamp": timestamps, "value": values})


def render_images(visual_values: np.ndarray) -> torch.Tensor:
    """Call the unchanged ViT4TS plotting implementation."""

    from src.preprocessing.preprocess import draw_windowed_images

    with tempfile.TemporaryDirectory(prefix="curve_aware_render_") as directory:
        success = draw_windowed_images(
            base_series_id="series",
            save_path=directory,
            time_series=visual_values,
            time_points=np.arange(len(visual_values)),
            window_size=224,
            step_size=56,
            override=True,
            save_image=False,
            image_size=(224, 224),
            dpi=100,
            plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
        )
        if not success:
            raise RuntimeError("ViT4TS renderer produced no windows")
        array = np.load(Path(directory) / "series_line_img.npy", allow_pickle=False)
    return torch.from_numpy(array).float()


def create_encoder(device: str, checkpoint: Path) -> FrozenCLIPBlock6Encoder:
    """Load the verified repository checkpoint once per worker."""
    return create_openai_block6_encoder(checkpoint, device)


def prepare_signal(task: CacheTask, frame: pd.DataFrame) -> dict[str, Any]:
    raw = frame["value"].to_numpy(dtype=np.float64)
    detrended = np.asarray(detrend(raw), dtype=np.float64)
    minimum = float(detrended.min())
    span = float(detrended.max() - minimum)
    visual = (detrended - minimum) / span if span > 0 else np.zeros_like(detrended)
    return {
        "raw": raw,
        "visual": visual,
        "digest": input_digest(raw),
        "n_windows": expected_window_count(raw.size),
        "context": {"dataset": task.dataset, "signal": task.signal},
    }


def extract_one(
    task: CacheTask,
    frame: pd.DataFrame,
    *,
    encoder: FrozenCLIPBlock6Encoder | None,
    device: str,
    batch_size: int,
    final_cache_roots: Sequence[Path],
    cache_root: Path,
) -> tuple[dict[str, Any], bool]:
    prepared = prepare_signal(task, frame)
    digest = prepared["digest"]
    context = prepared["context"]
    final, final_path = load_final_tokens(
        final_cache_roots, context, digest, prepared["n_windows"]
    )
    # ``final`` is intentionally validated and then left untouched.  The
    # curve-aware runner reads it from the old visual cache root.
    del final
    path = intermediate_cache_path(cache_root, context, digest)
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        validate_intermediate_cache(
            cached, digest=digest, expected_windows=prepared["n_windows"], path=path
        )
        return ({
            "dataset": task.dataset,
            "signal": task.signal,
            "input_digest": digest,
            "n_points": len(frame),
            "n_windows": prepared["n_windows"],
            "final_cache_path": str(final_path),
            "intermediate_cache_path": str(path),
            "status": "validated_existing",
        }, False)
    if encoder is None:
        return ({
            "dataset": task.dataset,
            "signal": task.signal,
            "input_digest": digest,
            "n_points": len(frame),
            "n_windows": prepared["n_windows"],
            "final_cache_path": str(final_path),
            "intermediate_cache_path": str(path),
            "status": "needs_extraction",
        }, True)

    windows = visual_windows(prepared["visual"])
    coordinates = curve_coordinates_from_plot_transform(windows)
    images = render_images(prepared["visual"])
    if images.shape[0] != prepared["n_windows"]:
        raise RuntimeError("rendered and numeric window counts differ")
    tokens = encode_block6(encoder, images, device=device, batch_size=batch_size)
    payload = build_intermediate_payload(
        tokens, coordinates, digest=digest, source_final_cache=final_path
    )
    published = publish_torch_no_replace(path, payload)
    stored = torch.load(path, map_location="cpu", weights_only=True)
    validate_intermediate_cache(
        stored, digest=digest, expected_windows=prepared["n_windows"], path=path
    )
    return ({
        "dataset": task.dataset,
        "signal": task.signal,
        "input_digest": digest,
        "n_points": len(frame),
        "n_windows": prepared["n_windows"],
        "final_cache_path": str(final_path),
        "intermediate_cache_path": str(path),
        "status": "published" if published else "peer_won",
    }, False)


def manifest_path(args: argparse.Namespace) -> Path:
    suffix = "" if args.limit is None else f"_limit_{args.limit:06d}"
    return args.manifest_dir / (
        f"curve_cache_worker_{args.worker_index:03d}_of_{args.num_workers:03d}{suffix}.json"
    )


def run(args: argparse.Namespace) -> int:
    configs = [load_json(path) for path in args.config]
    all_tasks, ignored = discover_tasks(configs, args.data_root)
    tasks = shard_tasks(all_tasks, args.worker_index, args.num_workers)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        tasks = tasks[: args.limit]
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.dataset] = counts.get(task.dataset, 0) + 1
    plan = {
        "configs": [str(path) for path in args.config],
        "data_root": str(args.data_root),
        "final_cache_roots": [str(path) for path in args.final_cache_root],
        "cache_root": str(args.cache_root),
        "device": args.device,
        "total_indexed_signals": len(all_tasks),
        "worker_signals": len(tasks),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "worker_dataset_counts": counts,
        "ignored_directory_csv": ignored,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    records: list[dict[str, Any]] = []
    failures = 0
    encoder: FrozenCLIPBlock6Encoder | None = None
    for ordinal, task in enumerate(tasks, 1):
        started = time.perf_counter()
        try:
            frame = load_signal(task)
            record, needs_encoder = extract_one(
                task,
                frame,
                encoder=encoder,
                device=args.device,
                batch_size=args.batch_size,
                final_cache_roots=args.final_cache_root,
                cache_root=args.cache_root,
            )
            if needs_encoder:
                if encoder is None:
                    encoder = create_encoder(args.device, args.checkpoint)
                record, needs_encoder = extract_one(
                    task,
                    frame,
                    encoder=encoder,
                    device=args.device,
                    batch_size=args.batch_size,
                    final_cache_roots=args.final_cache_root,
                    cache_root=args.cache_root,
                )
                if needs_encoder:
                    raise RuntimeError("encoder initialization did not enable extraction")
            record["elapsed_seconds"] = time.perf_counter() - started
            records.append(record)
            print(f"[{ordinal}/{len(tasks)}] {record['status']} {task.dataset}/{task.signal}")
        except Exception as error:
            failures += 1
            records.append({
                "dataset": task.dataset,
                "signal": task.signal,
                "source_path": str(task.path),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(f"[{ordinal}/{len(tasks)}] failed {task.dataset}/{task.signal}: {error}")

    manifest = {
        "schema_version": 1,
        "status": "complete" if failures == 0 else "failed",
        **plan,
        "config_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.config
        },
        "n_success": len(records) - failures,
        "n_failed": failures,
        "records": records,
    }
    path = manifest_path(args)
    action = "wrote" if publish_json_no_replace(path, manifest) else "kept existing"
    print(f"{action} manifest {path}")
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, action="append")
    result.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument("--final-cache-root", type=Path, action="append", required=True)
    result.add_argument("--cache-root", type=Path, required=True)
    result.add_argument("--manifest-dir", type=Path, required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_DATA_ROOT / "model_cache/ViT-B-16.pt",
    )
    result.add_argument("--batch-size", type=int, default=20)
    result.add_argument("--worker-index", type=int, default=0)
    result.add_argument("--num-workers", type=int, default=1)
    result.add_argument("--limit", type=int)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.config is None:
        args.config = DEFAULT_CONFIGS
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

