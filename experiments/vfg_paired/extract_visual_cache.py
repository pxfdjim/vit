#!/usr/bin/env python3
"""Extract reusable ViT4TS visual caches for indexed Yahoo S5 signals.

The real extraction path is delegated to ``VFGViT4TSBackend``. This entrypoint
only owns strict Yahoo task discovery, deterministic sharding, source row
alignment, and an atomic worker manifest. Dry-run stops before importing the
CLIP backend.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config_yahoo.json"
DEFAULT_DATA_ROOT = Path("/home/pengxf/jim/VLM4TS/data")
DEFAULT_CACHE_ROOT = REPO_ROOT / "data/runs/vfg_paired_yahoo_20261028/feature_cache"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vfg_paired.runner import publish_json_no_replace  # noqa: E402


YAHOO_DIRS = {
    "YAHOOA1": "A1Benchmark",
    "YAHOOA2": "A2Benchmark",
    "YAHOOA3": "A3Benchmark",
    "YAHOOA4": "A4Benchmark",
}
YAHOO_COLUMNS = {
    "YAHOOA1": ("timestamp", "value", "is_anomaly"),
    "YAHOOA2": ("timestamp", "value", "is_anomaly"),
    "YAHOOA3": ("timestamps", "value", "anomaly"),
    "YAHOOA4": ("timestamps", "value", "anomaly"),
}
WINDOW_SIZE = 224
STEP_SIZE = 56


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
    """Read the authoritative signal index without scanning Yahoo directories."""

    path = data_root / "datasets.csv"
    result: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if len(row) != 2:
                raise ValueError(f"{path}:{line_number}: expected two columns")
            dataset, encoded = row
            if dataset in result:
                raise ValueError(f"{path}:{line_number}: duplicate dataset {dataset}")
            signals = ast.literal_eval(encoded)
            if not isinstance(signals, (list, tuple)) or not all(
                isinstance(signal, str) and signal for signal in signals
            ):
                raise ValueError(f"{path}:{line_number}: invalid signal index")
            if len(set(signals)) != len(signals):
                raise ValueError(f"{path}:{line_number}: duplicate signal name")
            result[dataset] = list(signals)
    return result


def discover_tasks(
    config: Mapping[str, Any], data_root: Path
) -> tuple[list[CacheTask], dict[str, list[str]]]:
    """Resolve only files named by ``datasets.csv`` and report ignored extras."""

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config.datasets must be a nonempty list")
    if len(set(datasets)) != len(datasets):
        raise ValueError("config.datasets contains duplicates")
    unsupported = [name for name in datasets if name not in YAHOO_DIRS]
    if unsupported:
        raise ValueError(f"Visual cache entrypoint accepts Yahoo A1-A4 only: {unsupported}")

    index = load_dataset_index(data_root)
    expected_counts = config.get("expected_signal_counts", {})
    tasks: list[CacheTask] = []
    ignored: dict[str, list[str]] = {}
    for dataset in datasets:
        if dataset not in index:
            raise KeyError(f"{dataset} is absent from {data_root / 'datasets.csv'}")
        signals = index[dataset]
        configured = expected_counts.get(dataset)
        if configured is not None and int(configured) != len(signals):
            raise ValueError(
                f"{dataset}: config expects {configured}, datasets.csv indexes {len(signals)}"
            )
        directory = data_root / "Yahoo_S5" / YAHOO_DIRS[dataset]
        if not directory.is_dir():
            raise FileNotFoundError(f"Yahoo directory does not exist: {directory}")
        indexed_names = {f"{signal}.csv" for signal in signals}
        present_names = {path.name for path in directory.glob("*.csv")}
        missing = sorted(indexed_names - present_names)
        if missing:
            raise FileNotFoundError(
                f"{dataset}: {len(missing)} indexed CSV files are missing: {missing[:8]}"
            )
        ignored[dataset] = sorted(present_names - indexed_names)
        tasks.extend(
            CacheTask(dataset, signal, directory / f"{signal}.csv")
            for signal in signals
        )
    return tasks, ignored


def shard_tasks(
    tasks: list[CacheTask], worker_index: int, num_workers: int
) -> list[CacheTask]:
    if num_workers < 1 or not 0 <= worker_index < num_workers:
        raise ValueError("Require num_workers >= 1 and 0 <= worker_index < num_workers")
    return tasks[worker_index::num_workers]


def load_yahoo_signal(task: CacheTask) -> pd.DataFrame:
    """Load one univariate signal and replace source time with row positions 1..N."""

    source = pd.read_csv(task.path)
    required = set(YAHOO_COLUMNS[task.dataset])
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{task.path}: missing columns {missing}")
    if source.empty:
        raise ValueError(f"Empty Yahoo signal: {task.path}")
    values = pd.to_numeric(source["value"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite value in {task.path}")
    label_column = YAHOO_COLUMNS[task.dataset][2]
    labels = pd.to_numeric(source[label_column], errors="coerce").to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Non-binary {label_column} in {task.path}")
    if len(values) < WINDOW_SIZE:
        raise ValueError(f"{task.path}: fewer than {WINDOW_SIZE} points")
    return pd.DataFrame(
        {
            "timestamp": np.arange(1, len(values) + 1, dtype=np.int64),
            "value": values,
        }
    )


def window_count(length: int) -> int:
    if length < WINDOW_SIZE:
        return 0
    return 1 + (length - WINDOW_SIZE) // STEP_SIZE


def guided_reference_summary(n_windows: int) -> dict[str, int | bool]:
    """Describe D's ragged up-to-five pool under the exact overlap rule."""

    counts = [
        sum(abs(query - reference) >= 4 for reference in range(n_windows))
        for query in range(n_windows)
    ]
    if not counts:
        return {
            "guided_min_reference_windows": 0,
            "guided_max_reference_windows": 0,
            "guided_queries_below_target_5": 0,
            "guided_pool_is_ragged": False,
        }
    return {
        "guided_min_reference_windows": min(counts),
        "guided_max_reference_windows": min(5, max(counts)),
        "guided_queries_below_target_5": sum(count < 5 for count in counts),
        "guided_pool_is_ragged": any(count < 5 for count in counts),
    }


def create_backend(
    config: Mapping[str, Any], device: str, cache_root: Path
) -> Any:
    """Import and construct the real frozen OpenAI CLIP backend lazily."""

    from src.models.vfg_vit4ts import VFGViT4TSBackend

    return VFGViT4TSBackend(config=config, device=device, cache_dir=cache_root)


def extract_one(
    backend: Any, task: CacheTask, frame: pd.DataFrame
) -> dict[str, Any]:
    """Use the VFG backend's digest, renderer, encoder and atomic publisher."""

    from src.preprocessing.preprocess import preprocess_time_series

    raw_values = frame["value"].to_numpy(dtype=np.float64)
    digest = backend._input_digest(raw_values)
    context = {"dataset": task.dataset, "signal": task.signal}
    cache_path = backend._cache_path(context, digest)
    existed_before = cache_path.is_file()
    visual_values = np.asarray(preprocess_time_series(raw_values), dtype=np.float64)
    _, cache_hit = backend._load_or_extract_visual(visual_values, context, digest)
    if not cache_path.is_file():
        raise RuntimeError(f"Visual cache was not published: {cache_path}")
    n_windows = window_count(len(frame))
    return {
        "dataset": task.dataset,
        "signal": task.signal,
        "source_path": str(task.path),
        "n_points": int(len(frame)),
        "n_windows": n_windows,
        "all_paired_variants_eligible": n_windows >= 8,
        **guided_reference_summary(n_windows),
        "input_digest": digest,
        "cache_path": str(cache_path),
        "cache_existed_before": existed_before,
        "cache_hit": bool(cache_hit),
        "status": "validated_existing" if cache_hit else "published_or_peer_won",
    }


def manifest_path(args: argparse.Namespace) -> Path:
    suffix = "" if args.limit is None else f"_limit_{args.limit:06d}"
    name = (
        f"visual_cache_worker_{args.worker_index:03d}_of_"
        f"{args.num_workers:03d}{suffix}.json"
    )
    directory = args.manifest_dir or args.cache_root.parent / "cache_manifests"
    return directory / name


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    all_tasks, ignored = discover_tasks(config, args.data_root)
    tasks = shard_tasks(all_tasks, args.worker_index, args.num_workers)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        tasks = tasks[: args.limit]

    plan = {
        "config": str(args.config),
        "data_root": str(args.data_root),
        "cache_root": str(args.cache_root),
        "device": args.device,
        "total_indexed_signals": len(all_tasks),
        "worker_signals": len(tasks),
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "datasets": list(config["datasets"]),
        "ignored_directory_csv": ignored,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if args.dry_run:
        counts: dict[str, int] = {}
        ineligible = []
        guided_ragged = []
        for task in tasks:
            frame = load_yahoo_signal(task)
            counts[task.dataset] = counts.get(task.dataset, 0) + 1
            n_windows = window_count(len(frame))
            if n_windows < 8:
                ineligible.append(
                    {
                        "dataset": task.dataset,
                        "signal": task.signal,
                        "n_points": len(frame),
                        "n_windows": n_windows,
                    }
                )
            guided = guided_reference_summary(n_windows)
            if guided["guided_pool_is_ragged"]:
                guided_ragged.append(
                    {
                        "dataset": task.dataset,
                        "signal": task.signal,
                        "n_points": len(frame),
                        "n_windows": n_windows,
                        **guided,
                    }
                )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "validated_by_dataset": counts,
                    "fewer_than_8_windows": ineligible,
                    "guided_fewer_than_target_5_refs": guided_ragged,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    records: list[dict[str, Any]] = []
    failures = 0
    initialization_error = None
    backend = None
    try:
        backend = create_backend(config, args.device, args.cache_root)
    except Exception as error:
        initialization_error = f"{type(error).__name__}: {error}"
        failures = len(tasks) or 1

    if backend is not None:
        for ordinal, task in enumerate(tasks, 1):
            started = time.perf_counter()
            try:
                frame = load_yahoo_signal(task)
                record = extract_one(backend, task, frame)
                record["elapsed_seconds"] = time.perf_counter() - started
                records.append(record)
                print(
                    f"[{ordinal}/{len(tasks)}] {record['status']} "
                    f"{task.dataset}/{task.signal}"
                )
            except Exception as error:
                failures += 1
                records.append(
                    {
                        "dataset": task.dataset,
                        "signal": task.signal,
                        "source_path": str(task.path),
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                print(f"[{ordinal}/{len(tasks)}] failed {task.dataset}/{task.signal}: {error}")

    manifest = {
        "schema_version": 1,
        "status": "complete" if failures == 0 else "failed",
        **plan,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "initialization_error": initialization_error,
        "n_success": sum(record["status"] != "failed" for record in records),
        "n_failed": failures,
        "records": records,
    }
    path = manifest_path(args)
    action = "wrote" if publish_json_no_replace(path, manifest) else "kept existing"
    print(f"{action} manifest {path}")
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    result.add_argument("--manifest-dir", type=Path)
    result.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    result.add_argument("--worker-index", type=int, default=0)
    result.add_argument("--num-workers", type=int, default=1)
    result.add_argument("--limit", type=int)
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source files without importing CLIP and without writing",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
