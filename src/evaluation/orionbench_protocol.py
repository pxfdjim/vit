"""Dataset-native OrionBench data protocol for APRA training and evaluation."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DATASETS = (
    "realTraffic", "realTweets", "SMAP", "MSL",
    "artificialWithAnomaly", "realAdExchange", "realAWSCloudwatch",
    "YAHOOA1", "YAHOOA2", "YAHOOA3", "YAHOOA4",
)
NASA = frozenset(("MSL", "SMAP"))
YAHOO_DIRS = {
    "YAHOOA1": "A1Benchmark", "YAHOOA2": "A2Benchmark",
    "YAHOOA3": "A3Benchmark", "YAHOOA4": "A4Benchmark",
}
EXPECTED = {
    "realTraffic": 7, "realTweets": 10, "SMAP": 53, "MSL": 27,
    "artificialWithAnomaly": 6, "realAdExchange": 5,
    "realAWSCloudwatch": 17, "YAHOOA1": 67, "YAHOOA2": 100,
    "YAHOOA3": 100, "YAHOOA4": 100,
}
WINDOW = 224
STEP = 56


def _index(data_root: Path) -> dict[str, tuple[str, ...]]:
    with (data_root / "datasets.csv").open(newline="", encoding="utf-8") as handle:
        return {name: tuple(ast.literal_eval(signals)) for name, signals in csv.reader(handle)}


def signal_names(data_root: Path, dataset: str) -> tuple[str, ...]:
    names = _index(data_root)[dataset]
    if len(names) != EXPECTED[dataset] or len(set(names)) != len(names):
        raise RuntimeError(f"{dataset}: invalid authoritative signal index")
    return names


def signal_path(data_root: Path, dataset: str, signal: str) -> Path:
    if dataset in YAHOO_DIRS:
        return data_root / "Yahoo_S5" / YAHOO_DIRS[dataset] / f"{signal}.csv"
    return data_root / dataset / f"{signal}.csv"


def load_complete(data_root: Path, dataset: str, signal: str) -> pd.DataFrame:
    path = signal_path(data_root, dataset, signal)
    source = pd.read_csv(path)
    values = pd.to_numeric(source["value"], errors="coerce").to_numpy(np.float64)
    if dataset in YAHOO_DIRS:
        timestamps = np.arange(1, len(values) + 1, dtype=np.int64)
    else:
        timestamps = source["timestamp"].to_numpy()
    frame = pd.DataFrame({"timestamp": timestamps, "value": values})
    if len(frame) < WINDOW or frame.isna().any().any() or not np.isfinite(values).all():
        raise ValueError(f"invalid signal: {path}")
    return frame


def train_frame(data_root: Path, dataset: str, signal: str) -> pd.DataFrame:
    if dataset not in NASA:
        return load_complete(data_root, dataset, signal)
    frame = pd.read_csv(data_root / dataset / "official" / "train_csv" / f"{signal}.csv")
    values = frame["value"].to_numpy(np.float64)
    reference = np.load(data_root / dataset / "official" / "train" / f"{signal}.npy")[:, 0]
    if len(values) != len(reference) or not np.allclose(values, reference, rtol=0, atol=2e-14):
        raise ValueError(f"{dataset}/{signal}: official Train CSV does not match NASA array")
    return pd.DataFrame({"timestamp": np.arange(1, len(values) + 1), "value": values})


def evaluation_frame(data_root: Path, dataset: str, signal: str) -> pd.DataFrame:
    complete = load_complete(data_root, dataset, signal)
    if dataset not in NASA:
        return complete
    n_train = len(train_frame(data_root, dataset, signal))
    test = complete.iloc[n_train:].reset_index(drop=True)
    reference = np.load(data_root / dataset / "official" / "test_reference" / f"{signal}.npy")[:, 0]
    if len(test) != len(reference) or not np.allclose(test.value, reference, rtol=0, atol=2e-14):
        raise ValueError(f"{dataset}/{signal}: evaluation tail does not match NASA Test")
    return test


def fit_normalization(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    x = np.arange(len(values), dtype=np.float64)
    slope, intercept = np.polyfit(x, values, 1)
    residual = values - (intercept + slope * x)
    center = float(np.median(residual))
    mad = 1.4826 * np.median(np.abs(residual - center))
    q1, q3 = np.quantile(residual, [0.25, 0.75])
    scale = float(max(mad, (q3 - q1) / 1.349, np.std(residual) * 0.1, 1e-8))
    return {
        "fit_points": int(len(values)), "trend_slope": float(slope),
        "trend_intercept": float(intercept), "center": center, "scale": scale,
        "visual_min": float(residual.min()), "visual_max": float(residual.max()),
    }


def normalized(values: np.ndarray, stats: Mapping[str, Any], start_index: int) -> np.ndarray:
    values = np.asarray(values, np.float64)
    x = start_index + np.arange(len(values), dtype=np.float64)
    residual = values - (float(stats["trend_intercept"]) + float(stats["trend_slope"]) * x)
    return ((residual - float(stats["center"])) / float(stats["scale"])).astype(np.float32)


def visual_values(values: np.ndarray, stats: Mapping[str, Any], start_index: int) -> np.ndarray:
    values = np.asarray(values, np.float64)
    x = start_index + np.arange(len(values), dtype=np.float64)
    residual = values - (float(stats["trend_intercept"]) + float(stats["trend_slope"]) * x)
    lo, hi = float(stats["visual_min"]), float(stats["visual_max"])
    if hi <= lo:
        return np.zeros_like(residual)
    return np.clip((residual - lo) / (hi - lo), 0.0, 1.0)


def normalization_start(dataset: str, stats: Mapping[str, Any], split: str) -> int:
    if dataset in NASA and split == "evaluation":
        return int(stats["fit_points"])
    return 0


def load_truth(data_root: Path) -> dict[str, list[list[Any]]]:
    with (data_root / "anomalies.csv").open(newline="", encoding="utf-8") as handle:
        return {row["signal"]: ast.literal_eval(row["events"]) for row in csv.DictReader(handle)}


def labels_for(frame: pd.DataFrame, events: list[list[Any]]) -> np.ndarray:
    timestamps = frame.timestamp.to_numpy()
    labels = np.zeros(len(frame), np.uint8)
    for start, end in events:
        labels |= ((timestamps >= start) & (timestamps <= end)).astype(np.uint8)
    return labels


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).view(np.uint8)).hexdigest()


def load_stats(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return json.loads(path.read_text())

