#!/usr/bin/env python3
"""Validate NASA Train/Test boundaries and write the immutable protocol manifest."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import (
    DATASETS, NASA, EXPECTED, digest, evaluation_frame, fit_normalization,
    load_complete, load_truth, signal_names, train_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    truth = load_truth(args.data_root)
    rows, stats = [], {}
    failures = []
    for dataset in DATASETS:
        stats[dataset] = {}
        for signal in signal_names(args.data_root, dataset):
            train = train_frame(args.data_root, dataset, signal)
            evaluation = evaluation_frame(args.data_root, dataset, signal)
            complete = load_complete(args.data_root, dataset, signal)
            fitted = fit_normalization(train.value.to_numpy(np.float64))
            stats[dataset][signal] = fitted
            event_count = len(truth[signal])
            label_points = sum(((evaluation.timestamp >= a) & (evaluation.timestamp <= b)).sum() for a, b in truth[signal])
            if dataset in NASA:
                official = pd.read_csv(args.data_root / dataset / "official" / "labeled_anomalies.csv")
                sequences = ast.literal_eval(official.loc[official.chan_id == signal, "anomaly_sequences"].iloc[0])
                derived = [[int(evaluation.timestamp.iloc[a]), int(evaluation.timestamp.iloc[b])] for a, b in sequences]
                if derived != truth[signal]:
                    failures.append(f"{dataset}/{signal}: official label index mismatch")
            if len(evaluation) < 224:
                failures.append(f"{dataset}/{signal}: evaluation shorter than one window")
            rows.append({
                "dataset": dataset, "signal": signal,
                "training_source": "NASA_official_Train" if dataset in NASA else "complete_unlabeled_sequence",
                "evaluation_source": "NASA_official_Test" if dataset in NASA else "complete_sequence",
                "train_points": len(train), "evaluation_points": len(evaluation), "complete_points": len(complete),
                "train_plus_test_equals_complete": bool(dataset not in NASA or len(train) + len(evaluation) == len(complete)),
                "events": event_count, "anomaly_points": int(label_points),
                "train_sha256": digest(train.value.to_numpy()), "evaluation_sha256": digest(evaluation.value.to_numpy()),
                "labels_read_for_training": False,
            })
    frame = pd.DataFrame(rows)
    counts = frame.groupby("dataset", as_index=False).agg(
        signals=("signal", "count"), train_points=("train_points", "sum"),
        evaluation_points=("evaluation_points", "sum"), events=("events", "sum"),
        anomaly_points=("anomaly_points", "sum"),
    )
    for item in counts.itertuples():
        if item.signals != EXPECTED[item.dataset]:
            failures.append(f"{item.dataset}: expected {EXPECTED[item.dataset]}, got {item.signals}")
    result = {
        "status": "passed" if not failures and len(frame) == 492 else "failed",
        "signals": len(frame), "datasets": len(counts), "failures": failures,
        "nasa_protocol": "official Train for unlabeled fitting; official Test only for evaluation",
        "other_protocol": "complete unlabeled sequence for training and the same complete sequence for evaluation",
        "labels_used_for_training": False,
    }
    frame.to_csv(args.output_root / "data_protocol_per_signal.csv", index=False)
    counts.to_csv(args.output_root / "data_protocol_by_dataset.csv", index=False)
    (args.output_root / "normalization_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.output_root / "data_protocol_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print(counts.to_string(index=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
