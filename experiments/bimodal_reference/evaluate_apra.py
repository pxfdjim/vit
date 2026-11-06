#!/usr/bin/env python3
"""Evaluate Detect-only APRA scores with the immutable OrionBench metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.mindts_metrics import _events, affiliation_scores, fast_vus
from experiments.bimodal_reference.orionbench_protocol import (DATASETS, evaluation_frame,
    labels_for, load_truth, signal_names)
from src.models.model_utils import compute_detection_intervals


VARIANTS = ("B3", "E6_Detect_only")


def event_hits(events, intervals):
    return np.asarray([any(left <= end and right >= start for left, right in intervals)
                       for start, end in events], dtype=bool)


def false_positives(events, intervals):
    return sum(not any(start <= event_end and end >= event_start
                       for event_start, event_end in events) for start, end in intervals)


def aggregate(metrics: pd.DataFrame, config: dict):
    dataset_rows = []
    for (dataset, variant), group in metrics.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant, "signals": len(group)}
        for metric in ("aff_f1", "vus_pr", "vus_roc"):
            values = group[metric].to_numpy(float)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            row[f"{metric}_valid_signals"] = int(np.isfinite(values).sum())
        dataset_rows.append(row)
    by_dataset = pd.DataFrame(dataset_rows)
    scope_rows = []
    for scope, datasets in config["macro_scopes"].items():
        for variant in VARIANTS:
            group = by_dataset[(by_dataset.dataset.isin(datasets)) &
                               (by_dataset.variant == variant)]
            row = {"scope": scope, "variant": variant, "datasets": len(group)}
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                row[metric] = float(np.nanmean(group[metric]))
                row[f"{metric}_valid_signals"] = int(group[f"{metric}_valid_signals"].sum())
            scope_rows.append(row)
    return by_dataset, pd.DataFrame(scope_rows)


def evaluate_shard(args, config):
    truth = load_truth(Path(config["data_root"]))
    metric_rows, event_rows, alignment_rows = [], [], []
    task_index = 0
    for dataset in DATASETS:
        for signal in signal_names(Path(config["data_root"]), dataset):
            current = task_index
            task_index += 1
            if current % args.num_workers != args.worker_index:
                continue
            frame = evaluation_frame(Path(config["data_root"]), dataset, signal)
            labels = labels_for(frame, truth[signal])
            events = _events(labels)
            for variant in VARIANTS:
                with np.load(args.output_root / "point_scores" / dataset / signal /
                             f"{variant}.npz") as stored:
                    scores = stored["scores"].astype(float)
                    timestamps = stored["timestamps"]
                valid = (len(scores) == len(frame) and
                         np.array_equal(timestamps, frame.timestamp.to_numpy()) and
                         np.isfinite(scores).all())
                alignment_rows.append({"dataset": dataset, "signal": signal,
                                       "variant": variant, "points": len(scores), "valid": valid})
                if not valid:
                    raise RuntimeError(f"score alignment failure: {dataset}/{signal}/{variant}")
                base = {"dataset": dataset, "signal": signal, "variant": variant,
                        "points": len(scores), "anomaly_points": int(labels.sum())}
                if labels.min() == labels.max():
                    metric_rows.append({**base, "aff_f1": np.nan, "vus_pr": np.nan,
                                        "vus_roc": np.nan,
                                        "excluded_reason": "single_class_ground_truth"})
                    continue
                intervals, threshold, _ = compute_detection_intervals(
                    scores, alpha=float(config["evaluation"]["affiliation_alpha"]),
                    method=config["evaluation"]["threshold_method"],
                    smoothing=config["evaluation"]["smoothing"], sliding=False,
                    anomaly_padding=0)
                prediction = np.zeros(len(scores), np.uint8)
                for start, end in intervals:
                    prediction[start:end + 1] = 1
                precision, recall, f1 = affiliation_scores(
                    labels, prediction, Path(config["mindts_root"]),
                    empty_prediction_f1_zero=True)
                vus = fast_vus(labels, scores)
                hits = event_hits(events, intervals)
                metric_rows.append({**base, "aff_precision": precision,
                    "aff_recall": recall, "aff_f1": f1, "vus_pr": vus.vus_pr,
                    "vus_roc": vus.vus_roc, "aff_threshold": float(threshold),
                    "predicted_intervals": len(intervals), "excluded_reason": ""})
                event_rows.append({"dataset": dataset, "signal": signal,
                    "variant": variant, "true_events": len(events),
                    "tp_events": int(hits.sum()), "fn_events": int((~hits).sum()),
                    "fp_intervals": false_positives(events, intervals),
                    "predicted_intervals": len(intervals)})
    shard_root = args.output_root / "evaluation_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        shard_root / f"apra_final_metrics_worker_{args.worker_index:02d}.csv", index=False)
    pd.DataFrame(event_rows).to_csv(
        shard_root / f"apra_final_events_worker_{args.worker_index:02d}.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(
        shard_root / f"apra_final_alignment_worker_{args.worker_index:02d}.csv", index=False)


def combine(args, config):
    shard_root = args.output_root / "evaluation_shards"
    def read(kind):
        paths = sorted(shard_root.glob(f"apra_final_{kind}_worker_*.csv"))
        if len(paths) != args.num_workers:
            raise RuntimeError(f"expected {args.num_workers} {kind} shards, found {len(paths)}")
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics = read("metrics").sort_values(["dataset", "signal", "variant"])
    events = read("events").sort_values(["dataset", "signal", "variant"])
    alignment = read("alignment").sort_values(["dataset", "signal", "variant"])
    by_dataset, by_scope = aggregate(metrics, config)
    event_all = events.groupby("variant", as_index=False).agg(
        signals=("signal", "count"), true_events=("true_events", "sum"),
        tp_events=("tp_events", "sum"), fn_events=("fn_events", "sum"),
        fp_intervals=("fp_intervals", "sum"),
        predicted_intervals=("predicted_intervals", "sum"))
    event_dataset = events.groupby(["dataset", "variant"], as_index=False).agg(
        signals=("signal", "count"), true_events=("true_events", "sum"),
        tp_events=("tp_events", "sum"), fn_events=("fn_events", "sum"),
        fp_intervals=("fp_intervals", "sum"),
        predicted_intervals=("predicted_intervals", "sum"))
    manifests = [json.loads(path.read_text()) for path in sorted(
        (args.output_root / "inference_manifests").glob("*.json"))]
    diagnostics = pd.DataFrame([row for manifest in manifests
                                for row in manifest.get("diagnostics", [])])
    outputs = {
        "metrics_per_signal.csv": metrics,
        "metrics_by_dataset.csv": by_dataset,
        "metrics_by_scope.csv": by_scope,
        "event_counts_per_signal.csv": events,
        "event_counts_by_dataset.csv": event_dataset,
        "event_counts_all.csv": event_all,
        "alignment_audit.csv": alignment,
        "inference_diagnostics_per_signal.csv": diagnostics,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_root / name, index=False)
    print(by_scope.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--combine-shards", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.combine_shards:
        combine(args, config)
    else:
        evaluate_shard(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
