#!/usr/bin/env python3
"""Evaluate fixed-B3 AMR training ablations and feature diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.mindts_metrics import affiliation_scores, fast_vus
from experiments.bimodal_reference.orionbench_protocol import (
    DATASETS, evaluation_frame, labels_for, load_truth, signal_names,
)
from src.models.model_utils import compute_detection_intervals


def aggregate(per_signal: pd.DataFrame, config: dict, variants: list[str]):
    rows = []
    for (dataset, variant), group in per_signal.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant, "signals": len(group)}
        for metric in ("aff_f1", "vus_pr", "vus_roc"):
            values = group[metric].to_numpy(float)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            row[f"{metric}_valid_signals"] = int(np.isfinite(values).sum())
        rows.append(row)
    by_dataset = pd.DataFrame(rows)
    scope_rows = []
    for scope, datasets in config["macro_scopes"].items():
        for variant in variants:
            group = by_dataset[(by_dataset.dataset.isin(datasets)) & (by_dataset.variant == variant)]
            row = {"scope": scope, "variant": variant, "datasets": len(group)}
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                row[metric] = float(np.nanmean(group[metric]))
                row[f"{metric}_valid_signals"] = int(group[f"{metric}_valid_signals"].sum())
            scope_rows.append(row)
    return by_dataset, pd.DataFrame(scope_rows)


def metric_deltas(frame: pd.DataFrame, key: str, baselines: list[str]):
    rows = []
    for group_name, group in frame.groupby(key, sort=False):
        lookup = group.set_index("variant")
        for _, item in group.iterrows():
            row = {key: group_name, "variant": item.variant}
            for baseline in baselines:
                for metric in ("aff_f1", "vus_pr", "vus_roc"):
                    row[f"delta_{metric}_vs_{baseline}"] = float(item[metric] - lookup.loc[baseline, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("initial", "all"), required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--combine-shards", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_root = Path(config["data_root"]); mindts_root = Path(config["mindts_root"])
    variants = config["initial_variants"] if args.stage == "initial" else config["all_variants"]
    learned = [name for name in variants if name not in {"T3", "E2", "B3"}]
    truth = load_truth(data_root)
    metric_rows, feature_rows, correlation_rows, score_difference_rows = [], [], [], []
    task_index = 0
    if not args.combine_shards:
        for dataset in DATASETS:
            for signal in signal_names(data_root, dataset):
                current = task_index; task_index += 1
                if current % args.num_workers != args.worker_index:
                    continue
                frame = evaluation_frame(data_root, dataset, signal)
                labels = labels_for(frame, truth[signal])
                values = frame.value.to_numpy(np.float64)
                volatility = pd.Series(values).rolling(16, center=True, min_periods=4).std().bfill().ffill().to_numpy()
                normal = labels == 0
                high_volatility = normal & (volatility >= np.quantile(volatility[normal], 0.9))
                scores_by_variant = {}
                for variant in variants:
                    with np.load(args.output_root / "point_scores" / dataset / signal / f"{variant}.npz",
                                 allow_pickle=False) as stored:
                        scores = stored["scores"].astype(np.float64); timestamps = stored["timestamps"]
                    if len(scores) != len(frame) or not np.array_equal(timestamps, frame.timestamp.to_numpy()):
                        raise RuntimeError(f"score alignment failed: {dataset}/{signal}/{variant}")
                    scores_by_variant[variant] = scores
                    base = {"dataset": dataset, "signal": signal, "variant": variant,
                            "points": len(scores), "anomaly_points": int(labels.sum())}
                    if labels.min() == labels.max():
                        metric_rows.append({**base, "aff_precision": np.nan, "aff_recall": np.nan,
                                            "aff_f1": np.nan, "vus_pr": np.nan, "vus_roc": np.nan,
                                            "excluded_reason": "single_class_ground_truth"})
                        continue
                    intervals, threshold, _ = compute_detection_intervals(
                        scores, alpha=float(config["evaluation"]["affiliation_alpha"]),
                        method=config["evaluation"]["threshold_method"],
                        smoothing=bool(config["evaluation"]["smoothing"]),
                        sliding=bool(config["evaluation"]["sliding"]),
                        anomaly_padding=int(config["evaluation"]["anomaly_padding"]))
                    predictions = np.zeros(len(scores), np.uint8)
                    for start, end in intervals:
                        predictions[start:end + 1] = 1
                    precision, recall, aff_f1 = affiliation_scores(
                        labels, predictions, mindts_root, empty_prediction_f1_zero=True)
                    vus = fast_vus(labels, scores)
                    metric_rows.append({**base, "aff_precision": precision, "aff_recall": recall,
                                        "aff_f1": aff_f1, "vus_pr": vus.vus_pr,
                                        "vus_roc": vus.vus_roc, "aff_threshold": float(threshold),
                                        "predicted_intervals": len(intervals), "excluded_reason": ""})
                with np.load(args.output_root / "feature_diagnostics" / dataset / signal /
                             "numeric_scores.npz", allow_pickle=False) as stored:
                    raw_point = stored["point_raw_t3"].astype(np.float64)
                    learned_points = {name: stored[f"point_numeric__{name}"].astype(np.float64)
                                      for name in learned}
                full_point = learned_points["Full"]
                for variant, scores in learned_points.items():
                    normal_scores = scores[normal]
                    iqr = max(float(np.quantile(normal_scores, .75) - np.quantile(normal_scores, .25)), 1e-12)
                    feature_rows.append({
                        "dataset": dataset, "signal": signal, "variant": variant,
                        "mean": float(scores.mean()), "std": float(scores.std()),
                        "q90": float(np.quantile(scores, .9)), "q99": float(np.quantile(scores, .99)),
                        "normal_median": float(np.median(scores[normal])),
                        "anomaly_median": float(np.median(scores[labels > 0])) if (labels > 0).any() else np.nan,
                        "high_volatility_normal_median": float(np.median(scores[high_volatility])),
                        "anomaly_minus_high_volatility_normal_in_normal_iqr":
                            float((np.median(scores[labels > 0]) - np.median(scores[high_volatility])) / iqr)
                            if (labels > 0).any() else np.nan,
                        "normal_volatility_spearman": float(spearmanr(scores[normal], volatility[normal]).statistic)
                            if normal.sum() > 2 and np.std(scores[normal]) else np.nan,
                    })
                    correlation_rows.append({
                        "dataset": dataset, "signal": signal, "variant": variant,
                        "pearson_with_raw_E2": float(pearsonr(scores, raw_point).statistic),
                        "spearman_with_raw_E2": float(spearmanr(scores, raw_point).statistic),
                        "pearson_with_full_numeric": float(pearsonr(scores, full_point).statistic),
                        "spearman_with_full_numeric": float(spearmanr(scores, full_point).statistic),
                    })
                    score_difference_rows.append({
                        "dataset": dataset, "signal": signal, "variant": variant,
                        "mean_abs_numeric_score_diff_vs_full": float(np.mean(np.abs(scores - full_point))),
                        "max_abs_numeric_score_diff_vs_full": float(np.max(np.abs(scores - full_point))),
                        "mean_abs_b3_score_diff_vs_B3": float(np.mean(np.abs(
                            scores_by_variant[variant] - scores_by_variant["B3"]))),
                        "max_abs_b3_score_diff_vs_B3": float(np.max(np.abs(
                            scores_by_variant[variant] - scores_by_variant["B3"]))),
                    })
        shard_root = args.output_root / "evaluation_shards"; shard_root.mkdir(parents=True, exist_ok=True)
        frames = {"metrics": pd.DataFrame(metric_rows), "features": pd.DataFrame(feature_rows),
                  "correlations": pd.DataFrame(correlation_rows),
                  "score_differences": pd.DataFrame(score_difference_rows)}
        for name, frame in frames.items():
            frame.to_csv(shard_root / f"{name}_worker_{args.worker_index:02d}_{args.stage}.csv", index=False)
        print(json.dumps({"status": "passed", "worker": args.worker_index,
                          "metrics": len(metric_rows)}, indent=2))
        return 0

    shard_root = args.output_root / "evaluation_shards"
    def combine(name: str):
        paths = sorted(shard_root.glob(f"{name}_worker_*_{args.stage}.csv"))
        if len(paths) != args.num_workers:
            raise RuntimeError(f"expected {args.num_workers} {name} shards, found {len(paths)}")
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics = combine("metrics"); features = combine("features")
    correlations = combine("correlations"); score_differences = combine("score_differences")
    metrics = metrics.sort_values(["dataset", "signal", "variant"]).reset_index(drop=True)
    by_dataset, by_scope = aggregate(metrics, config, variants)
    deltas_dataset = metric_deltas(by_dataset, "dataset", ["T3", "E2", "B3", "Full"])
    deltas_scope = metric_deltas(by_scope, "scope", ["T3", "E2", "B3", "Full"])
    paired_rows, test_rows = [], []
    valid_metrics = metrics[metrics.excluded_reason.fillna("") == ""]
    for variant in variants:
        for baseline in ("B3", "Full"):
            if variant == baseline:
                continue
            left = valid_metrics[valid_metrics.variant == variant].set_index(["dataset", "signal"])
            right = valid_metrics[valid_metrics.variant == baseline].set_index(["dataset", "signal"])
            common = left.index.intersection(right.index)
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                delta = left.loc[common, metric] - right.loc[common, metric]
                for (dataset, signal), value in delta.items():
                    paired_rows.append({"dataset": dataset, "signal": signal,
                                        "variant": variant, "baseline": baseline,
                                        "metric": metric, "difference": float(value)})
                nonzero = delta[np.abs(delta) > 0]
                statistic, pvalue = (wilcoxon(nonzero).statistic, wilcoxon(nonzero).pvalue) if len(nonzero) else (0.0, 1.0)
                test_rows.append({"variant": variant, "baseline": baseline, "metric": metric,
                                  "signals": len(delta), "mean_difference": float(delta.mean()),
                                  "median_difference": float(delta.median()),
                                  "positive": int((delta > 0).sum()), "negative": int((delta < 0).sum()),
                                  "zero": int((delta == 0).sum()),
                                  "wilcoxon_statistic": float(statistic),
                                  "wilcoxon_pvalue_descriptive": float(pvalue)})
    outputs = {
        "metrics_per_signal.csv": metrics, "metrics_by_dataset.csv": by_dataset,
        "metrics_by_scope.csv": by_scope, "metric_deltas_by_dataset.csv": deltas_dataset,
        "metric_deltas_by_scope.csv": deltas_scope,
        "feature_distributions_per_signal.csv": features,
        "feature_correlations_per_signal.csv": correlations,
        "score_differences_per_signal.csv": score_differences,
        "paired_metric_differences_per_signal.csv": pd.DataFrame(paired_rows),
        "paired_metric_tests.csv": pd.DataFrame(test_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_root / name, index=False)
    print(by_scope.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
