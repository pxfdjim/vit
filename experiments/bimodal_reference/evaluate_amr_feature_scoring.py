#!/usr/bin/env python3
"""Evaluate preregistered AMR feature-scoring variants after inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.mindts_metrics import _events, affiliation_scores, fast_vus
from experiments.bimodal_reference.orionbench_protocol import (
    DATASETS, evaluation_frame, labels_for, load_truth, signal_names,
)
from src.models.model_utils import compute_detection_intervals


PRIMARY = [
    "T3", "AMR_native", "E2_T3_raw_l010", "A_AMR_combined_l010",
    "B1_numeric_feature", "B2_numeric_l010", "B3_raw_numeric_l010",
    "C1_cross_l010", "C2_raw_cross_l010", "C3_raw_late_l010", "C4_raw_numeric_l010",
    "D_E2_rawvol_l010", "D_B2_numericvol_l010", "D_C2_raw_crossvol_l010",
    "D_C3_raw_latevol_l010", "D_C4_raw_numericvol_l010",
]
FEATURE_KEYS = {
    "raw": "point_raw_t3", "numeric": "point_numeric_feature",
    "cross": "point_cross_feature", "late": "point_late_feature",
    "corrected_raw": "point_corrected_raw", "corrected_numeric": "point_corrected_numeric",
    "corrected_cross": "point_corrected_cross", "corrected_late": "point_corrected_late",
}


def event_hits(events: list[tuple[int, int]], intervals: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([any(left <= end and right >= start for left, right in intervals)
                       for start, end in events], dtype=bool)


def false_positive_intervals(events: list[tuple[int, int]], intervals: list[tuple[int, int]]) -> int:
    return sum(not any(start <= event_end and end >= event_start
                       for event_start, event_end in events) for start, end in intervals)


def aggregate(per_signal: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        for variant in config["variants_order"]:
            group = by_dataset[(by_dataset.dataset.isin(datasets)) & (by_dataset.variant == variant)]
            row = {"scope": scope, "variant": variant, "datasets": len(group)}
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                row[metric] = float(np.nanmean(group[metric]))
                row[f"{metric}_valid_signals"] = int(group[f"{metric}_valid_signals"].sum())
            scope_rows.append(row)
    return by_dataset, pd.DataFrame(scope_rows)


def deltas(frame: pd.DataFrame, group_key: str, baselines: list[str]) -> pd.DataFrame:
    rows = []
    for group_name, group in frame.groupby(group_key, sort=False):
        lookup = group.set_index("variant")
        for _, item in group.iterrows():
            row = {group_key: group_name, "variant": item.variant}
            for baseline in baselines:
                base = lookup.loc[baseline]
                for metric in ("aff_f1", "vus_pr", "vus_roc"):
                    row[f"delta_{metric}_vs_{baseline}"] = float(item[metric] - base[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mindts-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    truth = load_truth(args.data_root)
    variants = config["variants_order"]
    metric_rows, event_rows, event_details = [], [], []
    distribution_rows, correlation_rows, alignment_rows = [], [], []

    for dataset in DATASETS:
        for signal in signal_names(args.data_root, dataset):
            frame = evaluation_frame(args.data_root, dataset, signal)
            labels = labels_for(frame, truth[signal])
            events = _events(labels)
            values = frame.value.to_numpy(np.float64)
            volatility = pd.Series(values).rolling(16, center=True, min_periods=4).std().bfill().ffill().to_numpy()
            normal = labels == 0
            high_volatility = (normal & (volatility >= np.quantile(volatility[normal], 0.9))
                               if normal.any() else normal)
            score_cache = {}
            prediction_cache = {}
            for variant in variants:
                path = args.output_root / "point_scores" / dataset / signal / f"{variant}.npz"
                with np.load(path, allow_pickle=False) as stored:
                    scores = stored["scores"].astype(np.float64)
                    timestamps = stored["timestamps"]
                valid = (len(scores) == len(frame) and np.array_equal(timestamps, frame.timestamp.to_numpy())
                         and np.isfinite(scores).all())
                alignment_rows.append({"dataset": dataset, "signal": signal,
                                       "variant": variant, "points": len(scores), "valid": valid})
                if not valid:
                    raise RuntimeError(f"alignment failed: {dataset}/{signal}/{variant}")
                score_cache[variant] = scores
                base = {"dataset": dataset, "signal": signal, "variant": variant,
                        "points": len(scores), "anomaly_points": int(labels.sum())}
                if labels.min() == labels.max():
                    metric_rows.append({**base, "aff_precision": np.nan, "aff_recall": np.nan,
                                        "aff_f1": np.nan, "vus_pr": np.nan, "vus_roc": np.nan,
                                        "aff_threshold": np.nan, "predicted_intervals": np.nan,
                                        "excluded_reason": "single_class_ground_truth"})
                    continue
                intervals, threshold, _ = compute_detection_intervals(
                    scores, alpha=float(config["evaluation"]["affiliation_alpha"]),
                    method=config["evaluation"]["threshold_method"],
                    smoothing=bool(config["evaluation"]["smoothing"]),
                    sliding=bool(config["evaluation"]["sliding"]),
                    anomaly_padding=int(config["evaluation"]["anomaly_padding"]),
                )
                predictions = np.zeros(len(scores), np.uint8)
                for start, end in intervals:
                    predictions[start:end + 1] = 1
                precision, recall, aff_f1 = affiliation_scores(
                    labels, predictions, args.mindts_root, empty_prediction_f1_zero=True)
                vus = fast_vus(labels, scores)
                metric_rows.append({**base, "aff_precision": precision, "aff_recall": recall,
                                    "aff_f1": aff_f1, "vus_pr": vus.vus_pr, "vus_roc": vus.vus_roc,
                                    "aff_threshold": float(threshold), "predicted_intervals": len(intervals),
                                    "excluded_reason": "", "empty_prediction_zero_applied": len(intervals) == 0})
                prediction_cache[variant] = intervals
                if variant in PRIMARY:
                    hits = event_hits(events, intervals)
                    event_rows.append({"dataset": dataset, "signal": signal, "variant": variant,
                                       "true_events": len(events), "tp_events": int(hits.sum()),
                                       "fn_events": int((~hits).sum()),
                                       "fp_intervals": false_positive_intervals(events, intervals),
                                       "predicted_intervals": len(intervals)})
                    for event_index, ((start, end), hit) in enumerate(zip(events, hits)):
                        event_details.append({"dataset": dataset, "signal": signal,
                                              "variant": variant, "event_index": event_index,
                                              "start": start, "end": end, "detected": bool(hit)})

            diagnostic = args.output_root / "feature_diagnostics" / dataset / signal / "evidence.npz"
            with np.load(diagnostic, allow_pickle=False) as stored:
                feature_points = {name: stored[key].astype(np.float64) for name, key in FEATURE_KEYS.items()}
                reference_ratio = stored["reference_volatility_ratio"].astype(np.float64)
            raw_feature = feature_points["raw"]
            t3_score = score_cache["T3"]
            for name, scores in feature_points.items():
                normal_scores = scores[normal]
                normal_iqr = max(float(np.quantile(normal_scores, .75) - np.quantile(normal_scores, .25)), 1e-12)
                distribution_rows.append({
                    "dataset": dataset, "signal": signal, "feature": name,
                    "mean": float(scores.mean()), "std": float(scores.std()),
                    "q50": float(np.quantile(scores, .5)), "q90": float(np.quantile(scores, .9)),
                    "q99": float(np.quantile(scores, .99)),
                    "normal_median": float(np.median(scores[normal])),
                    "anomaly_median": float(np.median(scores[labels > 0])) if (labels > 0).any() else np.nan,
                    "high_volatility_normal_median": float(np.median(scores[high_volatility])) if high_volatility.any() else np.nan,
                    "anomaly_minus_high_volatility_normal_in_normal_iqr":
                        float((np.median(scores[labels > 0]) - np.median(scores[high_volatility])) / normal_iqr)
                        if (labels > 0).any() and high_volatility.any() else np.nan,
                    "normal_volatility_spearman": float(spearmanr(scores[normal], volatility[normal]).statistic)
                        if normal.sum() > 2 and np.std(scores[normal]) else np.nan,
                    "mean_reference_volatility_ratio": float(reference_ratio.mean()),
                    "std_reference_volatility_ratio": float(reference_ratio.std()),
                })
                correlation_rows.append({
                    "dataset": dataset, "signal": signal, "feature": name,
                    "pearson_with_raw_E2": float(pearsonr(scores, raw_feature).statistic)
                        if np.std(scores) and np.std(raw_feature) else np.nan,
                    "spearman_with_raw_E2": float(spearmanr(scores, raw_feature).statistic)
                        if np.std(scores) and np.std(raw_feature) else np.nan,
                    "pearson_with_T3": float(pearsonr(scores, t3_score).statistic)
                        if np.std(scores) and np.std(t3_score) else np.nan,
                    "spearman_with_T3": float(spearmanr(scores, t3_score).statistic)
                        if np.std(scores) and np.std(t3_score) else np.nan,
                })

    per_signal = pd.DataFrame(metric_rows)
    by_dataset, by_scope = aggregate(per_signal, config)
    delta_dataset = deltas(by_dataset, "dataset", ["T3", "AMR_native", "E2_T3_raw_l010"])
    delta_scope = deltas(by_scope, "scope", ["T3", "AMR_native", "E2_T3_raw_l010"])
    event_frame = pd.DataFrame(event_rows)
    event_summary = event_frame.groupby(["dataset", "variant"], as_index=False).agg(
        signals=("signal", "count"), true_events=("true_events", "sum"),
        tp_events=("tp_events", "sum"), fn_events=("fn_events", "sum"),
        fp_intervals=("fp_intervals", "sum"), predicted_intervals=("predicted_intervals", "sum"))
    event_all = event_frame.groupby("variant", as_index=False).agg(
        signals=("signal", "count"), true_events=("true_events", "sum"),
        tp_events=("tp_events", "sum"), fn_events=("fn_events", "sum"),
        fp_intervals=("fp_intervals", "sum"), predicted_intervals=("predicted_intervals", "sum"))
    event_detail = pd.DataFrame(event_details)
    pivot = event_detail.pivot_table(index=["dataset", "signal", "event_index", "start", "end"],
                                     columns="variant", values="detected", aggfunc="first").reset_index()
    complement_rows = []
    for baseline in ("T3", "E2_T3_raw_l010"):
        for variant in PRIMARY:
            if variant == baseline:
                continue
            for dataset, group in pivot.groupby("dataset", sort=False):
                complement_rows.append({
                    "dataset": dataset, "variant": variant, "baseline": baseline,
                    "events": len(group),
                    "recovered_baseline_misses": int(((~group[baseline]) & group[variant]).sum()),
                    "lost_baseline_hits": int((group[baseline] & (~group[variant])).sum()),
                })
    complement = pd.DataFrame(complement_rows)
    event_delta_rows = []
    for baseline in ("T3", "E2_T3_raw_l010"):
        for dataset, group in event_summary.groupby("dataset", sort=False):
            base = group[group.variant == baseline].iloc[0]
            for _, row in group.iterrows():
                event_delta_rows.append({
                    "dataset": dataset, "variant": row.variant, "baseline": baseline,
                    "delta_tp_events": int(row.tp_events - base.tp_events),
                    "delta_fn_events": int(row.fn_events - base.fn_events),
                    "delta_fp_intervals": int(row.fp_intervals - base.fp_intervals),
                })
    event_deltas = pd.DataFrame(event_delta_rows)
    distribution = pd.DataFrame(distribution_rows)
    distribution_dataset = distribution.groupby(["dataset", "feature"], as_index=False).agg(
        signals=("signal", "count"), mean=("mean", "mean"), std=("std", "mean"),
        q90=("q90", "mean"), q99=("q99", "mean"),
        anomaly_minus_high_volatility_normal_in_normal_iqr=(
            "anomaly_minus_high_volatility_normal_in_normal_iqr", "median"),
        normal_volatility_spearman=("normal_volatility_spearman", "mean"),
        mean_reference_volatility_ratio=("mean_reference_volatility_ratio", "mean"))
    correlations = pd.DataFrame(correlation_rows)
    correlation_dataset = correlations.groupby(["dataset", "feature"], as_index=False).agg(
        signals=("signal", "count"), pearson_with_raw_E2=("pearson_with_raw_E2", "mean"),
        spearman_with_raw_E2=("spearman_with_raw_E2", "mean"),
        pearson_with_T3=("pearson_with_T3", "mean"),
        spearman_with_T3=("spearman_with_T3", "mean"))
    alignment = pd.DataFrame(alignment_rows)

    manifest_paths = sorted((args.output_root / "inference_manifests").glob("*.json"))
    manifests = [json.loads(path.read_text()) for path in manifest_paths]
    records = [record for manifest in manifests for record in manifest["records"]
               if record["status"] == "passed"]
    inference = pd.DataFrame(records)
    runtime = inference.groupby("dataset", as_index=False).agg(
        signals=("signal", "count"), total_seconds=("elapsed_seconds", "sum"),
        mean_seconds=("elapsed_seconds", "mean"), max_seconds=("elapsed_seconds", "max"))

    outputs = {
        "metrics_per_signal.csv": per_signal, "metrics_by_dataset.csv": by_dataset,
        "metrics_by_scope.csv": by_scope, "metric_deltas_by_dataset.csv": delta_dataset,
        "metric_deltas_by_scope.csv": delta_scope, "event_counts_per_signal.csv": event_frame,
        "event_counts_by_dataset.csv": event_summary, "event_counts_all.csv": event_all,
        "event_complementarity.csv": complement, "event_deltas_by_dataset.csv": event_deltas,
        "feature_distributions_per_signal.csv": distribution,
        "feature_distributions_by_dataset.csv": distribution_dataset,
        "feature_correlations_per_signal.csv": correlations,
        "feature_correlations_by_dataset.csv": correlation_dataset,
        "alignment_audit.csv": alignment, "inference_runtime.csv": runtime,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_root / name, index=False)

    valid_counts = per_signal.groupby("variant").aff_f1.apply(lambda x: int(np.isfinite(x).sum()))
    excluded_counts = per_signal.groupby("variant").excluded_reason.apply(
        lambda x: int((x == "single_class_ground_truth").sum()))
    empty = per_signal[(per_signal.predicted_intervals == 0) & per_signal.aff_f1.notna()]
    expected = 492 * len(variants)
    complete = (len(per_signal) == expected and alignment.valid.all()
                and valid_counts.eq(488).all() and excluded_counts.eq(4).all()
                and empty.aff_f1.eq(0).all() and len(inference) == 492
                and all(manifest["status"] == "passed" for manifest in manifests)
                and not inference.labels_opened.any()
                and inference.amr_reference_error.max() == 0
                and inference.amr_column_max_error.max() <= 1e-6
                and inference.amr_visual_local_max_error.max() <= 1e-6
                and inference.e2_formula_max_error.max() <= 1e-12
                and inference.b3_c4_duplicate_max_error.max() == 0)
    validation = {
        "status": "passed" if complete else "failed", "signals": 492,
        "variants": len(variants), "score_files": len(list(
            (args.output_root / "point_scores").glob("*/*/*.npz"))),
        "metric_rows": len(per_signal), "valid_signals_per_variant": valid_counts.to_dict(),
        "excluded_single_class_per_variant": excluded_counts.to_dict(),
        "empty_prediction_rows": len(empty),
        "empty_predictions_are_zero": bool(empty.aff_f1.eq(0).all()),
        "labels_opened_during_inference": bool(inference.labels_opened.any()),
        "maximum_amr_reference_error": float(inference.amr_reference_error.max()),
        "maximum_amr_column_error": float(inference.amr_column_max_error.max()),
        "maximum_amr_visual_local_error": float(inference.amr_visual_local_max_error.max()),
        "maximum_e2_formula_error": float(inference.e2_formula_max_error.max()),
        "maximum_b3_c4_duplicate_error": float(inference.b3_c4_duplicate_max_error.max()),
    }
    (args.output_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({key: value for key, value in validation.items()
                      if not isinstance(value, dict)}, indent=2))
    print(by_scope[by_scope.variant.isin(PRIMARY)].to_string(index=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
