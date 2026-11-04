#!/usr/bin/env python3
"""Evaluate RCCA/CADM scores, reference controls, stability, and diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.mindts_metrics import _events, affiliation_scores, fast_vus
from experiments.bimodal_reference.orionbench_protocol import (
    DATASETS, evaluation_frame, labels_for, load_truth, signal_names,
)
from experiments.bimodal_reference.run_rcca_cadm_inference import output_variants
from src.models.model_utils import compute_detection_intervals


def event_hits(events, intervals):
    return np.asarray([any(left <= end and right >= start for left, right in intervals)
                       for start, end in events], dtype=bool)


def false_positive_intervals(events, intervals):
    return sum(not any(start <= ee and end >= es for es, ee in events)
               for start, end in intervals)


def aggregate(metrics: pd.DataFrame, config: dict, variants: list[str]):
    rows = []
    for (dataset, variant), group in metrics.groupby(["dataset", "variant"], sort=False):
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
            group = by_dataset[(by_dataset.dataset.isin(datasets)) &
                               (by_dataset.variant == variant)]
            row = {"scope": scope, "variant": variant, "datasets": len(group)}
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                row[metric] = float(np.nanmean(group[metric]))
                row[f"{metric}_valid_signals"] = int(group[f"{metric}_valid_signals"].sum())
            scope_rows.append(row)
    return by_dataset, pd.DataFrame(scope_rows)


def deltas(frame, key, baselines):
    rows = []
    for name, group in frame.groupby(key, sort=False):
        lookup = group.set_index("variant")
        for _, item in group.iterrows():
            row = {key: name, "variant": item.variant}
            for baseline in baselines:
                for metric in ("aff_f1", "vus_pr", "vus_roc"):
                    row[f"delta_{metric}_vs_{baseline}"] = float(
                        item[metric] - lookup.loc[baseline, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def safe_correlation(function, left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(function(left, right).statistic)


def evaluate_shard(args, config, variants):
    truth = load_truth(Path(config["data_root"])); metric_rows = []; event_rows = []
    feature_rows = []; similarity_rows = []; alignment_rows = []
    task_index = 0
    for dataset in DATASETS:
        for signal in signal_names(Path(config["data_root"]), dataset):
            current = task_index; task_index += 1
            if current % args.num_workers != args.worker_index:
                continue
            frame = evaluation_frame(Path(config["data_root"]), dataset, signal)
            labels = labels_for(frame, truth[signal]); events = _events(labels)
            values = frame.value.to_numpy(np.float64)
            volatility = pd.Series(values).rolling(16, center=True, min_periods=4).std().bfill().ffill().to_numpy()
            normal = labels == 0
            high_vol = normal & (volatility >= np.quantile(volatility[normal], .9))
            score_cache = {}
            for variant in variants:
                path = args.output_root / "point_scores" / dataset / signal / f"{variant}.npz"
                with np.load(path, allow_pickle=False) as stored:
                    scores = stored["scores"].astype(np.float64); timestamps = stored["timestamps"]
                valid = (len(scores) == len(frame) and np.array_equal(
                    timestamps, frame.timestamp.to_numpy()) and np.isfinite(scores).all())
                alignment_rows.append({"dataset": dataset, "signal": signal,
                                       "variant": variant, "points": len(scores), "valid": valid})
                if not valid:
                    raise RuntimeError(f"score alignment failed: {dataset}/{signal}/{variant}")
                score_cache[variant] = scores
                base = {"dataset": dataset, "signal": signal, "variant": variant,
                        "points": len(scores), "anomaly_points": int(labels.sum())}
                if labels.min() == labels.max():
                    metric_rows.append({**base, "aff_precision": np.nan, "aff_recall": np.nan,
                                        "aff_f1": np.nan, "vus_pr": np.nan, "vus_roc": np.nan,
                                        "predicted_intervals": np.nan,
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
                    labels, predictions, Path(config["mindts_root"]),
                    empty_prediction_f1_zero=True)
                vus = fast_vus(labels, scores); hits = event_hits(events, intervals)
                metric_rows.append({**base, "aff_precision": precision, "aff_recall": recall,
                                    "aff_f1": aff_f1, "vus_pr": vus.vus_pr,
                                    "vus_roc": vus.vus_roc, "aff_threshold": float(threshold),
                                    "predicted_intervals": len(intervals), "excluded_reason": ""})
                event_rows.append({"dataset": dataset, "signal": signal, "variant": variant,
                                   "true_events": len(events), "tp_events": int(hits.sum()),
                                   "fn_events": int((~hits).sum()),
                                   "fp_intervals": false_positive_intervals(events, intervals),
                                   "predicted_intervals": len(intervals),
                                   "event_hits_json": json.dumps(hits.astype(int).tolist())})
            diag_path = args.output_root / "feature_diagnostics" / dataset / signal / "evidence.npz"
            with np.load(diag_path, allow_pickle=False) as stored:
                features = {key: stored[key].astype(np.float64) for key in stored.files}
            feature_map = {"raw": features["point_raw_t3"]}
            for model in ("M1_full", "M2_no_rcca", "M3_same_window", "M4_no_synthetic_objective"):
                feature_map[f"numeric__{model}"] = features[f"point_numeric__{model}"]
                feature_map[f"cross__{model}"] = features[f"point_cross__{model}"]
            feature_map["cross__M5_shuffled_reference"] = features[
                "point_cross__M5_shuffled_reference"]
            raw_feature = feature_map["raw"]; full_cross = feature_map["cross__M1_full"]
            for feature, scores in feature_map.items():
                iqr = max(float(np.quantile(scores[normal], .75) -
                                np.quantile(scores[normal], .25)), 1e-12)
                feature_rows.append({
                    "dataset": dataset, "signal": signal, "feature": feature,
                    "mean": float(scores.mean()), "std": float(scores.std()),
                    "q90": float(np.quantile(scores, .9)), "q99": float(np.quantile(scores, .99)),
                    "anomaly_minus_high_volatility_normal_in_normal_iqr": float(
                        (np.median(scores[labels > 0]) - np.median(scores[high_vol])) / iqr)
                        if (labels > 0).any() and high_vol.any() else np.nan,
                    "normal_volatility_spearman": safe_correlation(
                        spearmanr, scores[normal], volatility[normal]),
                })
                similarity_rows.append({
                    "dataset": dataset, "signal": signal, "feature": feature,
                    "pearson_with_raw": safe_correlation(pearsonr, scores, raw_feature),
                    "spearman_with_raw": safe_correlation(spearmanr, scores, raw_feature),
                    "pearson_with_M1_cross": safe_correlation(pearsonr, scores, full_cross),
                    "spearman_with_M1_cross": safe_correlation(spearmanr, scores, full_cross),
                    "pearson_with_B3": safe_correlation(pearsonr, scores, score_cache["B3"]),
                })
    shard = args.output_root / "evaluation_shards"; shard.mkdir(parents=True, exist_ok=True)
    frames = {"metrics": metric_rows, "events": event_rows, "features": feature_rows,
              "similarity": similarity_rows, "alignment": alignment_rows}
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(shard / f"rcca_{name}_worker_{args.worker_index:02d}.csv",
                                  index=False)
    print(json.dumps({"status": "passed", "worker": args.worker_index,
                      "metric_rows": len(metric_rows)}))


def combine(args, config, variants):
    shard = args.output_root / "evaluation_shards"
    def read(name):
        paths = sorted(shard.glob(f"rcca_{name}_worker_*.csv"))
        if len(paths) != args.num_workers:
            raise RuntimeError(f"expected {args.num_workers} {name} shards, found {len(paths)}")
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics = read("metrics").sort_values(["dataset", "signal", "variant"])
    events = read("events"); features = read("features"); similarity = read("similarity")
    alignment = read("alignment")
    by_dataset, by_scope = aggregate(metrics, config, variants)
    delta_dataset = deltas(by_dataset, "dataset", ["T3", "E2", "B3", "C2_simplified", "M1_full"])
    delta_scope = deltas(by_scope, "scope", ["T3", "E2", "B3", "C2_simplified", "M1_full"])
    event_dataset = events.groupby(["dataset", "variant"], as_index=False).agg(
        signals=("signal", "count"), true_events=("true_events", "sum"),
        tp_events=("tp_events", "sum"), fn_events=("fn_events", "sum"),
        fp_intervals=("fp_intervals", "sum"), predicted_intervals=("predicted_intervals", "sum"))
    event_all = events.groupby("variant", as_index=False).agg(
        signals=("signal", "count"), true_events=("true_events", "sum"),
        tp_events=("tp_events", "sum"), fn_events=("fn_events", "sum"),
        fp_intervals=("fp_intervals", "sum"), predicted_intervals=("predicted_intervals", "sum"))
    event_delta_rows = []
    for (dataset, signal), group in events.groupby(["dataset", "signal"], sort=False):
        lookup = group.set_index("variant")
        for variant in variants:
            hits = np.asarray(json.loads(lookup.loc[variant, "event_hits_json"]), bool)
            for baseline in ("B3", "M1_full"):
                base_hits = np.asarray(json.loads(lookup.loc[baseline, "event_hits_json"]), bool)
                event_delta_rows.append({"dataset": dataset, "signal": signal,
                    "variant": variant, "baseline": baseline,
                    "recovered_baseline_misses": int(((~base_hits) & hits).sum()),
                    "lost_baseline_hits": int((base_hits & (~hits)).sum()),
                    "delta_fp_intervals": int(lookup.loc[variant, "fp_intervals"] -
                                              lookup.loc[baseline, "fp_intervals"])})
    event_deltas = pd.DataFrame(event_delta_rows)
    paired_rows = []
    valid = metrics[metrics.excluded_reason.fillna("") == ""]
    for variant in variants:
        for baseline in ("B3", "M1_full"):
            if variant == baseline:
                continue
            left = valid[valid.variant == variant].set_index(["dataset", "signal"])
            right = valid[valid.variant == baseline].set_index(["dataset", "signal"])
            common = left.index.intersection(right.index)
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                difference = left.loc[common, metric] - right.loc[common, metric]
                nonzero = difference[difference != 0]
                stat, pvalue = wilcoxon(nonzero) if len(nonzero) else (0.0, 1.0)
                paired_rows.append({"variant": variant, "baseline": baseline, "metric": metric,
                    "signals": len(difference), "mean_difference": float(difference.mean()),
                    "median_difference": float(difference.median()),
                    "positive": int((difference > 0).sum()), "negative": int((difference < 0).sum()),
                    "zero": int((difference == 0).sum()), "wilcoxon_statistic": float(stat),
                    "wilcoxon_pvalue_descriptive": float(pvalue)})
    seed_variants = ["M1_full"] + [f"M1_seed_{seed}" for seed in config["stability_seeds"]
                                    if seed != config["primary_seed"]]
    stability_rows = []
    for scope, group in by_scope[by_scope.variant.isin(seed_variants)].groupby("scope"):
        for metric in ("aff_f1", "vus_pr", "vus_roc"):
            stability_rows.append({"scope": scope, "metric": metric,
                "seeds": len(group), "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=0)), "min": float(group[metric].min()),
                "max": float(group[metric].max())})
    per_signal_stability = metrics[metrics.variant.isin(seed_variants)].groupby(
        ["dataset", "signal"], as_index=False).agg(
            aff_f1_mean=("aff_f1", "mean"), aff_f1_std=("aff_f1", "std"),
            vus_pr_mean=("vus_pr", "mean"), vus_pr_std=("vus_pr", "std"),
            vus_roc_mean=("vus_roc", "mean"), vus_roc_std=("vus_roc", "std"))
    checkpoint_rows = []
    for path in sorted((args.output_root / "checkpoints").glob("*/*/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        row = {"variant": payload["variant"], "seed": payload["seed"],
               "dataset": payload["dataset"], "elapsed_seconds": payload["elapsed_seconds"],
               "optimizer_updates": payload["optimizer_updates"],
               "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
               "numeric_max_abs_change": payload["numeric_max_abs_change"],
               **{f"parameters_{k}": v for k, v in payload["parameter_counts"].items()},
               **{f"gradient_{k}": v for k, v in payload["gradient_audit"].items()},
               **payload["sensitivity_audit"], "labels_used": payload["labels_used"]}
        checkpoint_rows.append(row)
    manifests = [json.loads(path.read_text()) for path in sorted(
        (args.output_root / "inference_manifests").glob("*.json"))]
    diagnostics = pd.DataFrame([row for manifest in manifests
                                for row in manifest.get("diagnostics", [])])
    diagnostic_summary = diagnostics.groupby("variant", as_index=False).agg(
        signals=("signal", "count"), attention_entropy=("attention_entropy", "mean"),
        attention_diagonal_mass=("attention_diagonal_mass", "mean"),
        cadm_logit_mean=("cadm_logit_mean", "mean"), cadm_logit_std=("cadm_logit_std", "mean"),
        conditional_difference_norm=("conditional_difference_norm", "mean"),
        inference_seconds=("inference_seconds", "sum"))
    outputs = {
        "metrics_per_signal.csv": metrics, "metrics_by_dataset.csv": by_dataset,
        "metrics_by_scope.csv": by_scope, "metric_deltas_by_dataset.csv": delta_dataset,
        "metric_deltas_by_scope.csv": delta_scope, "event_counts_per_signal.csv": events,
        "event_counts_by_dataset.csv": event_dataset, "event_counts_all.csv": event_all,
        "event_differences_per_signal.csv": event_deltas,
        "feature_distributions_per_signal.csv": features,
        "feature_similarity_per_signal.csv": similarity,
        "alignment_audit.csv": alignment, "paired_metric_tests.csv": pd.DataFrame(paired_rows),
        "stability_by_scope.csv": pd.DataFrame(stability_rows),
        "stability_per_signal.csv": per_signal_stability,
        "training_checkpoint_summary.csv": pd.DataFrame(checkpoint_rows),
        "inference_diagnostics_per_signal.csv": diagnostics,
        "inference_diagnostics_summary.csv": diagnostic_summary,
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
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    variants = output_variants(config)
    if args.combine_shards:
        combine(args, config, variants)
    else:
        evaluate_shard(args, config, variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
