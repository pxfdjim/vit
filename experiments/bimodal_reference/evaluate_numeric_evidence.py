#!/usr/bin/env python3
"""Evaluate all preregistered fixed-reference numeric evidence variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.mindts_metrics import (
    MINDTS_COMMIT, _events, affiliation_scores, fast_vus,
)
from experiments.bimodal_reference.orionbench_protocol import (
    DATASETS, evaluation_frame, labels_for, load_truth, signal_names,
)
from src.models.model_utils import compute_detection_intervals


FOCUS = ["MSL", "SMAP", "realAdExchange", "realAWSCloudwatch", "realTraffic"]
PRIMARY = ["T3", "AMR_native", "E1_A_level", "E1_B_dynamic", "E1_C_combined",
           "E2_C_combined_l010", "E3_A_stat_l010", "E3_B_cross_l010", "E3_C_late_l010"]


def md(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


def event_hits(events: list[tuple[int, int]], intervals: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([any(left <= end and right >= start for left, right in intervals)
                       for start, end in events], dtype=bool)


def interval_false_positives(events: list[tuple[int, int]], intervals: list[tuple[int, int]]) -> int:
    return sum(not any(start <= event_end and end >= event_start
                       for event_start, event_end in events) for start, end in intervals)


def aggregate(per_signal: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows = []
    for (dataset, variant), group in per_signal.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": dataset, "variant": variant, "signals": len(group)}
        for metric in ("aff_f1", "vus_pr", "vus_roc"):
            values = group[metric].to_numpy(float)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            row[f"{metric}_valid_signals"] = int(np.isfinite(values).sum())
        dataset_rows.append(row)
    by_dataset = pd.DataFrame(dataset_rows)
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


def deltas(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for group_name, group in frame.groupby(key, sort=False):
        base = group[group.variant == "T3"].iloc[0]
        for _, item in group.iterrows():
            rows.append({key: group_name, "variant": item.variant,
                         **{f"delta_{metric}": float(item[metric] - base[metric])
                            for metric in ("aff_f1", "vus_pr", "vus_roc")}})
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
    metric_rows, event_rows, event_detail, separation_rows, alignment_rows = [], [], [], [], []
    score_cache: dict[tuple[str, str, str], np.ndarray] = {}
    prediction_cache: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for dataset in DATASETS:
        for signal in signal_names(args.data_root, dataset):
            frame = evaluation_frame(args.data_root, dataset, signal)
            labels = labels_for(frame, truth[signal])
            events = _events(labels)
            values = frame.value.to_numpy(np.float64)
            volatility = pd.Series(values).rolling(16, center=True, min_periods=4).std().bfill().ffill().to_numpy()
            normal = labels == 0
            high_volatility = normal & (volatility >= np.quantile(volatility[normal], 0.9)) if normal.any() else normal
            for variant in variants:
                path = args.output_root / "point_scores" / dataset / signal / f"{variant}.npz"
                with np.load(path, allow_pickle=False) as stored:
                    scores = stored["scores"].astype(np.float64)
                    timestamps = stored["timestamps"]
                valid_alignment = (len(scores) == len(frame) and np.array_equal(timestamps, frame.timestamp.to_numpy())
                                   and np.isfinite(scores).all())
                alignment_rows.append({"dataset": dataset, "signal": signal, "variant": variant,
                                       "points": len(scores), "valid": valid_alignment})
                if not valid_alignment:
                    raise RuntimeError(f"alignment failed: {dataset}/{signal}/{variant}")
                score_cache[(dataset, signal, variant)] = scores
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
                prediction_cache[(dataset, signal, variant)] = intervals
                hits = event_hits(events, intervals)
                event_rows.append({"dataset": dataset, "signal": signal, "variant": variant,
                                   "true_events": len(events), "tp_events": int(hits.sum()),
                                   "fn_events": int((~hits).sum()),
                                   "fp_intervals": interval_false_positives(events, intervals),
                                   "predicted_intervals": len(intervals)})
                for event_index, ((start, end), hit) in enumerate(zip(events, hits)):
                    event_detail.append({"dataset": dataset, "signal": signal, "variant": variant,
                                         "event_index": event_index, "start": start, "end": end,
                                         "detected": bool(hit)})
                normal_scores = scores[normal]
                normal_iqr = max(float(np.quantile(normal_scores, .75) - np.quantile(normal_scores, .25)), 1e-12)
                rho = spearmanr(scores[normal], volatility[normal]).statistic if normal.sum() > 2 else np.nan
                separation_rows.append({
                    "dataset": dataset, "signal": signal, "variant": variant,
                    "normal_volatility_spearman": float(rho),
                    "anomaly_median": float(np.median(scores[labels > 0])),
                    "high_volatility_normal_median": float(np.median(scores[high_volatility])),
                    "anomaly_minus_high_volatility_normal_in_normal_iqr":
                        float((np.median(scores[labels > 0]) - np.median(scores[high_volatility])) / normal_iqr),
                })

    per_signal = pd.DataFrame(metric_rows)
    by_dataset, by_scope = aggregate(per_signal, config)
    event_frame = pd.DataFrame(event_rows)
    event_detail_frame = pd.DataFrame(event_detail)
    event_summary = event_frame.groupby(["dataset", "variant"], as_index=False).agg(
        signals=("signal", "count"), true_events=("true_events", "sum"),
        tp_events=("tp_events", "sum"), fn_events=("fn_events", "sum"),
        fp_intervals=("fp_intervals", "sum"), predicted_intervals=("predicted_intervals", "sum"))
    event_scope = event_frame.groupby("variant", as_index=False).agg(
        signals=("signal", "count"), true_events=("true_events", "sum"),
        tp_events=("tp_events", "sum"), fn_events=("fn_events", "sum"),
        fp_intervals=("fp_intervals", "sum"), predicted_intervals=("predicted_intervals", "sum"))
    event_deltas = []
    for dataset, group in event_summary.groupby("dataset", sort=False):
        base = group[group.variant == "T3"].iloc[0]
        for _, row in group.iterrows():
            event_deltas.append({"dataset": dataset, "variant": row.variant,
                                 "delta_tp_events": int(row.tp_events - base.tp_events),
                                 "delta_fn_events": int(row.fn_events - base.fn_events),
                                 "delta_fp_intervals": int(row.fp_intervals - base.fp_intervals)})
    event_delta_frame = pd.DataFrame(event_deltas)
    detail = event_detail_frame.pivot_table(index=["dataset", "signal", "event_index", "start", "end"],
                                            columns="variant", values="detected", aggfunc="first").reset_index()
    complementarity_rows = []
    for variant in variants:
        if variant == "T3":
            continue
        for dataset, group in detail.groupby("dataset", sort=False):
            complementarity_rows.append({
                "dataset": dataset, "variant": variant, "events": len(group),
                "recovered_T3_misses": int(((~group.T3) & group[variant]).sum()),
                "lost_T3_hits": int((group.T3 & (~group[variant])).sum()),
            })
    complementarity = pd.DataFrame(complementarity_rows)
    separation = pd.DataFrame(separation_rows)
    separation_dataset = separation.groupby(["dataset", "variant"], as_index=False).agg(
        normal_volatility_spearman=("normal_volatility_spearman", "mean"),
        anomaly_minus_high_volatility_normal_in_normal_iqr=("anomaly_minus_high_volatility_normal_in_normal_iqr", "median"))

    gate_rows, inference_rows = [], []
    for path in sorted((args.output_root / "evidence_diagnostics").glob("*/*/*.npz")):
        dataset, signal = path.parts[-3], path.parts[-2]
        with np.load(path, allow_pickle=False) as stored:
            for gate in ("stat", "cross", "late"):
                values = stored[f"gate_{gate}_window"]
                gate_rows.append({"dataset": dataset, "signal": signal, "gate": gate,
                                  "mean_gate": float(values.mean()), "std_gate": float(values.std()),
                                  "min_gate": float(values.min()), "max_gate": float(values.max())})
    for path in sorted((args.output_root / "inference_manifests").glob("*.json")):
        inference_rows.extend(json.loads(path.read_text())["records"])
    gates = pd.DataFrame(gate_rows)
    gate_summary = gates.groupby("gate", as_index=False).agg(
        signals=("signal", "count"), mean_gate=("mean_gate", "mean"),
        between_signal_std=("mean_gate", "std"), mean_within_signal_std=("std_gate", "mean"))
    inference = pd.DataFrame(inference_rows)
    runtime = inference.groupby("dataset", as_index=False).agg(
        signals=("signal", "count"), total_seconds=("elapsed_seconds", "sum"),
        mean_seconds=("elapsed_seconds", "mean"), max_seconds=("elapsed_seconds", "max"))

    scope_delta = deltas(by_scope, "scope")
    dataset_delta = deltas(by_dataset, "dataset")
    alignment = pd.DataFrame(alignment_rows)
    outputs = {
        "metrics_per_signal.csv": per_signal,
        "metrics_by_dataset.csv": by_dataset,
        "metrics_by_scope.csv": by_scope,
        "metric_deltas_by_scope.csv": scope_delta,
        "metric_deltas_by_dataset.csv": dataset_delta,
        "event_counts_per_signal.csv": event_frame,
        "event_counts_by_dataset.csv": event_summary,
        "event_counts_all.csv": event_scope,
        "event_deltas_by_dataset.csv": event_delta_frame,
        "event_complementarity.csv": complementarity,
        "score_volatility_per_signal.csv": separation,
        "score_volatility_by_dataset.csv": separation_dataset,
        "gate_per_signal.csv": gates,
        "gate_summary.csv": gate_summary,
        "inference_runtime.csv": runtime,
        "alignment_audit.csv": alignment,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_root / name, index=False)

    manifests = [json.loads(path.read_text()) for path in (args.output_root / "inference_manifests").glob("*.json")]
    expected_scores = 492 * len(variants)
    fixed_variants = [variant for variant in variants if variant not in {"AMR_native"}]
    valid_common = by_scope[by_scope.scope == "all_11"].aff_f1_valid_signals.eq(488).all()
    complete = (len(per_signal) == expected_scores and alignment.valid.all() and valid_common
                and len(manifests) == 8 and all(item["status"] == "passed" for item in manifests)
                and len(gates) == 492 * 3)
    validation = {
        "status": "passed" if complete else "failed", "signals": 492,
        "variants": len(variants), "score_files": len(list((args.output_root / "point_scores").glob("*/*/*.npz"))),
        "metric_rows": len(per_signal), "common_aff_valid_signals": 488,
        "single_class_signals": 4, "gate_rows": len(gates),
        "all_fixed_reference_variants": fixed_variants,
        "t3_path_audit_passed": bool(inference.path_reference_match.all()
                                     and inference.path_column_max_error.max() <= 1e-6
                                     and inference.visual_local_max_error.max() <= 1e-6),
        "labels_opened_during_inference": bool(inference.labels_opened.any()),
        "mindts_commit": MINDTS_COMMIT,
        "code_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                cwd=ROOT, text=True).strip(),
    }
    (args.output_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    scope_primary = by_scope[by_scope.variant.isin(PRIMARY)]
    e1 = by_scope[by_scope.variant.isin(["T3", "E1_A_level", "E1_B_dynamic", "E1_C_combined"])]
    e2 = by_scope[by_scope.variant.str.startswith("E2_C") | (by_scope.variant == "T3")]
    e3 = by_scope[by_scope.variant.isin(["T3", "E2_C_combined_l010", "E3_A_stat_l010",
                                         "E3_B_cross_l010", "E3_C_late_l010"])]
    focus_metrics = by_dataset[(by_dataset.dataset.isin(FOCUS)) & (by_dataset.variant.isin(PRIMARY))]
    focus_events = event_delta_frame[(event_delta_frame.dataset.isin(FOCUS)) &
                                     (event_delta_frame.variant.isin(PRIMARY))]
    focus_separation = separation_dataset[(separation_dataset.dataset.isin(FOCUS)) &
                                           (separation_dataset.variant.isin(PRIMARY))]
    report = f"""# OrionBench fixed-reference numeric anomaly evidence

## Protocol integrity

- This run uses the dataset-native OrionBench protocol: NASA official Train statistics and Test evaluation for MSL/SMAP; complete-sequence transductive fitting/evaluation for Yahoo and NAB.
- T3 reference windows and exact constrained-DTW paths are immutable for every E1/E2/E3 variant. All 492 paths are independently reconstructed from cached frozen CLIP Block-6 features and checked against the saved T3 references, columns and local visual scores.
- E1 component scales, point-score calibration and E3 gates use unlabeled signal statistics only. The 13 variants and all lambdas were committed before anomaly labels were opened.
- Aff-F1 uses fixed `alpha=0.01`; a complete miss with real events contributes zero. Four all-normal signals are excluded identically. VUS consumes continuous scores. No F1-max or label-selected threshold is computed.

## Methods

- **E1-A level:** median pointwise absolute difference between each query 16-point Patch and its T3-aligned reference Patch.
- **E1-B dynamics:** the mean of first-difference, robust spread, slope and endpoint-change residuals. Each component uses positive-median `log1p` normalization fitted without labels.
- **E1-C combined:** equal mean of normalized level and dynamic evidence. Restored numeric point scores are positive-median matched to the T3 score scale.
- **E2:** fixed pointwise fusion with E1-C at `lambda=0.05/0.10/0.20`; level-only and dynamics-only `lambda=0.10` controls are also retained.
- **E3-A:** `lambda=0.10` is multiplied by the geometric mean of Top-5 visual/numeric rank agreement, support for the visual winner, and visual/numeric candidate margins.
- **E3-B/C:** the E3-A gate is further conditioned on candidate ranking reliability from the existing AMR Cross-Attention cost or corrected Late-joint cost. Both representations use the same visual Top-5, raw numeric evidence and fixed T3 path.

## Primary scope metrics

{md(scope_primary)}

## Experiment 1: standalone numeric evidence

{md(e1)}

## Experiment 2: fixed fusion sensitivity

{md(e2)}

## Experiment 3: reliability-aware fusion

{md(e3)}

## Complete scope deltas relative to T3

{md(scope_delta)}

## Focus-dataset metrics

{md(focus_metrics)}

## Focus-dataset event changes relative to T3

{md(focus_events)}

## Normal high-volatility response

`anomaly_minus_high_volatility_normal_in_normal_iqr` is positive when anomalies score above the highest-volatility 10% of normal points. Spearman correlation is computed on normal points only.

{md(focus_separation)}

## Gate activity

{md(gate_summary)}

## Runtime

{md(runtime)}

## Result interpretation

All comparisons above are complete prespecified results. Cross-Attention contribution is read directly from E3-B minus E3-A/E3-C under identical evidence and paths; the report does not select a lambda or gate from evaluation labels. Dataset-level and event-level CSVs retain every variant for detailed auditing.

## Validation

- Status: `{validation['status']}`; score files: `{validation['score_files']}`; metric rows: `{validation['metric_rows']}`.
- Exact T3 reference/path audit: `{validation['t3_path_audit_passed']}`; labels opened during inference: `{validation['labels_opened_during_inference']}`.
- Every model uses 488 Aff-F1/VUS-defined signals and the same four single-class exclusions.

## Artifacts

- `point_scores/`, `evidence_diagnostics/`, `inference_manifests/`
- `metrics_per_signal.csv`, `metrics_by_dataset.csv`, `metrics_by_scope.csv`
- `metric_deltas_by_scope.csv`, `metric_deltas_by_dataset.csv`
- `event_counts_*.csv`, `event_deltas_by_dataset.csv`, `event_complementarity.csv`
- `score_volatility_*.csv`, `gate_per_signal.csv`, `gate_summary.csv`
- `inference_runtime.csv`, `alignment_audit.csv`, `validation.json`
"""
    (args.output_root / "numeric_evidence_final_report.md").write_text(report)
    print(json.dumps(validation, indent=2))
    print(scope_primary.to_string(index=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
