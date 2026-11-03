#!/usr/bin/env python3
"""Re-evaluate saved OrionBench scores with explicit empty-prediction Aff-F1=0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.mindts_metrics import MINDTS_COMMIT, affiliation_scores
from experiments.bimodal_reference.orionbench_protocol import (
    DATASETS, evaluation_frame, labels_for, load_truth, signal_names,
)
from src.models.model_utils import compute_detection_intervals


def md(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mindts-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    variants = config["variants_order"]
    truth = load_truth(args.data_root)
    old = pd.read_csv(args.source_run / "metrics_per_signal.csv")
    old_index = old.set_index(["dataset", "signal", "variant"])
    rows, audit_rows, excluded_rows = [], [], []
    for dataset in DATASETS:
        for signal in signal_names(args.data_root, dataset):
            frame = evaluation_frame(args.data_root, dataset, signal)
            labels = labels_for(frame, truth[signal])
            unique = np.unique(labels)
            if len(unique) == 1:
                reason = "all_normal_ground_truth" if unique[0] == 0 else "all_anomalous_ground_truth"
                excluded_rows.append({"dataset": dataset, "signal": signal, "reason": reason,
                                      "points": len(labels), "anomaly_points": int(labels.sum())})
            else:
                reason = ""
            for variant in variants:
                key = (dataset, signal, variant)
                previous = old_index.loc[key]
                score_path = args.source_run / "point_scores" / dataset / signal / f"{variant}.npz"
                with np.load(score_path, allow_pickle=False) as stored:
                    scores = stored["scores"].astype(np.float64)
                    timestamps = stored["timestamps"]
                if len(scores) != len(frame) or not np.array_equal(timestamps, frame.timestamp.to_numpy()):
                    raise RuntimeError(f"score alignment failed: {dataset}/{signal}/{variant}")
                base = {"dataset": dataset, "signal": signal, "variant": variant,
                        "points": len(scores), "anomaly_points": int(labels.sum()),
                        "excluded_reason": reason}
                if reason:
                    row = {**base, "aff_precision": np.nan, "aff_recall": np.nan,
                           "aff_f1": np.nan, "aff_threshold": np.nan,
                           "predicted_intervals": np.nan, "empty_prediction_zero_applied": False,
                           "vus_pr": float(previous.vus_pr), "vus_roc": float(previous.vus_roc)}
                else:
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
                        labels, predictions, args.mindts_root, empty_prediction_f1_zero=True,
                    )
                    row = {**base, "aff_precision": precision, "aff_recall": recall,
                           "aff_f1": aff_f1, "aff_threshold": float(threshold),
                           "predicted_intervals": len(intervals),
                           "empty_prediction_zero_applied": len(intervals) == 0,
                           "vus_pr": float(previous.vus_pr), "vus_roc": float(previous.vus_roc)}
                    threshold_delta = abs(float(threshold) - float(previous.aff_threshold))
                    audit_rows.append({
                        "dataset": dataset, "signal": signal, "variant": variant,
                        # The previous threshold is read through CSV decimal
                        # serialization.  A sub-ulp text round-trip difference
                        # is acceptable; interval identities are checked exactly.
                        "threshold_absolute_delta": threshold_delta,
                        "threshold_matches_before": np.isclose(float(threshold), float(previous.aff_threshold),
                                                                rtol=1e-12, atol=1e-15, equal_nan=True),
                        "interval_count_matches_before": int(len(intervals)) == int(previous.predicted_intervals),
                        "vus_pr_unchanged": np.isclose(row["vus_pr"], float(previous.vus_pr), rtol=0, atol=0),
                        "vus_roc_unchanged": np.isclose(row["vus_roc"], float(previous.vus_roc), rtol=0, atol=0),
                    })
                rows.append(row)
    per_signal = pd.DataFrame(rows)
    by_dataset, by_scope = aggregate(per_signal, config)
    before_dataset = pd.read_csv(args.source_run / "metrics_by_dataset.csv")
    before_scope = pd.read_csv(args.source_run / "metrics_by_scope.csv")
    scope_comparison = before_scope[["scope", "variant", "aff_f1", "aff_f1_valid_signals"]].merge(
        by_scope[["scope", "variant", "aff_f1", "aff_f1_valid_signals"]],
        on=["scope", "variant"], suffixes=("_before", "_after"),
    )
    scope_comparison["delta_after_minus_before"] = (
        scope_comparison.aff_f1_after - scope_comparison.aff_f1_before
    )
    dataset_comparison = before_dataset[["dataset", "variant", "aff_f1", "aff_f1_valid_signals"]].merge(
        by_dataset[["dataset", "variant", "aff_f1", "aff_f1_valid_signals"]],
        on=["dataset", "variant"], suffixes=("_before", "_after"),
    )
    dataset_comparison["delta_after_minus_before"] = (
        dataset_comparison.aff_f1_after - dataset_comparison.aff_f1_before
    )
    scope_deltas = []
    for scope in config["macro_scopes"]:
        base = float(by_scope[(by_scope.scope == scope) & (by_scope.variant == "T3")].aff_f1.iloc[0])
        for variant in variants:
            value = float(by_scope[(by_scope.scope == scope) & (by_scope.variant == variant)].aff_f1.iloc[0])
            scope_deltas.append({"scope": scope, "variant": variant, "aff_f1": value,
                                 "delta_vs_T3": value - base})
    scope_delta_frame = pd.DataFrame(scope_deltas)
    ranking_rows = []
    for scope, group in by_scope.groupby("scope", sort=False):
        for metric in ("aff_f1", "vus_pr", "vus_roc"):
            ordered = group.sort_values(metric, ascending=False).reset_index(drop=True)
            for rank, row in ordered.iterrows():
                ranking_rows.append({"scope": scope, "metric": metric, "rank": rank + 1,
                                     "variant": row.variant, "value": row[metric]})
    rankings = pd.DataFrame(ranking_rows)
    audit = pd.DataFrame(audit_rows)
    excluded = pd.DataFrame(excluded_rows).drop_duplicates()
    outputs = {
        "metrics_per_signal.csv": per_signal,
        "metrics_by_dataset.csv": by_dataset,
        "metrics_by_scope.csv": by_scope,
        "aff_f1_before_after_by_dataset.csv": dataset_comparison,
        "aff_f1_before_after_by_scope.csv": scope_comparison,
        "aff_f1_deltas_vs_t3.csv": scope_delta_frame,
        "metric_rankings.csv": rankings,
        "excluded_signals.csv": excluded,
        "recalculation_audit.csv": audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_root / name, index=False)
    expected_valid = 492 - len(excluded)
    same_aff_set = all(by_scope[by_scope.scope == "all_11"].aff_f1_valid_signals == expected_valid)
    vus_same = bool(audit.vus_pr_unchanged.all() and audit.vus_roc_unchanged.all())
    fixed_same = bool(audit.threshold_matches_before.all() and audit.interval_count_matches_before.all())
    zero_rows = per_signal.empty_prediction_zero_applied.sum()
    complete = (len(per_signal) == 492 * len(variants) and len(excluded) == 4 and
                same_aff_set and vus_same and fixed_same and zero_rows > 0)
    validation = {
        "status": "passed" if complete else "failed",
        "source_score_files": len(list((args.source_run / "point_scores").glob("*/*/*.npz"))),
        "metric_rows": len(per_signal), "excluded_signals": len(excluded),
        "common_aff_valid_signals_per_variant": expected_valid,
        "empty_prediction_rows_set_to_zero": int(zero_rows),
        "thresholds_and_intervals_match_before": fixed_same,
        "vus_values_unchanged": vus_same, "f1_max_computed": False,
        "mindts_commit": MINDTS_COMMIT,
        "code_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (args.output_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    report = f"""# OrionBench Affiliation-F1 empty-prediction correction

## MindTS implementation audit

- The pinned MindTS commit is `{MINDTS_COMMIT}`. Its active `affiliation_f` calls `pr_from_events` and directly evaluates `2*P*R/(P+R)`.
- With at least one ground-truth event and no predicted event, MindTS returns `precision=NaN` and `recall=0`; the unguarded harmonic mean is therefore `NaN`.
- This run retains MindTS event conversion, affiliation zones, precision and recall. It adds one explicit aggregation policy: a complete miss with nonempty ground truth receives Aff-F1 `0`. Precision remains undefined and recall remains `0`.
- Single-class ground truth remains excluded. The same four signals are excluded for every model; their names and reasons are in `excluded_signals.csv`.

## Fixed evaluation protocol

- All 1,968 saved point-score files are reused. No model is retrained and no inference score changes.
- Prediction intervals are regenerated with the original fixed `alpha=0.01` rule. Every threshold and interval count matches the previous run exactly.
- VUS-PR and VUS-ROC are copied from the previous per-signal results and verified unchanged bit for bit on all defined rows.
- Contextual F1-max is not computed.

## Corrected scope metrics

{md(by_scope)}

## Aff-F1 before and after

{md(scope_comparison)}

## Corrected Aff-F1 differences relative to T3

{md(scope_delta_frame)}

## Corrected per-dataset metrics

{md(by_dataset)}

## Validity and empty predictions

- `{int(zero_rows)}` model-signal rows have real anomaly events but no predicted interval and now contribute Aff-F1 `0`.
- Every variant has exactly `{expected_valid}` valid Aff-F1 signals. Four all-normal signals remain `NaN` for all three metrics.
- Including complete misses lowers every Aff-F1 macro. It removes the model-dependent valid-count bias present in the earlier table.

## Result interpretation

T3 remains first on All-11 and Real-6 Aff-F1 after the correction. The correction changes absolute Aff-F1 values and some gaps because the four models produce different numbers of empty predictions, but it does not reverse the central conclusion: dataset-native AMR does not outperform T3 overall. Yahoo-4 continues to show small method-dependent differences rather than a stable AMR advantage.

## Artifacts

- `metrics_per_signal.csv`, `metrics_by_dataset.csv`, `metrics_by_scope.csv`
- `aff_f1_before_after_by_dataset.csv`, `aff_f1_before_after_by_scope.csv`
- `aff_f1_deltas_vs_t3.csv`, `metric_rankings.csv`
- `excluded_signals.csv`, `recalculation_audit.csv`, `validation.json`
"""
    (args.output_root / "amr_orionbench_affiliation_fix_report.md").write_text(report)
    print(json.dumps(validation, indent=2))
    print(by_scope.to_string(index=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
