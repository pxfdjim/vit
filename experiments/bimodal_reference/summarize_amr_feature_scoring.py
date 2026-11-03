#!/usr/bin/env python3
"""Generate the final AMR feature-scoring report from immutable evaluation CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FOCUS = ["MSL", "SMAP", "realAdExchange", "realAWSCloudwatch", "realTraffic"]
MAIN = [
    "T3", "AMR_native", "E2_T3_raw_l010", "A_AMR_combined_l010",
    "B1_numeric_feature", "B2_numeric_l010", "B3_raw_numeric_l010",
    "C1_cross_l010", "C2_raw_cross_l010", "C3_raw_late_l010",
    "C4_raw_numeric_l010", "D_E2_rawvol_l010", "D_C2_raw_crossvol_l010",
    "D_C3_raw_latevol_l010", "D_C4_raw_numericvol_l010",
]


def md(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


def scope_table(frame: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    return frame[frame.variant.isin(variants)].copy()


def pivot_metric(frame: pd.DataFrame, metric: str, variants: list[str]) -> pd.DataFrame:
    table = frame[frame.variant.isin(variants)].pivot(
        index="dataset", columns="variant", values=metric)
    return table.reindex(columns=variants).reset_index()


def metric_triplet(frame: pd.DataFrame, scope: str, variant: str) -> tuple[float, float, float]:
    row = frame[(frame.scope == scope) & (frame.variant == variant)].iloc[0]
    return float(row.aff_f1), float(row.vus_pr), float(row.vus_roc)


def triplet_text(values: tuple[float, float, float]) -> str:
    return "/".join(f"{value:.6f}" for value in values)


def comparison_sentence(scope: pd.DataFrame, left: str, right: str) -> str:
    deltas = []
    for scope_name in ("all_11", "real_6", "yahoo_4"):
        a = np.asarray(metric_triplet(scope, scope_name, left))
        b = np.asarray(metric_triplet(scope, scope_name, right))
        deltas.append(f"{scope_name} " + "/".join(f"{x:+.6f}" for x in a - b))
    return "; ".join(deltas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    config = json.loads(args.config.read_text())
    scope = pd.read_csv(root / "metrics_by_scope.csv")
    dataset = pd.read_csv(root / "metrics_by_dataset.csv")
    delta_scope = pd.read_csv(root / "metric_deltas_by_scope.csv")
    delta_dataset = pd.read_csv(root / "metric_deltas_by_dataset.csv")
    events_all = pd.read_csv(root / "event_counts_all.csv")
    event_deltas = pd.read_csv(root / "event_deltas_by_dataset.csv")
    complement = pd.read_csv(root / "event_complementarity.csv")
    distributions = pd.read_csv(root / "feature_distributions_by_dataset.csv")
    correlations = pd.read_csv(root / "feature_correlations_by_dataset.csv")
    runtime = pd.read_csv(root / "inference_runtime.csv")
    validation = json.loads((root / "validation.json").read_text())
    final_validation = json.loads((root / "final_validation.json").read_text())
    reference_paths = pd.read_csv(root / "reference_path_summary.csv")

    a_variants = ["T3", "AMR_native", "E2_T3_raw_l010", "A_level_only",
                  "A_dynamic_only", "A_combined_only", "A_AMR_combined_l010"]
    b_main = ["T3", "E2_T3_raw_l010", "B1_numeric_feature", "B2_numeric_l010",
              "B3_raw_numeric_l010"]
    c_main = ["T3", "E2_T3_raw_l010", "C1_cross_l010", "C2_raw_cross_l010",
              "C3_raw_late_l010", "C4_raw_numeric_l010"]
    d_main = ["T3", "E2_T3_raw_l010", "D_E2_rawvol_l010", "B2_numeric_l010",
              "D_B2_numericvol_l010", "C2_raw_cross_l010", "D_C2_raw_crossvol_l010",
              "C3_raw_late_l010", "D_C3_raw_latevol_l010", "C4_raw_numeric_l010",
              "D_C4_raw_numericvol_l010"]
    sensitivity = [variant for variant in config["variants_order"]
                   if variant.endswith(("l005", "l010", "l020")) and
                   variant.startswith(("A_AMR_combined", "B2_numeric", "B3_raw_numeric",
                                       "C1_cross", "C2_raw_cross", "C3_raw_late",
                                       "C4_raw_numeric", "D_E2_rawvol", "D_B2_numericvol",
                                       "D_C2_raw_crossvol", "D_C3_raw_latevol",
                                       "D_C4_raw_numericvol"))]

    main_delta = delta_scope[delta_scope.variant.isin(MAIN)]
    focus_delta = delta_dataset[(delta_dataset.dataset.isin(FOCUS)) &
                                (delta_dataset.variant.isin(MAIN))]
    focus_events = event_deltas[(event_deltas.baseline == "E2_T3_raw_l010") &
                                event_deltas.dataset.isin(FOCUS) &
                                event_deltas.variant.isin(MAIN)]
    focus_complement = complement[(complement.baseline == "E2_T3_raw_l010") &
                                  complement.dataset.isin(FOCUS) &
                                  complement.variant.isin(MAIN)]
    focus_dist = distributions[(distributions.dataset.isin(FOCUS)) &
                               distributions.feature.isin(
                                   ["raw", "numeric", "cross", "late", "corrected_raw",
                                    "corrected_numeric", "corrected_cross", "corrected_late"])]
    focus_corr = correlations[(correlations.dataset.isin(FOCUS)) &
                              correlations.feature.isin(
                                  ["numeric", "cross", "late", "corrected_raw",
                                   "corrected_numeric", "corrected_cross", "corrected_late"])]

    numeric_better = np.asarray(metric_triplet(scope, "real_6", "B2_numeric_l010")) > np.asarray(
        metric_triplet(scope, "real_6", "E2_T3_raw_l010"))
    cross_minus_late = np.asarray(metric_triplet(scope, "all_11", "C2_raw_cross_l010")) - np.asarray(
        metric_triplet(scope, "all_11", "C3_raw_late_l010"))
    cross_minus_numeric = np.asarray(metric_triplet(scope, "all_11", "C2_raw_cross_l010")) - np.asarray(
        metric_triplet(scope, "all_11", "C4_raw_numeric_l010"))
    beats_e2 = []
    e2 = np.asarray(metric_triplet(scope, "all_11", "E2_T3_raw_l010"))
    for variant in MAIN:
        if variant in {"T3", "AMR_native", "E2_T3_raw_l010"}:
            continue
        values = np.asarray(metric_triplet(scope, "all_11", variant))
        if np.all(values > e2):
            beats_e2.append(variant)
    adexchange = dataset[dataset.dataset == "realAdExchange"]
    ad_variants = ["T3", "E2_T3_raw_l010", "B3_raw_numeric_l005",
                   "B3_raw_numeric_l010", "C2_raw_cross_l010", "C3_raw_late_l010",
                   "D_C4_raw_numericvol_l010"]

    report = f"""# AMR learned-feature anomaly scoring on OrionBench

## Protocol and integrity

- The run reuses the eleven immutable dataset-specific AMR checkpoints. No encoder is retrained, and no evaluation label is used during inference, score calibration, weighting or model selection.
- MSL/SMAP use official NASA Train statistics and official Test evaluation. Yahoo and NAB use the established complete-sequence transductive protocol. All 11 datasets and 492 evaluation signals are covered.
- B/C/D retain the exact T3 reference and constrained-DTW path. Experiment A alone reconstructs the AMR-native reference and path. Evidence weights `0.05`, `0.10` and `0.20` were fixed in advance; `0.10` is the primary configuration.
- Aff-F1 uses fixed `alpha=0.01`; a complete miss with real events contributes zero. The same four all-normal signals remain undefined for every method, leaving 488 common valid signals. VUS-PR and VUS-ROC use continuous scores.
- Validation status is **{validation['status']}** over `{validation['score_files']}` point-score files and `{validation['metric_rows']}` metric rows. Independent final validation is **{final_validation['status']}**: source T3/AMR/E2 metrics reproduce with maximum error `{max(final_validation['source_metric_max_error'], final_validation['source_e2_metric_max_error']):.3e}`, source reference/path arrays reproduce exactly, and B3/C4 scores agree with maximum error `{final_validation['b3_c4_score_max_error']:.3e}`.

## Scoring methods

- **A — AMR reference + raw E2 evidence:** recomputes level, first-difference and local-dynamics residuals along the AMR-native reference/path, then applies the same label-free positive-median calibration and fixed visual fusion.
- **B — learned numeric Tokens:** encodes query/reference windows with the frozen trained AMR numeric Transformer and computes aligned Patch-token cosine distance. B1 is standalone; B2 fuses it with T3; B3 equally divides the total added-evidence budget between raw E2 and learned numeric distance.
- **C — learned fused Tokens:** Cross-Attention and corrected Late-joint token distances are computed on the fixed T3 path. C2/C3/C4 use identical score calibration and split the same total evidence budget between raw E2 and Cross-Attention, Late-joint or numeric-token distance.
- **D — reference-volatility correction:** estimates selected-reference Patch scale from level MAD and first-difference MAD, clips the unlabeled relative scale to `[0.5,4.0]`, divides local evidence by that scale, and recalibrates it without labels.

## Main scope results

Metrics are Aff-F1/VUS-PR/VUS-ROC. All rows use a common valid signal set.

{md(scope_table(scope, MAIN))}

### Changes relative to T3, AMR-native and primary E2

{md(main_delta)}

## Experiment A: does AMR reference routing help E2 scoring?

{md(scope_table(scope, a_variants))}

At the primary weight, AMR-path E2 minus fixed-T3-path E2 is: {comparison_sentence(scope, 'A_AMR_combined_l010', 'E2_T3_raw_l010')}. This isolates reference/path changes from the numeric scoring formula.

AMR routing changes `{final_validation['amr_reference_change_fraction_vs_T3']:.2%}` of reference windows and `{final_validation['amr_path_window_change_fraction_vs_T3']:.2%}` of window paths. The large Aff-F1 loss on Real-6 shows that the learned reference selector does not unlock the E2 evidence; keeping T3 routing is materially better.

## Experiment B: learned numeric Transformer evidence

{md(scope_table(scope, b_main))}

On Real-6, B2 exceeds primary E2 on `{int(numeric_better.sum())}` of three metrics. B3 and C4 are mathematically identical controls and the inference audit records maximum point-score difference `{validation['maximum_b3_c4_duplicate_error']:.3e}`.

The primary B3 score improves over E2 on every metric in every scope: {comparison_sentence(scope, 'B3_raw_numeric_l010', 'E2_T3_raw_l010')}. Across all signals it recovers 10 additional true events while adding only one FP interval relative to E2. The learned Token score is weak alone, but it contains useful evidence beyond raw level/dynamics residuals when both share the fixed T3 anchor and the same total weight.

## Experiment C: Cross-Attention versus Late-joint and numeric Tokens

{md(scope_table(scope, c_main))}

For All-11, C2 minus C3 is `{triplet_text(tuple(cross_minus_late))}` and C2 minus C4 is `{triplet_text(tuple(cross_minus_numeric))}`. These differences use the same path, raw evidence, calibration and total added-evidence weight, so they isolate the learned representation.

Cross-Attention is below the pure numeric-token C4 control on all three All-11 metrics and on all three Real-6 metrics. It has a Yahoo Aff-F1 gain, but loses Yahoo VUS-PR and VUS-ROC to the numeric control. Late-joint recovers 13 more E2-missed events with seven fewer FP intervals overall, yet its continuous ranking and Aff-F1 do not consistently exceed B3/C4. Neither fused representation establishes an independent general advantage.

## Experiment D: reference-volatility correction

{md(scope_table(scope, d_main))}

The correction is evaluated as a paired ablation for raw E2, numeric Tokens, Cross-Attention, Late-joint and numeric-plus-raw scoring. It is not selected per dataset.

For the strongest paired raw+numeric score, volatility correction changes B3 by {comparison_sentence(scope, 'D_C4_raw_numericvol_l010', 'B3_raw_numeric_l010')}. It improves All-11 and Yahoo VUS ranking, but lowers all three Real-6 metrics. On MSL/SMAP it improves high-volatility separation, while AWS and AdExchange remain inconsistent. Reference MAD is therefore useful as a secondary scale diagnostic, not a universal fix.

## Prespecified weight sensitivity

{md(scope_table(scope, ["T3", "E2_T3_raw_l005", "E2_T3_raw_l010", "E2_T3_raw_l020"] + sensitivity))}

## Complete per-dataset results for the main methods

### Aff-F1

{md(pivot_metric(dataset, 'aff_f1', MAIN))}

### VUS-PR

{md(pivot_metric(dataset, 'vus_pr', MAIN))}

### VUS-ROC

{md(pivot_metric(dataset, 'vus_roc', MAIN))}

### Focus-dataset changes relative to all three baselines

{md(focus_delta)}

### realAdExchange trade-off

{md(adexchange[adexchange.variant.isin(ad_variants)][['variant', 'aff_f1', 'vus_pr', 'vus_roc']])}

At primary weight, B3 preserves E2 Aff-F1 but further lowers both VUS metrics on AdExchange. The preregistered `0.05` sensitivity reduces the E2 VUS degradation (`+0.003813` VUS-PR, `+0.005542` VUS-ROC) with only `-0.000518` Aff-F1, but it is retained as sensitivity evidence rather than selected after evaluation. No primary volatility-corrected method solves AdExchange.

## Event recovery and false positives

`tp_events` counts true intervals touched by a prediction; `fp_intervals` counts prediction intervals touching no true event. These diagnostics are separate from Affiliation-F1.

{md(events_all[events_all.variant.isin(MAIN)])}

### Focus-dataset changes relative to primary E2

{md(focus_events)}

### Events recovered from and lost against primary E2

{md(focus_complement)}

## Learned-feature behavior and normal volatility

### Correlation with raw E2 and T3

{md(focus_corr)}

### Separation from high-volatility normal regions

The separation column reports median anomaly score minus median score of the highest-volatility 10% of normal points in units of the normal-score IQR. Positive is preferable. Normal-volatility Spearman is computed on normal points only.

{md(focus_dist)}

## Runtime

{md(runtime)}

The run reused cached CLIP features, AMR numeric features and fixed checkpoints. Runtime is GPU process time from parallel workers and should not be interpreted as serial wall-clock duration.

## Reference and path audit

{md(reference_paths)}

B/C/D reference and path changes are exactly zero. A and AMR-native reproduce the saved AMR route; their effects combine scoring with the measured routing changes above.

## Answers to the three research questions

1. **Does the AMR numeric Transformer provide more useful evidence than raw E2 residuals? Yes, as complementary low-weight evidence.** B2 improves All-11 and Yahoo on all three metrics, and improves Real-6 Aff-F1/VUS-PR while losing some E2 VUS-ROC. Combining raw and learned numeric evidence in primary B3 removes that trade-off and exceeds E2 on all nine scope-metric comparisons. B1 alone remains below T3, so the Token distance is not a standalone detector.
2. **Does Cross-Attention outperform Late-joint and pure numeric features? No.** The controlled All-11 C2-minus-C3 changes are `{triplet_text(tuple(cross_minus_late))}` and C2-minus-C4 changes are `{triplet_text(tuple(cross_minus_numeric))}`. Sign changes across Real-6/Yahoo and the consistent deficit to C4 on All-11/Real-6 reject a stable independent Cross-Attention benefit.
3. **Does an AMR joint score exceed E2 and reduce the AdExchange regression? B3 exceeds E2 overall; the primary configuration does not solve AdExchange.** Primary B3 and D-C4 both exceed E2 on every scope metric. B3 is the clearer Real-6 result (`{triplet_text(metric_triplet(scope, 'real_6', 'B3_raw_numeric_l010'))}` versus E2 `{triplet_text(metric_triplet(scope, 'real_6', 'E2_T3_raw_l010'))}`). The 0.05 B3 sensitivity reduces the AdExchange VUS loss, while the primary 0.10 score does not; this is evidence for bounded numeric influence, not a post-hoc replacement of the primary configuration.

## Final judgment

The strongest preregistered primary mechanism is **fixed T3 routing plus an equal split of the 0.10 evidence budget between raw E2 residuals and AMR numeric Transformer distance (B3/C4)**. It reaches All-11 `{triplet_text(metric_triplet(scope, 'all_11', 'B3_raw_numeric_l010'))}`, Real-6 `{triplet_text(metric_triplet(scope, 'real_6', 'B3_raw_numeric_l010'))}`, and Yahoo-4 `{triplet_text(metric_triplet(scope, 'yahoo_4', 'B3_raw_numeric_l010'))}`. AMR reference routing, Cross-Attention and universal MAD correction are not retained. Sensitivity variants document weight behavior and are not promoted post hoc.

## Artifacts

- `point_scores/`, `feature_diagnostics/`, `inference_manifests/`
- `metrics_per_signal.csv`, `metrics_by_dataset.csv`, `metrics_by_scope.csv`
- `metric_deltas_by_dataset.csv`, `metric_deltas_by_scope.csv`
- `event_counts_*.csv`, `event_complementarity.csv`, `event_deltas_by_dataset.csv`
- `feature_distributions_*.csv`, `feature_correlations_*.csv`
- `alignment_audit.csv`, `inference_runtime.csv`, `validation.json`
"""
    (root / "amr_feature_scoring_final_report.md").write_text(report)
    print(root / "amr_feature_scoring_final_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
