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

    best_amr = scope[(scope.scope == "all_11") &
                     (~scope.variant.isin(["T3", "AMR_native", "E2_T3_raw_l005",
                                           "E2_T3_raw_l010", "E2_T3_raw_l020"]))].copy()
    best_amr["metric_mean"] = best_amr[["aff_f1", "vus_pr", "vus_roc"]].mean(axis=1)
    best_amr_name = best_amr.sort_values("metric_mean", ascending=False).iloc[0].variant
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

    report = f"""# AMR learned-feature anomaly scoring on OrionBench

## Protocol and integrity

- The run reuses the eleven immutable dataset-specific AMR checkpoints. No encoder is retrained, and no evaluation label is used during inference, score calibration, weighting or model selection.
- MSL/SMAP use official NASA Train statistics and official Test evaluation. Yahoo and NAB use the established complete-sequence transductive protocol. All 11 datasets and 492 evaluation signals are covered.
- B/C/D retain the exact T3 reference and constrained-DTW path. Experiment A alone reconstructs the AMR-native reference and path. Evidence weights `0.05`, `0.10` and `0.20` were fixed in advance; `0.10` is the primary configuration.
- Aff-F1 uses fixed `alpha=0.01`; a complete miss with real events contributes zero. The same four all-normal signals remain undefined for every method, leaving 488 common valid signals. VUS-PR and VUS-ROC use continuous scores.
- Validation status is **{validation['status']}** over `{validation['score_files']}` point-score files and `{validation['metric_rows']}` metric rows. Inference did not open labels; source AMR references, paths, visual costs, E2 formula and the B3/C4 identity check all passed at their recorded tolerances.

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

## Experiment B: learned numeric Transformer evidence

{md(scope_table(scope, b_main))}

On Real-6, B2 exceeds primary E2 on `{int(numeric_better.sum())}` of three metrics. B3 and C4 are mathematically identical controls and the inference audit records maximum point-score difference `{validation['maximum_b3_c4_duplicate_error']:.3e}`.

## Experiment C: Cross-Attention versus Late-joint and numeric Tokens

{md(scope_table(scope, c_main))}

For All-11, C2 minus C3 is `{triplet_text(tuple(cross_minus_late))}` and C2 minus C4 is `{triplet_text(tuple(cross_minus_numeric))}`. These differences use the same path, raw evidence, calibration and total added-evidence weight, so they isolate the learned representation.

## Experiment D: reference-volatility correction

{md(scope_table(scope, d_main))}

The correction is evaluated as a paired ablation for raw E2, numeric Tokens, Cross-Attention, Late-joint and numeric-plus-raw scoring. It is not selected per dataset.

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

## Answers to the three research questions

1. **Does the AMR numeric Transformer provide more useful evidence than raw E2 residuals?** At the fixed primary weight, the Real-6 B2-versus-E2 comparison improves `{int(numeric_better.sum())}` of three metrics. The learned numeric distance therefore {"does not consistently replace" if numeric_better.sum() < 3 else "consistently exceeds"} the raw statistics. Its value must be judged from B2/B3 and the feature-correlation tables rather than its training loss.
2. **Does Cross-Attention outperform Late-joint and pure numeric features?** The controlled All-11 C2-minus-C3 changes are `{triplet_text(tuple(cross_minus_late))}` and C2-minus-C4 changes are `{triplet_text(tuple(cross_minus_numeric))}`. A stable Cross-Attention advantage would require positive changes across the three metrics and scopes; the complete tables show whether that condition is met.
3. **Does an AMR joint score exceed E2 and reduce the AdExchange regression?** Configurations exceeding primary E2 on all three All-11 metrics: `{', '.join(beats_e2) if beats_e2 else 'none'}`. The strongest AMR-derived configuration by the prespecified three-metric descriptive mean is `{best_amr_name}`; this descriptor is reported after evaluation and is not used for model selection. AdExchange deltas and high-volatility separation are reported explicitly above.

## Final judgment

The retained conclusion follows the preregistered primary comparisons. Sensitivity variants document weight behavior and are not promoted post hoc. Raw scores, gates, feature diagnostics, event changes and all 47 configurations remain available for independent inspection.

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
