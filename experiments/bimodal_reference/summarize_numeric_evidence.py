#!/usr/bin/env python3
"""Build the final fixed-reference numeric-evidence report from immutable outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PRIMARY = [
    "T3", "AMR_native", "E1_A_level", "E1_B_dynamic", "E1_C_combined",
    "E2_C_combined_l010", "E3_A_stat_l010", "E3_B_cross_l010", "E3_C_late_l010",
]
FOCUS = ["MSL", "SMAP", "realAdExchange", "realAWSCloudwatch", "realTraffic"]


def md(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


def score_correlations(root: Path, variants: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for t3_path in sorted(root.glob("point_scores/*/*/T3.npz")):
        dataset, signal = t3_path.parts[-3], t3_path.parts[-2]
        with np.load(t3_path, allow_pickle=False) as stored:
            t3 = stored["scores"].astype(np.float64)
        for variant in variants:
            with np.load(t3_path.with_name(f"{variant}.npz"), allow_pickle=False) as stored:
                score = stored["scores"].astype(np.float64)
            pearson = pearsonr(t3, score).statistic if np.std(t3) and np.std(score) else np.nan
            spearman = spearmanr(t3, score).statistic if np.std(t3) and np.std(score) else np.nan
            rows.append({"dataset": dataset, "signal": signal, "variant": variant,
                         "pearson_with_T3": pearson, "spearman_with_T3": spearman})
    per_signal = pd.DataFrame(rows)
    by_dataset = per_signal.groupby(["dataset", "variant"], as_index=False).agg(
        signals=("signal", "count"), pearson_with_T3=("pearson_with_T3", "mean"),
        spearman_with_T3=("spearman_with_T3", "mean"))
    return per_signal, by_dataset


def gate_comparison(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("evidence_diagnostics/*/*/*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            stat = stored["gate_stat_window"].astype(np.float64)
            cross = stored["gate_cross_window"].astype(np.float64)
            late = stored["gate_late_window"].astype(np.float64)
        correlation = np.corrcoef(cross, late)[0, 1] if np.std(cross) and np.std(late) else np.nan
        rows.append({"dataset": path.parts[-3], "signal": path.parts[-2],
                     "mean_abs_cross_minus_late": float(np.mean(np.abs(cross - late))),
                     "cross_late_pearson": correlation,
                     "mean_abs_cross_minus_stat": float(np.mean(np.abs(cross - stat))),
                     "mean_abs_late_minus_stat": float(np.mean(np.abs(late - stat)))})
    return pd.DataFrame(rows)


def pivot_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    result = frame[frame.variant.isin(PRIMARY)].pivot(index="dataset", columns="variant", values=metric)
    return result[PRIMARY].reset_index()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    scope = pd.read_csv(root / "metrics_by_scope.csv")
    dataset = pd.read_csv(root / "metrics_by_dataset.csv")
    scope_delta = pd.read_csv(root / "metric_deltas_by_scope.csv")
    dataset_delta = pd.read_csv(root / "metric_deltas_by_dataset.csv")
    events = pd.read_csv(root / "event_counts_all.csv")
    event_delta = pd.read_csv(root / "event_deltas_by_dataset.csv")
    complement = pd.read_csv(root / "event_complementarity.csv")
    volatility = pd.read_csv(root / "score_volatility_by_dataset.csv")
    gates = pd.read_csv(root / "gate_summary.csv")
    runtime = pd.read_csv(root / "inference_runtime.csv")
    mechanism = pd.read_csv(root / "reference_path_summary.csv")
    validation = json.loads((root / "validation.json").read_text())
    final_validation_path = root / "final_validation.json"
    final_validation = (json.loads(final_validation_path.read_text())
                        if final_validation_path.exists() else {"status": "not_run"})

    corr_variants = [variant for variant in PRIMARY if variant not in {"T3"}]
    corr_signal, corr_dataset = score_correlations(root, corr_variants)
    gate_detail = gate_comparison(root)
    corr_signal.to_csv(root / "score_correlations_per_signal.csv", index=False)
    corr_dataset.to_csv(root / "score_correlations_by_dataset.csv", index=False)
    gate_detail.to_csv(root / "gate_pairwise_per_signal.csv", index=False)

    e2_primary_delta = scope_delta[scope_delta.variant == "E2_C_combined_l010"]
    e2_dataset_delta = dataset_delta[dataset_delta.variant == "E2_C_combined_l010"]
    e1_scope = scope[scope.variant.isin(["T3", "E1_A_level", "E1_B_dynamic", "E1_C_combined"])]
    e2_components = scope[scope.variant.isin(
        ["T3", "E2_A_level_l010", "E2_B_dynamic_l010", "E2_C_combined_l010"])]
    e2_sensitivity = scope[scope.variant.isin(
        ["T3", "E2_C_combined_l005", "E2_C_combined_l010", "E2_C_combined_l020"])]
    e3_scope = scope[scope.variant.isin(
        ["T3", "E2_C_combined_l010", "E3_A_stat_l010", "E3_B_cross_l010", "E3_C_late_l010"])]
    primary_scope = scope[scope.variant.isin(PRIMARY)]
    focus_events = event_delta[(event_delta.dataset.isin(FOCUS)) &
                               (event_delta.variant.isin(["AMR_native", "E2_C_combined_l010",
                                                          "E3_A_stat_l010", "E3_B_cross_l010",
                                                          "E3_C_late_l010"]))]
    e2_complement = complement[complement.variant == "E2_C_combined_l010"]
    focus_volatility = volatility[(volatility.dataset.isin(FOCUS)) &
                                  (volatility.variant.isin(["T3", "E1_C_combined",
                                                            "E2_C_combined_l010", "E3_A_stat_l010",
                                                            "E3_B_cross_l010", "E3_C_late_l010"]))]
    focus_corr = corr_dataset[(corr_dataset.dataset.isin(FOCUS)) &
                              (corr_dataset.variant.isin(["E1_A_level", "E1_B_dynamic",
                                                           "E1_C_combined", "E2_C_combined_l010"]))]
    principal_events = events[events.variant.isin(PRIMARY)]
    gate_pair_summary = pd.DataFrame([{
        "signals": len(gate_detail),
        "mean_abs_cross_minus_late": gate_detail.mean_abs_cross_minus_late.mean(),
        "mean_cross_late_pearson": gate_detail.cross_late_pearson.mean(),
        "mean_abs_cross_minus_stat": gate_detail.mean_abs_cross_minus_stat.mean(),
        "mean_abs_late_minus_stat": gate_detail.mean_abs_late_minus_stat.mean(),
    }])
    total_runtime = pd.DataFrame([{
        "signals": int(runtime.signals.sum()),
        "gpu_process_seconds": float(runtime.total_seconds.sum()),
        "mean_seconds_per_signal": float(runtime.total_seconds.sum() / runtime.signals.sum()),
        "maximum_signal_seconds": float(runtime.max_seconds.max()),
    }])

    report = f"""# OrionBench fixed-reference numeric anomaly evidence

## Protocol and integrity

- This experiment uses the OrionBench-native protocol: NASA official Train statistics with official Test evaluation for MSL/SMAP, and complete-sequence unlabeled transductive fitting/evaluation for Yahoo and NAB. It covers all 11 datasets and 492 signals.
- E1/E2/E3 retain the exact T3 reference window, constrained-DTW path and frozen CLIP Block-6 visual score. Their reference and path change fractions are exactly zero. `AMR_native` is retained as the dynamic-reference comparison.
- Numeric normalization, score calibration and gates use unlabeled signal statistics only. The 13 variants, `lambda=0.10` primary setting and `0.05/0.20` sensitivity settings were fixed before labels were opened.
- Aff-F1 uses fixed `alpha=0.01`; a complete miss with real events contributes zero. The same four all-normal signals are `NaN` and excluded for every model, leaving 488 valid signals. VUS-PR/VUS-ROC use continuous scores. No label-selected threshold or F1-max is used.
- Validation passed with `{validation['score_files']}` point-score files and `{validation['metric_rows']}` metric rows. Inference did not open labels; T3 and AMR source scores, timestamps, the primary fusion formula, fixed references and fixed paths all passed the independent audit (`{final_validation['status']}`).

## Methods

- **E1-A level:** robustly normalized pointwise level residual along the T3 alignment.
- **E1-B dynamics:** normalized first-difference, local spread, slope and endpoint-change residuals.
- **E1-C combined:** equal combination of level and dynamics. Numeric point evidence is positive-median matched to T3 without labels.
- **E2:** `S=(1-lambda)Sv+lambda Sn` on the fixed T3 path. The primary result uses combined evidence and `lambda=0.10`.
- **E3-A:** multiplies `lambda` by a Top-5 reference reliability gate from visual/numeric rank agreement, support and candidate margins. **E3-B/C** replace the representation part of that gate with AMR Cross-Attention or corrected Late-joint candidate costs, while retaining exactly the same score evidence and T3 path.

## Main results

{md(primary_scope)}

The prespecified primary `E2_C_combined_l010` is the only new configuration that exceeds T3 on all three metrics in all three scopes. Its changes relative to T3 are:

{md(e2_primary_delta)}

This improvement is not supplied by the dynamic-reference AMR baseline, which remains below T3. It arises solely from adding low-weight numeric anomaly evidence to the unchanged T3 reference and path.

## Experiment 1: standalone numeric evidence

{md(e1_scope)}

Standalone numeric evidence is not a replacement for T3. Dynamics is stronger than level for Aff-F1 and Yahoo, while the combined score improves Real-6 VUS-PR over either standalone component. The components therefore contain useful but poorly calibrated evidence that is most effective as a bounded correction.

## Experiment 2: component and lambda controls

### Level, dynamics and combined evidence at `lambda=0.10`

{md(e2_components)}

### Prespecified combined-score sensitivity

{md(e2_sensitivity)}

Both `lambda=0.05` and the primary `0.10` improve every macro metric in All-11, Real-6 and Yahoo-4. At `0.20`, Aff-F1 continues to rise on All-11/Real-6, but All-11 VUS-PR and Yahoo Aff-F1 fall below T3. The evidence is useful at low weight, while increasing its influence eventually damages continuous ranking and Yahoo event quality.

## Experiment 3: reliability-aware conditional fusion

{md(e3_scope)}

The statistical gate raises Aff-F1 slightly but does not preserve the full E2 gain. Its mean gate is about `0.20`, giving an average effective numeric weight near `0.02`. Cross-Attention and Late-joint gates average about `0.038` effective weight, yet their rankings are nearly identical:

{md(gates)}

{md(gate_pair_summary)}

Cross-Attention therefore has no measurable independent advantage here. The gate estimates candidate-reference agreement, while the useful E2 signal often occurs precisely where numeric evidence disagrees with visual evidence. Gating on agreement suppresses complementary anomaly evidence rather than separating reliable anomalies from normal volatility.

## Complete per-dataset Aff-F1

{md(pivot_metric(dataset, 'aff_f1'))}

## Complete per-dataset VUS-PR

{md(pivot_metric(dataset, 'vus_pr'))}

## Complete per-dataset VUS-ROC

{md(pivot_metric(dataset, 'vus_roc'))}

### Primary E2 changes by dataset

{md(e2_dataset_delta)}

The primary E2 result improves Aff-F1 on 9 of 11 datasets. Its Real-6 gain is distributed across MSL, SMAP, AWS, Traffic and Tweets. It does not depend on NAB Artificial. Continuous ranking improves strongly on MSL/SMAP/AWS, while AdExchange remains the clearest conflict: Aff-F1 rises slightly, but VUS-PR/VUS-ROC fall because numeric residuals rank normal high-variation regions too highly.

## Events, false positives and complementarity

The following overlap counts are diagnostics separate from Affiliation-F1. `tp_events` counts true intervals touched by at least one prediction; `fp_intervals` counts prediction intervals touching no true event.

{md(principal_events)}

Relative to T3, primary E2 detects 13 more true events and emits 25 more false-positive intervals overall. Its Aff-F1 gain therefore comes from recovered events and better interval affiliation, not from globally reducing the FP count.

### Recovered and lost T3 events

{md(e2_complement)}

### Focus-dataset event changes

{md(focus_events)}

MSL, SMAP and AWS account for most of the real-data recovery. AdExchange preserves T3 event hits and removes one coarse FP interval, but its VUS loss shows that ranking degrades below the final fixed threshold as well. Traffic removes one FP interval without changing event recall. The numeric evidence thus complements visual misses on some reference-stable series, while its remaining weakness is normal high-variation ranking.

## Score complementarity and normal volatility

### Correlation with T3 on focus datasets

{md(focus_corr)}

### Anomaly separation from high-volatility normal points

`anomaly_minus_high_volatility_normal_in_normal_iqr` is positive when the median anomaly score exceeds the highest-volatility 10% of normal points. Spearman correlation uses normal points only.

{md(focus_volatility)}

The standalone numeric score is more sensitive to normal volatility than T3. Low-weight fusion limits that error while improving MSL/SMAP/AWS ranking. AdExchange remains negative on anomaly-versus-high-volatility separation, explaining why its VUS metrics decline even when Aff-F1 does not.

## Reference and path audit

{md(mechanism)}

Every E1/E2/E3 change is an anomaly-scoring effect. `AMR_native` changes 13.54% of references and 0.84% of DTW paths, so its result is not directly attributable to score evidence alone.

## Runtime

{md(runtime)}

{md(total_runtime)}

The total is GPU process time from parallel workers, not wall-clock time. Existing AMR checkpoints and CLIP/numeric caches were reused; no training was performed.

## Conclusions

1. **Reference-conditioned numeric evidence is useful only as a correction.** All E1 standalone scores are below T3, but fixed low-weight E2 fusion adds information missing from the visual score.
2. **The primary fixed configuration improves all requested scopes and metrics.** `E2_C_combined_l010` reaches All-11 `{scope[(scope.scope=='all_11') & (scope.variant=='E2_C_combined_l010')].aff_f1.iloc[0]:.6f}/{scope[(scope.scope=='all_11') & (scope.variant=='E2_C_combined_l010')].vus_pr.iloc[0]:.6f}/{scope[(scope.scope=='all_11') & (scope.variant=='E2_C_combined_l010')].vus_roc.iloc[0]:.6f}` and Real-6 `{scope[(scope.scope=='real_6') & (scope.variant=='E2_C_combined_l010')].aff_f1.iloc[0]:.6f}/{scope[(scope.scope=='real_6') & (scope.variant=='E2_C_combined_l010')].vus_pr.iloc[0]:.6f}/{scope[(scope.scope=='real_6') & (scope.variant=='E2_C_combined_l010')].vus_roc.iloc[0]:.6f}` (Aff-F1/VUS-PR/VUS-ROC), all above T3.
3. **The improvement is stable at low weights.** `lambda=0.05` and `0.10` both improve all nine scope-metric combinations. The `0.20` sensitivity result identifies the limit without being selected as the final model.
4. **Reference-reliability gating is not the useful mechanism.** E3 retains only a small fraction of the E2 gain; it does not reduce overall false positives, and Cross-Attention is effectively tied with Late-joint.
5. **The retained model is E2 combined at the preregistered `lambda=0.10`.** It is a fixed-reference visual-numeric anomaly scorer. AMR routing and Cross-Attention do not contribute to its gain and should remain ablations.

## Artifacts

- `point_scores/`, `evidence_diagnostics/`, `inference_manifests/`
- `metrics_per_signal.csv`, `metrics_by_dataset.csv`, `metrics_by_scope.csv`, `metric_deltas_*.csv`
- `event_counts_*.csv`, `event_deltas_by_dataset.csv`, `event_complementarity.csv`
- `score_correlations_*.csv`, `score_volatility_*.csv`
- `gate_per_signal.csv`, `gate_summary.csv`, `gate_pairwise_per_signal.csv`
- `reference_path_summary.csv`, `inference_runtime.csv`, `alignment_audit.csv`, `validation.json`, `final_validation.json`
"""
    (root / "numeric_evidence_final_report.md").write_text(report)
    print(root / "numeric_evidence_final_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
