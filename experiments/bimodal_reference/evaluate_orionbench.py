#!/usr/bin/env python3
"""Evaluate dataset-native AMR with Aff-F1, VUS-PR and VUS-ROC only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.mindts_metrics import MINDTS_COMMIT, affiliation_scores, fast_vus
from experiments.bimodal_reference.orionbench_protocol import (
    DATASETS, evaluation_frame, labels_for, load_truth, signal_names,
)
from src.models.model_utils import compute_detection_intervals


def md(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mindts-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    variants = config["variants_order"]
    truth = load_truth(args.data_root)
    rows, alignment = [], []
    for dataset in DATASETS:
        for signal in signal_names(args.data_root, dataset):
            frame = evaluation_frame(args.data_root, dataset, signal)
            labels = labels_for(frame, truth[signal])
            for variant in variants:
                path = args.output_root / "point_scores" / dataset / signal / f"{variant}.npz"
                with np.load(path, allow_pickle=False) as stored:
                    scores = stored["scores"].astype(np.float64)
                    timestamps = stored["timestamps"]
                    metadata = json.loads(str(stored["metadata_json"][0]))
                    timing = json.loads(str(stored["timing_json"][0]))
                valid = len(scores) == len(frame) and np.array_equal(timestamps, frame.timestamp.to_numpy()) and np.isfinite(scores).all()
                alignment.append({"dataset": dataset, "signal": signal, "variant": variant,
                                  "points": len(scores), "valid": valid})
                base = {"dataset": dataset, "signal": signal, "variant": variant,
                        "points": len(scores), "anomaly_points": int(labels.sum()),
                        "checkpoint_protocol": metadata.get("checkpoint_protocol"),
                        "training_dataset": metadata.get("training_dataset"),
                        "shared_inference_seconds": timing.get("shared_encoding_and_cost_seconds"),
                        "variant_inference_seconds": timing.get("inference_seconds")}
                if not valid:
                    raise RuntimeError(f"score alignment failed: {dataset}/{signal}/{variant}")
                if labels.min() == labels.max():
                    rows.append({**base, "aff_precision": np.nan, "aff_recall": np.nan,
                                 "aff_f1": np.nan, "vus_pr": np.nan, "vus_roc": np.nan,
                                 "undefined_reason": "single_class_ground_truth",
                                 "aff_undefined_reason": "single_class_ground_truth",
                                 "vus_undefined_reason": "single_class_ground_truth"})
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
                precision, recall, aff_f1 = affiliation_scores(labels, predictions, args.mindts_root)
                vus = fast_vus(labels, scores)
                rows.append({**base, "aff_precision": precision, "aff_recall": recall,
                             "aff_f1": aff_f1, "vus_pr": vus.vus_pr, "vus_roc": vus.vus_roc,
                             "aff_threshold": float(threshold), "predicted_intervals": len(intervals),
                             "vus_max_window": vus.max_window, "undefined_reason": "",
                             "aff_undefined_reason": "" if np.isfinite(aff_f1) else
                             ("no_predicted_intervals" if len(intervals) == 0 else "mindts_affiliation_undefined"),
                             "vus_undefined_reason": "" if np.isfinite(vus.vus_pr) and np.isfinite(vus.vus_roc)
                             else "mindts_vus_undefined"})
    per_signal = pd.DataFrame(rows)
    alignment_frame = pd.DataFrame(alignment)
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
        for variant in variants:
            group = by_dataset[(by_dataset.dataset.isin(datasets)) & (by_dataset.variant == variant)]
            row = {"scope": scope, "variant": variant, "datasets": len(group)}
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                row[metric] = float(np.nanmean(group[metric]))
                row[f"{metric}_valid_signals"] = int(group[f"{metric}_valid_signals"].sum())
            scope_rows.append(row)
    by_scope = pd.DataFrame(scope_rows)
    mechanism_rows = []
    for path in sorted((args.output_root / "path_diagnostics").glob("*/*/*.npz")):
        with np.load(path) as stored:
            mechanism_rows.append({"dataset": path.parts[-3], "signal": path.parts[-2], "variant": path.stem,
                                   "windows": len(stored["reference_window_ids"]),
                                   "reference_change_fraction": float(stored["reference_changed"].mean()),
                                   "path_change_fraction": float(stored["path_changed"].mean()),
                                   "mean_visual_local_cost": float(stored["visual_local_costs"].mean())})
    mechanism = pd.DataFrame(mechanism_rows)
    mechanism_summary = mechanism.groupby("variant", as_index=False).agg(
        signals=("signal", "count"), windows=("windows", "sum"),
        reference_change_fraction=("reference_change_fraction", "mean"),
        path_change_fraction=("path_change_fraction", "mean"),
        mean_visual_local_cost=("mean_visual_local_cost", "mean"),
    )
    training_rows = []
    for dataset, path in config["model"]["checkpoints"].items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        training_rows.append({"dataset": dataset, "signals": payload["signals"],
                              "train_windows": payload["train_windows"],
                              "epochs": len(payload["history"]), "final_loss": payload["history"][-1]["total"],
                              "training_seconds": payload["elapsed_seconds"],
                              "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
                              "labels_used": payload["labels_used"],
                              "validation_split_used": payload["validation_split_used"]})
    training = pd.DataFrame(training_rows)
    protocol = pd.read_csv(args.output_root / "data_protocol_by_dataset.csv")
    validity_rows = []
    for variant, group in per_signal.groupby("variant", sort=False):
        validity_rows.append({
            "variant": variant,
            "signals": len(group),
            "single_class_signals": int((group.undefined_reason == "single_class_ground_truth").sum()),
            "aff_no_prediction_signals": int((group.aff_undefined_reason == "no_predicted_intervals").sum()),
            "aff_valid_signals": int(np.isfinite(group.aff_f1).sum()),
            "vus_pr_valid_signals": int(np.isfinite(group.vus_pr).sum()),
            "vus_roc_valid_signals": int(np.isfinite(group.vus_roc).sum()),
        })
    validity = pd.DataFrame(validity_rows)
    inference_records = []
    for path in sorted((args.output_root / "inference_manifests").glob("*.json")):
        inference_records.extend(json.loads(path.read_text())["records"])
    inference_per_signal = pd.DataFrame(inference_records)
    inference_runtime = inference_per_signal.groupby("dataset", as_index=False).agg(
        signals=("signal", "count"), total_seconds=("elapsed_seconds", "sum"),
        mean_seconds=("elapsed_seconds", "mean"), max_seconds=("elapsed_seconds", "max"),
    )
    inference_runtime = pd.concat([
        inference_runtime,
        pd.DataFrame([{"dataset": "ALL", "signals": len(inference_per_signal),
                       "total_seconds": inference_per_signal.elapsed_seconds.sum(),
                       "mean_seconds": inference_per_signal.elapsed_seconds.mean(),
                       "max_seconds": inference_per_signal.elapsed_seconds.max()}]),
    ], ignore_index=True)
    for frame, name in ((per_signal, "metrics_per_signal.csv"), (by_dataset, "metrics_by_dataset.csv"),
                        (by_scope, "metrics_by_scope.csv"), (alignment_frame, "alignment_audit.csv"),
                        (mechanism, "mechanism_per_signal.csv"), (mechanism_summary, "mechanism_summary.csv"),
                        (training, "training_summary.csv"), (validity, "metric_validity_summary.csv"),
                        (inference_runtime, "inference_runtime.csv")):
        frame.to_csv(args.output_root / name, index=False)
    complete = (len(per_signal) == 492 * len(variants) and alignment_frame.valid.all()
                and len(by_dataset) == 11 * len(variants) and len(training) == 11
                and not training.labels_used.any() and not training.validation_split_used.any())
    validation = {"status": "passed" if complete else "failed", "signals": 492,
                  "variants": len(variants), "metric_rows": len(per_signal),
                  "single_class_rows": int((per_signal.undefined_reason == "single_class_ground_truth").sum()),
                  "score_files": len(list((args.output_root / "point_scores").glob("*/*/*.npz"))),
                  "path_files": len(list((args.output_root / "path_diagnostics").glob("*/*/*.npz"))),
                  "f1_max_computed": False, "mindts_commit": MINDTS_COMMIT,
                  "code_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()}
    (args.output_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    scope_display = by_scope[["scope", "variant", "aff_f1", "vus_pr", "vus_roc",
                              "aff_f1_valid_signals", "vus_pr_valid_signals", "vus_roc_valid_signals"]]
    dataset_display = by_dataset[["dataset", "variant", "aff_f1", "vus_pr", "vus_roc",
                                  "aff_f1_valid_signals", "vus_pr_valid_signals", "vus_roc_valid_signals"]]
    comparison = by_scope.pivot(index="scope", columns="variant", values=["aff_f1", "vus_pr", "vus_roc"])
    deltas = []
    for scope in config["macro_scopes"]:
        for variant in variants:
            if variant == "T3":
                continue
            row = {"scope": scope, "variant": variant}
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                base = float(by_scope[(by_scope.scope == scope) & (by_scope.variant == "T3")][metric].iloc[0])
                model = float(by_scope[(by_scope.scope == scope) & (by_scope.variant == variant)][metric].iloc[0])
                row[f"delta_{metric}"] = model - base
            deltas.append(row)
    delta_frame = pd.DataFrame(deltas)
    dataset_deltas = []
    for dataset in DATASETS:
        for variant in variants:
            if variant == "T3":
                continue
            row = {"dataset": dataset, "variant": variant}
            for metric in ("aff_f1", "vus_pr", "vus_roc"):
                base = float(by_dataset[(by_dataset.dataset == dataset) & (by_dataset.variant == "T3")][metric].iloc[0])
                model = float(by_dataset[(by_dataset.dataset == dataset) & (by_dataset.variant == variant)][metric].iloc[0])
                row[f"delta_{metric}"] = model - base
            dataset_deltas.append(row)
    dataset_delta_frame = pd.DataFrame(dataset_deltas)
    dataset_delta_frame.to_csv(args.output_root / "metric_deltas_by_dataset.csv", index=False)
    delta_frame.to_csv(args.output_root / "metric_deltas_by_scope.csv", index=False)
    amr_late = []
    for scope in config["macro_scopes"]:
        row = {"scope": scope}
        for metric in ("aff_f1", "vus_pr", "vus_roc"):
            amr = float(by_scope[(by_scope.scope == scope) & (by_scope.variant == "AMR_native")][metric].iloc[0])
            late = float(by_scope[(by_scope.scope == scope) & (by_scope.variant == "Late_joint_native")][metric].iloc[0])
            row[f"AMR_minus_Late_{metric}"] = amr - late
        amr_late.append(row)
    amr_late_frame = pd.DataFrame(amr_late)
    amr_late_frame.to_csv(args.output_root / "amr_minus_late_joint.csv", index=False)
    amr_dataset = dataset_delta_frame[dataset_delta_frame.variant == "AMR_native"]
    report = f"""# Dataset-native OrionBench AMR evaluation

## Protocol integrity

- MSL (27 signals) and SMAP (53 signals) use NASA official Train for unlabeled fitting and training, and official Test only for evaluation. The project concatenations were split at the exact official boundary and verified against the NASA arrays.
- Yahoo A1-A4 and the five NAB groups use each complete unlabeled sequence for training and evaluate the same complete sequence. No 80/20 split and no three-fold checkpoint are used.
- One AMR checkpoint is trained independently for each dataset. Every checkpoint is the fixed final epoch; labels are absent from training, checkpoint selection and configuration.
- NASA numerical and visual normalization parameters are fitted only on official Train. Inference candidates come only from the evaluation sequence and the unchanged overlap exclusion forbids the query and its three neighboring windows from serving as references.
- Aff-F1 uses the preregistered label-free `alpha=0.01` detector intervals. VUS-PR and VUS-ROC consume continuous scores. Contextual F1-max is not computed.
- Four single-class signals have undefined metrics and remain `NaN`. Aff-F1 is also undefined when the fixed label-free threshold emits no prediction interval; these rows remain `NaN` rather than being filled with zero. Both causes and valid counts are reported explicitly.

## Verified dataset boundaries

{md(protocol)}

## Scope macro metrics

{md(scope_display)}

## Changes relative to native-protocol T3

{md(delta_frame)}

## Cross-Attention contribution relative to Late-joint

{md(amr_late_frame)}

## Complete per-dataset metrics

{md(dataset_display)}

### AMR-native per-dataset change relative to T3

{md(amr_dataset)}

## Reference and DTW changes relative to T3

{md(mechanism_summary)}

## Dataset-specific training cost

{md(training)}

The eleven fixed trainings total `{training.training_seconds.sum():.2f}` GPU-process seconds. Their maximum trainable-model CUDA allocation is `{int(training.peak_cuda_memory_bytes.max())}` bytes. Times from simultaneously scheduled jobs are process times and must not be summed as wall-clock duration.

## Full inference cost

{md(inference_runtime)}

## Metric validity

{md(validity)}

## Findings

- **T3 remains the strongest overall model.** AMR-native changes All-11 by `{delta_frame[(delta_frame.scope == 'all_11') & (delta_frame.variant == 'AMR_native')].delta_aff_f1.iloc[0]:+.6f}` Aff-F1, `{delta_frame[(delta_frame.scope == 'all_11') & (delta_frame.variant == 'AMR_native')].delta_vus_pr.iloc[0]:+.6f}` VUS-PR and `{delta_frame[(delta_frame.scope == 'all_11') & (delta_frame.variant == 'AMR_native')].delta_vus_roc.iloc[0]:+.6f}` VUS-ROC. All three changes are negative.
- **Dataset-native training does not repair Real-6 transfer.** AMR is lower than T3 on Real-6 by `{abs(delta_frame[(delta_frame.scope == 'real_6') & (delta_frame.variant == 'AMR_native')].delta_aff_f1.iloc[0]):.6f}` Aff-F1, `{abs(delta_frame[(delta_frame.scope == 'real_6') & (delta_frame.variant == 'AMR_native')].delta_vus_pr.iloc[0]):.6f}` VUS-PR and `{abs(delta_frame[(delta_frame.scope == 'real_6') & (delta_frame.variant == 'AMR_native')].delta_vus_roc.iloc[0]):.6f}` VUS-ROC. Its MSL and Traffic Aff-F1 gains are offset principally by AdExchange and AWS losses.
- **The clearest temporal-only gains are dataset specific.** TS-only improves all three metrics on MSL and SMAP, but changes half of the references and is substantially worse on AdExchange and Traffic. It therefore does not supply a stable general replacement for visual matching.
- **Cross-Attention has no stable independent advantage over corrected Late-joint.** AMR changes fewer references (`{mechanism_summary[mechanism_summary.variant == 'AMR_native'].reference_change_fraction.iloc[0]:.4f}` versus `{mechanism_summary[mechanism_summary.variant == 'Late_joint_native'].reference_change_fraction.iloc[0]:.4f}`), but its AMR-minus-Late deltas change sign by scope and metric. On Yahoo it adds VUS-PR while reducing VUS-ROC; on Real-6 it raises Aff-F1 slightly while reducing both VUS metrics.
- **Yahoo alone does not justify the model.** AMR is nearly tied on Yahoo Aff-F1, adds `{delta_frame[(delta_frame.scope == 'yahoo_4') & (delta_frame.variant == 'AMR_native')].delta_vus_pr.iloc[0]:+.6f}` VUS-PR, and loses `{abs(delta_frame[(delta_frame.scope == 'yahoo_4') & (delta_frame.variant == 'AMR_native')].delta_vus_roc.iloc[0]):.6f}` VUS-ROC. These small mixed changes do not compensate for the All-11 and Real-6 losses.

## Final conclusion

Under the requested OrionBench-native protocol, independently training one unlabeled AMR per dataset does **not** outperform T3. The result rules out mixed-dataset training and three-fold checkpoint ownership as the main cause of the earlier AMR deficit. Reference routing remains the limiting mechanism: the numeric branch can help MSL/SMAP, but its benefits are not consistent across real datasets, and Cross-Attention does not add a stable ranking gain over independent late fusion. No evaluation label, epoch, threshold or hyperparameter is selected from these outcomes.

## Artifacts

- `data_protocol_per_signal.csv`, `data_protocol_by_dataset.csv`, `data_protocol_validation.json`
- `normalization_stats.json`, dataset training caches and eleven immutable checkpoints
- `point_scores/`, `path_diagnostics/`, `metrics_per_signal.csv`
- `metrics_by_dataset.csv`, `metrics_by_scope.csv`, `mechanism_*.csv`
- `metric_deltas_by_dataset.csv`, `metric_deltas_by_scope.csv`, `amr_minus_late_joint.csv`
- `training_summary.csv`, `inference_runtime.csv`, `metric_validity_summary.csv`
- `alignment_audit.csv`, `validation.json`
"""
    (args.output_root / "amr_orionbench_native_final_report.md").write_text(report)
    print(json.dumps(validation, indent=2)); print(scope_display.to_string(index=False))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
