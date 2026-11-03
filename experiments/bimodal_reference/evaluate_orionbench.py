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
                valid = len(scores) == len(frame) and np.array_equal(timestamps, frame.timestamp.to_numpy()) and np.isfinite(scores).all()
                alignment.append({"dataset": dataset, "signal": signal, "variant": variant,
                                  "points": len(scores), "valid": valid})
                base = {"dataset": dataset, "signal": signal, "variant": variant,
                        "points": len(scores), "anomaly_points": int(labels.sum()),
                        "checkpoint_protocol": metadata.get("checkpoint_protocol"),
                        "training_dataset": metadata.get("training_dataset")}
                if not valid:
                    raise RuntimeError(f"score alignment failed: {dataset}/{signal}/{variant}")
                if labels.min() == labels.max():
                    rows.append({**base, "aff_precision": np.nan, "aff_recall": np.nan,
                                 "aff_f1": np.nan, "vus_pr": np.nan, "vus_roc": np.nan,
                                 "undefined_reason": "single_class_ground_truth"})
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
                             "vus_max_window": vus.max_window, "undefined_reason": ""})
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
    for frame, name in ((per_signal, "metrics_per_signal.csv"), (by_dataset, "metrics_by_dataset.csv"),
                        (by_scope, "metrics_by_scope.csv"), (alignment_frame, "alignment_audit.csv"),
                        (mechanism, "mechanism_per_signal.csv"), (mechanism_summary, "mechanism_summary.csv"),
                        (training, "training_summary.csv")):
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
        row = {"scope": scope}
        for metric in ("aff_f1", "vus_pr", "vus_roc"):
            base = float(by_scope[(by_scope.scope == scope) & (by_scope.variant == "T3")][metric].iloc[0])
            model = float(by_scope[(by_scope.scope == scope) & (by_scope.variant == "AMR_native")][metric].iloc[0])
            row[f"AMR_minus_T3_{metric}"] = model - base
        deltas.append(row)
    delta_frame = pd.DataFrame(deltas)
    report = f"""# Dataset-native OrionBench AMR evaluation

## Protocol integrity

- MSL (27 signals) and SMAP (53 signals) use NASA official Train for unlabeled fitting and training, and official Test only for evaluation. The project concatenations were split at the exact official boundary and verified against the NASA arrays.
- Yahoo A1-A4 and the five NAB groups use each complete unlabeled sequence for training and evaluate the same complete sequence. No 80/20 split and no three-fold checkpoint are used.
- One AMR checkpoint is trained independently for each dataset. Every checkpoint is the fixed final epoch; labels are absent from training, checkpoint selection and configuration.
- NASA numerical and visual normalization parameters are fitted only on official Train. Inference candidates come only from the evaluation sequence and the unchanged overlap exclusion forbids the query and its three neighboring windows from serving as references.
- Aff-F1 uses the preregistered label-free `alpha=0.01` detector intervals. VUS-PR and VUS-ROC consume continuous scores. Contextual F1-max is not computed.
- Four single-class signals have undefined metrics and remain `NaN`; valid counts are reported explicitly.

## Scope macro metrics

{md(scope_display)}

## AMR-native change relative to native-protocol T3

{md(delta_frame)}

## Complete per-dataset metrics

{md(dataset_display)}

## Reference and DTW changes relative to T3

{md(mechanism_summary)}

## Dataset-specific training cost

{md(training)}

## Interpretation

The tables above are the complete prespecified result set. AMR is considered a stable improvement only when its changes versus T3 have the same useful direction on All-11 and Real-6 without relying solely on Yahoo. The report intentionally does not select a model, threshold, epoch or hyperparameter from these evaluation labels.

## Artifacts

- `data_protocol_per_signal.csv`, `data_protocol_by_dataset.csv`, `data_protocol_validation.json`
- `normalization_stats.json`, dataset training caches and eleven immutable checkpoints
- `point_scores/`, `path_diagnostics/`, `metrics_per_signal.csv`
- `metrics_by_dataset.csv`, `metrics_by_scope.csv`, `mechanism_*.csv`
- `training_summary.csv`, `alignment_audit.csv`, `validation.json`
"""
    (args.output_root / "amr_orionbench_native_final_report.md").write_text(report)
    print(json.dumps(validation, indent=2)); print(scope_display.to_string(index=False))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
