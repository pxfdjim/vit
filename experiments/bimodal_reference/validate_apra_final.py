#!/usr/bin/env python3
"""Validate the streamlined E1 APRA implementation against immutable history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import DATASETS, signal_names
from src.models.apra import load_apra_checkpoint


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frame_error(current, historical, keys, current_variant="E1_APRA",
                historical_variant="APRA_only"):
    left = current[current.variant == current_variant].copy()
    right = historical[historical.variant == historical_variant].copy()
    index = [key for key in keys if key != "variant"]
    left = left.set_index(index).sort_index()
    right = right.set_index(index).sort_index()
    if not left.index.equals(right.index):
        raise RuntimeError(f"row index mismatch for {index}")
    columns = [column for column in ("aff_f1", "vus_pr", "vus_roc")
               if column in left and column in right]
    errors = []
    for column in columns:
        a = left[column].to_numpy(float)
        b = right[column].to_numpy(float)
        errors.append(float(np.nanmax(np.abs(a - b))))
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise RuntimeError(f"NaN mask mismatch for {column}")
    return max(errors, default=0.0), len(left)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    historical_root = Path(config["e1_result_root"])
    checkpoint_root = Path(config["e1_checkpoint_root"])
    data_root = Path(config["data_root"])

    checkpoint_errors = []
    state_digests = {}
    parameter_counts = {}
    for dataset in DATASETS:
        model, payload = load_apra_checkpoint(checkpoint_root / f"{dataset}.pt")
        current = model.state_dict()
        historical = payload["state_dict"]
        checkpoint_errors.append(max(float((current[key] - historical[key]).abs().max())
                                     for key in current))
        state_digests[dataset] = tensor_digest(current)
        parameter_counts[dataset] = sum(parameter.numel() for parameter in model.parameters())

    point_errors, delta_errors, timestamp_errors = [], [], []
    signals = 0
    points = 0
    for dataset in DATASETS:
        for signal in signal_names(data_root, dataset):
            signals += 1
            current_path = args.run / "point_scores" / dataset / signal / "E1_APRA.npz"
            historical_path = historical_root / "point_scores" / dataset / signal / "APRA_only.npz"
            with np.load(current_path) as current, np.load(historical_path) as historical:
                point_errors.append(float(np.max(np.abs(current["scores"] - historical["scores"]))))
                timestamp_errors.append(0.0 if np.array_equal(current["timestamps"], historical["timestamps"])
                                        else 1.0)
                points += len(current["scores"])
            with np.load(args.run / "feature_diagnostics" / dataset / signal / "evidence.npz") as current, \
                 np.load(historical_root / "feature_diagnostics" / dataset / signal / "evidence.npz") as historical:
                delta_errors.append(float(np.max(np.abs(
                    current["point_delta"] - historical["point_delta__APRA_only"]))))

    metric_checks = {}
    for name, keys in (("per_signal", ["dataset", "signal", "variant"]),
                       ("by_dataset", ["dataset", "variant"]),
                       ("by_scope", ["scope", "variant"])):
        current = pd.read_csv(args.run / f"metrics_{name}.csv")
        historical = pd.read_csv(historical_root / f"metrics_{name}.csv")
        error, rows = frame_error(current, historical, keys)
        metric_checks[name] = {"max_error": error, "rows": rows}

    current_events = pd.read_csv(args.run / "event_counts_per_signal.csv")
    historical_events = pd.read_csv(historical_root / "event_counts_per_signal.csv")
    current_events = current_events[current_events.variant == "E1_APRA"].drop(columns="variant")
    historical_events = historical_events[historical_events.variant == "APRA_only"].drop(columns="variant")
    event_index = ["dataset", "signal"]
    current_events = current_events.set_index(event_index).sort_index()
    historical_events = historical_events.set_index(event_index).sort_index()
    event_columns = ["true_events", "tp_events", "fn_events", "fp_intervals",
                     "predicted_intervals"]
    event_error = max(float(np.max(np.abs(current_events[column].to_numpy(float) -
                                           historical_events[column].to_numpy(float))))
                      for column in event_columns)

    module = torch.load(args.run / "module_equivalence.pt", map_location="cpu", weights_only=False)
    scope = pd.read_csv(args.run / "metrics_by_scope.csv")
    scope = scope[scope.variant == "E1_APRA"]
    event_all = pd.read_csv(args.run / "event_counts_all.csv")
    event_all = event_all[event_all.variant == "E1_APRA"].iloc[0]
    checks = {
        "checkpoint_load_max_error": max(checkpoint_errors),
        "state_dict_keys": 0,
        "parameter_counts": sorted(set(parameter_counts.values())),
        "point_score_max_error": max(point_errors),
        "calibrated_delta_max_error": max(delta_errors),
        "timestamp_max_error": max(timestamp_errors),
        "event_count_max_error": event_error,
        "module_state_max_error": module["state_dict_max_error"],
        "module_output_max_error": module["output_max_error"],
        "module_loss_max_error": module["loss_max_error"],
        "module_gradient_max_error": module["gradient_max_error"],
    }
    # Fix the key count without relying on the expression above in reports.
    first_model, _ = load_apra_checkpoint(checkpoint_root / f"{DATASETS[0]}.pt")
    checks["state_dict_keys"] = len(first_model.state_dict())
    passed = (all(value == 0 for key, value in checks.items()
                  if key.endswith("_error")) and checks["parameter_counts"] == [2059973]
              and signals == 492)
    validation = {
        "status": "passed" if passed else "failed",
        "signals": signals, "points": points, "checkpoints": len(DATASETS),
        "checks": checks, "metrics": metric_checks,
        "state_dict_digests": state_digests,
        "scope_results": scope[["scope", "aff_f1", "vus_pr", "vus_roc"]].to_dict("records"),
        "events": {key: int(event_all[key]) for key in event_columns},
        "history_preserved_at": str(historical_root),
        "labels_used_for_training_or_calibration": False,
    }
    (args.run / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
