#!/usr/bin/env python3
"""Independent completeness and integrity checks for AMR training ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import signal_names


def numeric_state(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {name: value for name, value in payload["state_dict"].items()
            if name.startswith("numeric.")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()); root = args.output_root
    errors = []; checks = {}
    for stage in ("initial", "loss_ablation"):
        path = root / f"training_manifest_{stage}.json"
        if not path.is_file() or json.loads(path.read_text()).get("status") != "passed":
            errors.append(f"missing or failed training manifest: {stage}")
    trained = list(cfg["training_variants"])
    missing_checkpoints = []
    labels_or_selection = []
    for variant in trained:
        for dataset in cfg["datasets"]:
            path = root / "checkpoints" / variant / f"{dataset}.pt"
            if not path.is_file():
                missing_checkpoints.append(str(path)); continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["labels_used"] is not False or payload["validation_split_used"] is not False:
                labels_or_selection.append(f"{variant}/{dataset}")
    if missing_checkpoints: errors.append(f"missing checkpoints: {len(missing_checkpoints)}")
    if labels_or_selection: errors.append(f"label/selection violations: {labels_or_selection}")
    checks["checkpoints"] = len(trained) * len(cfg["datasets"]) - len(missing_checkpoints)

    expected_signals = sum(len(signal_names(Path(cfg["data_root"]), dataset)) for dataset in cfg["datasets"])
    expected_scores = expected_signals * len(cfg["all_variants"])
    score_paths = list((root / "point_scores").glob("*/*/*.npz"))
    if len(score_paths) != expected_scores:
        errors.append(f"point score count {len(score_paths)} != {expected_scores}")
    checks["point_scores"] = len(score_paths)
    metrics = pd.read_csv(root / "metrics_per_signal.csv")
    if len(metrics) != expected_scores:
        errors.append(f"metric row count {len(metrics)} != {expected_scores}")
    if metrics.groupby(["dataset", "signal"]).variant.nunique().min() != len(cfg["all_variants"]):
        errors.append("incomplete per-signal variant coverage")
    checks["metric_rows"] = len(metrics)

    manifests = [json.loads(path.read_text()) for path in
                 sorted((root / "inference_manifests").glob("*_all.json"))]
    if not manifests or any(item.get("status") != "passed" for item in manifests):
        errors.append("missing or failed all-stage inference manifests")
    records = [record for item in manifests for record in item["records"] if record["status"] == "passed"]
    if len(records) != expected_signals:
        errors.append(f"inference records {len(records)} != {expected_signals}")
    reproduction = max((record["full_b3_reproduction_error"] for record in records), default=np.inf)
    if reproduction > 1e-12:
        errors.append(f"Full B3 reproduction error {reproduction}")
    checks["full_b3_max_abs_error"] = reproduction

    duplicate_pairs = (("Full", "C1_full"), ("T2_numeric_ssl", "C4_numeric_only"),
                       ("T1_random", "L5_no_alignment"))
    duplicate_errors = {}
    for left, right in duplicate_pairs:
        maximum = 0.0
        for dataset in cfg["datasets"]:
            for signal in signal_names(Path(cfg["data_root"]), dataset):
                with np.load(root / "point_scores" / dataset / signal / f"{left}.npz") as a, \
                     np.load(root / "point_scores" / dataset / signal / f"{right}.npz") as b:
                    maximum = max(maximum, float(np.max(np.abs(a["scores"] - b["scores"]))))
        duplicate_errors[f"{left}_vs_{right}"] = maximum
        if maximum != 0:
            errors.append(f"controlled duplicate differs: {left}/{right} {maximum}")
    checks["controlled_score_duplicate_max_abs_errors"] = duplicate_errors

    state_errors = {}
    for dataset in cfg["datasets"]:
        t1 = numeric_state(root / "checkpoints/T1_random" / f"{dataset}.pt")
        l5 = numeric_state(root / "checkpoints/L5_no_alignment" / f"{dataset}.pt")
        maximum = max(float((t1[name] - l5[name]).abs().max()) for name in t1)
        state_errors[dataset] = maximum
    if max(state_errors.values()) != 0:
        errors.append("T1/L5 numeric initialization equality failed")
    checks["t1_l5_numeric_state_max_abs_errors"] = state_errors

    gradients = pd.read_csv(root / "gradient_path_audit.csv")
    expected_zero = {
        "T1_random": ("numeric_transformer", "cross_attention", "fusion", "reconstruction_head"),
        "C2_no_cross": ("cross_attention", "fusion"),
        "C3_late_joint": ("cross_attention",),
        "L1_no_patch": ("cross_attention", "fusion"),
        "L5_no_alignment": ("numeric_transformer", "cross_attention", "fusion", "reconstruction_head"),
    }
    for variant, groups in expected_zero.items():
        subset = gradients[gradients.variant == variant]
        for group in groups:
            if (subset[group].fillna(0).abs() > 0).any():
                errors.append(f"unexpected {group} gradient in {variant}")
    if not (gradients[gradients.variant == "T2_numeric_ssl"].numeric_transformer > 0).any():
        errors.append("T2 has no numeric Transformer gradient")
    if not (gradients[gradients.variant == "C3_late_joint"].fusion > 0).any():
        errors.append("C3 has no Late-joint fusion gradient")
    checks["gradient_audit_rows"] = len(gradients)

    result = {"status": "passed" if not errors else "failed", "errors": errors,
              "checks": checks, "labels_used": False}
    (root / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
