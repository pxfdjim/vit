#!/usr/bin/env python3
"""Independent completeness and integrity validation for RCCA/CADM."""
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
from experiments.bimodal_reference.run_rcca_cadm_inference import output_variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); root = args.output_root
    variants = output_variants(cfg); errors = []; checks = {}
    expected_signals = sum(len(signal_names(Path(cfg["data_root"]), dataset))
                           for dataset in cfg["datasets"])
    checkpoints = list((root / "checkpoints").glob("*/*/*.pt"))
    if len(checkpoints) != 66:
        errors.append(f"checkpoint count {len(checkpoints)} != 66")
    checkpoint_violations = []; gradient_violations = []; minimum_ref_effect = np.inf
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["labels_used"] or payload["validation_split_used"] or \
                payload["checkpoint_selection"] != "fixed_final_epoch":
            checkpoint_violations.append(str(path))
        gradients = payload["gradient_audit"]
        if gradients["numeric"] <= 0 or gradients["cadm"] <= 0:
            gradient_violations.append(f"{payload['variant']}/{payload['seed']}/{payload['dataset']}: core")
        if payload["variant"] != "M2_no_rcca" and gradients["rcca"] <= 0:
            gradient_violations.append(f"{payload['variant']}/{payload['seed']}/{payload['dataset']}: rcca")
        if payload["variant"] == "M2_no_rcca" and gradients["rcca"] != 0:
            gradient_violations.append(f"{payload['variant']}/{payload['seed']}/{payload['dataset']}: unexpected")
        minimum_ref_effect = min(minimum_ref_effect, payload["sensitivity_audit"][
            "reference_replacement_mean_abs_logit_change"])
    if checkpoint_violations:
        errors.append(f"label/selection violations: {len(checkpoint_violations)}")
    if gradient_violations:
        errors.append(f"gradient violations: {gradient_violations[:5]}")
    if minimum_ref_effect <= 1e-8:
        errors.append(f"checkpoint reference effect too small: {minimum_ref_effect}")
    checks.update({"checkpoints": len(checkpoints), "minimum_training_reference_effect": minimum_ref_effect})

    caches = list((root / "synthetic_training_cache").glob("seed_*/*.pt"))
    cache_violations = []; excluded_short_windows = 0
    for path in caches:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["labels_read"] or not cached["uses_all_windows"] or not cached["t3_reference_fixed"]:
            cache_violations.append(str(path))
        if torch.any((cached["reference_ids"] - torch.arange(cached["windows"])).abs() < 4):
            cache_violations.append(f"overlap:{path}")
        excluded_short_windows += int(cached.get("excluded_short_windows", 0))
    if len(caches) != 33:
        errors.append(f"cache count {len(caches)} != 33")
    if cache_violations:
        errors.append(f"cache protocol violations: {cache_violations[:5]}")
    checks.update({"training_caches": len(caches),
                   "excluded_short_training_windows_across_seed_caches": excluded_short_windows})

    score_paths = list((root / "point_scores").glob("*/*/*.npz"))
    expected_scores = expected_signals * len(variants)
    if len(score_paths) != expected_scores:
        errors.append(f"point score count {len(score_paths)} != {expected_scores}")
    metrics = pd.read_csv(root / "metrics_per_signal.csv")
    if len(metrics) != expected_scores:
        errors.append(f"metric rows {len(metrics)} != {expected_scores}")
    if metrics.groupby(["dataset", "signal"]).variant.nunique().min() != len(variants):
        errors.append("incomplete variant coverage")
    valid = metrics.groupby("variant").aff_f1.apply(lambda values: int(np.isfinite(values).sum()))
    if not valid.eq(488).all():
        errors.append(f"common valid signal mismatch: {valid.to_dict()}")
    alignment = pd.read_csv(root / "alignment_audit.csv")
    if len(alignment) != expected_scores or not alignment.valid.all():
        errors.append("score/timestamp alignment failed")
    checks.update({"point_scores": len(score_paths), "metric_rows": len(metrics),
                   "valid_signals_per_variant": valid.to_dict()})

    worker_manifest = json.loads((root / "inference_workers.json").read_text())
    if worker_manifest["status"] != "passed":
        errors.append("inference worker manifest failed")
    full_manifests = [json.loads(path.read_text()) for path in sorted(
        (root / "inference_manifests").glob("worker_*_of_08.json"))]
    if len(full_manifests) != 8 or any(item["status"] != "passed" for item in full_manifests):
        errors.append("full inference manifests missing or failed")
    records = [record for manifest in full_manifests for record in manifest["records"]]
    if len(records) != expected_signals:
        errors.append(f"inference records {len(records)} != {expected_signals}")
    passed_records = [record for record in records if record["status"] == "passed"]
    smoke = json.loads((root / "inference_manifests/worker_00_of_01.json").read_text())
    passed_records += [record for record in smoke["records"] if record["status"] == "passed"]
    min_m5 = min(record["m5_reference_effect"] for record in passed_records)
    max_m6 = max(record["m6_formula_error"] for record in passed_records)
    if min_m5 <= 1e-8:
        errors.append(f"M5 reference replacement effect {min_m5}")
    if max_m6 != 0:
        errors.append(f"M6 formula error {max_m6}")
    if any(record["t3_reference_error"] != 0 or record["t3_path_error"] != 0
           for record in passed_records):
        errors.append("fixed T3 reference/path reproduction failed")
    checks.update({"minimum_M5_reference_effect": min_m5,
                   "maximum_M6_formula_error": max_m6,
                   "inference_records": len(records)})

    source_roots = {
        "T3": (Path(cfg["feature_scoring_run"]), "T3"),
        "E2": (Path(cfg["feature_scoring_run"]), "E2_T3_raw_l010"),
        "B3": (Path(cfg["feature_scoring_run"]), "B3_raw_numeric_l010"),
        "AMR_native": (Path(cfg["feature_scoring_run"]), "AMR_native"),
        "C2_simplified": (Path(cfg["training_ablation_run"]), "C2_no_cross"),
    }
    reproduction = {}
    for variant, (source_root, source_name) in source_roots.items():
        maximum = 0.0
        for dataset in cfg["datasets"]:
            for signal in signal_names(Path(cfg["data_root"]), dataset):
                with np.load(root / "point_scores" / dataset / signal / f"{variant}.npz") as got, \
                     np.load(source_root / "point_scores" / dataset / signal / f"{source_name}.npz") as src:
                    maximum = max(maximum, float(np.max(np.abs(got["scores"] - src["scores"]))))
        reproduction[variant] = maximum
        if maximum != 0:
            errors.append(f"baseline reproduction failed: {variant} {maximum}")
    checks["baseline_reproduction_max_abs_errors"] = reproduction
    diagnostics = pd.read_csv(root / "inference_diagnostics_per_signal.csv")
    if not np.isfinite(diagnostics["cadm_logit_mean"]).all():
        errors.append("nonfinite CADM diagnostics")
    result = {"status": "passed" if not errors else "failed", "errors": errors,
              "checks": checks, "labels_used_for_training_or_selection": False}
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
