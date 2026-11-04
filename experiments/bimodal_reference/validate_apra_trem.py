#!/usr/bin/env python3
"""Independent completeness, leakage, gradient and score validation."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import signal_names
from experiments.bimodal_reference.run_apra_trem_inference import output_variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); root = args.output_root
    variants = output_variants(config); errors = []; checks = {}
    signals = sum(len(signal_names(Path(config["data_root"]), dataset))
                  for dataset in config["datasets"])
    checkpoints = list((root / "checkpoints").glob("*/*.pt"))
    expected_checkpoints = len(config["variants"]) * len(config["datasets"])
    if len(checkpoints) != expected_checkpoints:
        errors.append(f"checkpoint count {len(checkpoints)} != {expected_checkpoints}")
    gradient_errors = []; integrity_errors = []; initial_errors = []
    input_effect_errors = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["labels_used"] or payload["validation_split_used"] or
                payload["checkpoint_selection"] != "fixed_final_epoch"):
            integrity_errors.append(str(path))
        if (not payload["zero_initialized_trem"] or
                payload["initial_equivalence_max_error"] != 0):
            initial_errors.append(str(path))
        if payload["active_losses"] != ["path", "detect"]:
            integrity_errors.append(str(path))
        path_gradient = payload["gradient_audit_by_loss"]["path"]
        detect_gradient = payload["detect_gradient_after_one_step"]
        if path_gradient["apra"] <= 0:
            gradient_errors.append(f"{path}:L_path/APRA")
        if detect_gradient["apra"] <= 0 or detect_gradient["trem"] <= 0:
            gradient_errors.append(f"{path}:L_detect")
        if path_gradient["visual"] != 0 or detect_gradient["visual"] != 0:
            gradient_errors.append(f"{path}:frozen visual")
        if payload["residual_input_effect"] <= 0 or payload["reliability_input_effect"] <= 0:
            input_effect_errors.append(str(path))
    if integrity_errors: errors.append(f"selection/leakage failures {len(integrity_errors)}")
    if initial_errors: errors.append(f"zero/equivalence failures {len(initial_errors)}")
    if gradient_errors: errors.append(f"gradient failures {gradient_errors[:8]}")
    if input_effect_errors: errors.append(f"TREM input effect failures {input_effect_errors[:8]}")
    checks.update({
        "checkpoints": len(checkpoints),
        "zero_initialized_and_E1_equivalent": len(checkpoints) - len(initial_errors),
        "gradient_valid_checkpoints": len(checkpoints) - len(gradient_errors),
        "input_effect_valid_checkpoints": len(checkpoints) - len(input_effect_errors),
    })
    score_paths = list((root / "point_scores").glob("*/*/*.npz"))
    expected_scores = signals * len(variants)
    if len(score_paths) != expected_scores:
        errors.append(f"score files {len(score_paths)} != {expected_scores}")
    metrics = pd.read_csv(root / "metrics_per_signal.csv")
    if len(metrics) != expected_scores:
        errors.append(f"metric rows {len(metrics)} != {expected_scores}")
    if metrics.groupby(["dataset", "signal"]).variant.nunique().min() != len(variants):
        errors.append("variant coverage incomplete")
    valid = metrics.groupby("variant").aff_f1.apply(lambda values: int(np.isfinite(values).sum()))
    if not valid.eq(488).all():
        errors.append(f"common valid mismatch {valid.to_dict()}")
    alignment = pd.read_csv(root / "alignment_audit.csv")
    if len(alignment) != expected_scores or not alignment.valid.all():
        errors.append("timestamp/score alignment")
    manifests = [json.loads(path.read_text()) for path in sorted(
        (root / "inference_manifests").glob("worker_*_of_08.json"))]
    if len(manifests) != 8 or any(item["status"] != "passed" for item in manifests):
        errors.append("inference manifests")
    records = [record for manifest in manifests for record in manifest["records"]]
    if (len(records) != signals or any(record.get("t3_reference_error", 0) != 0 or
                                       record.get("t3_path_error", 0) != 0 for record in records)):
        errors.append("fixed T3 audit")
    sources = {
        "E0_B3": (Path(config["feature_scoring_run"]), "B3_raw_numeric_l010"),
        "E1_APRA_only": (Path(config["apra_run"]), "APRA_only"),
    }
    reproduction = {}
    for name, (source_root, source_name) in sources.items():
        maximum = 0.0
        for dataset in config["datasets"]:
            for signal in signal_names(Path(config["data_root"]), dataset):
                with np.load(root / "point_scores" / dataset / signal / f"{name}.npz") as got, \
                     np.load(source_root / "point_scores" / dataset / signal / f"{source_name}.npz") as expected:
                    maximum = max(maximum, float(np.max(np.abs(got["scores"] - expected["scores"]))))
        reproduction[name] = maximum
        if maximum != 0:
            errors.append(f"baseline reproduction {name}: {maximum}")
    checks.update({
        "point_scores": len(score_paths), "metric_rows": len(metrics),
        "valid_signals_per_variant": valid.to_dict(),
        "baseline_reproduction_max_abs_errors": reproduction,
        "inference_records": len(records),
    })
    result = {
        "status": "passed" if not errors else "failed", "errors": errors,
        "checks": checks,
        "labels_used_for_training_calibration_or_selection": False,
    }
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

