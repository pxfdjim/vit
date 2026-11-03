#!/usr/bin/env python3
"""Independent integrity checks for fixed-reference numeric-evidence outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return stored["scores"].astype(np.float64), stored["timestamps"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    score_root = args.output_root / "point_scores"
    signals = sorted(score_root.glob("*/*/T3.npz"))
    max_t3_error = max_amr_error = max_formula_error = max_timestamp_error = 0.0
    for t3_path in signals:
        relative = t3_path.relative_to(score_root)
        new_t3, new_time = load(t3_path)
        old_t3, old_time = load(args.source_root / "point_scores" / relative)
        new_amr, amr_time = load(t3_path.with_name("AMR_native.npz"))
        old_amr, old_amr_time = load(args.source_root / "point_scores" / relative.with_name("AMR_native.npz"))
        combined, combined_time = load(t3_path.with_name("E1_C_combined.npz"))
        fused, fused_time = load(t3_path.with_name("E2_C_combined_l010.npz"))
        max_t3_error = max(max_t3_error, float(np.max(np.abs(new_t3 - old_t3))))
        max_amr_error = max(max_amr_error, float(np.max(np.abs(new_amr - old_amr))))
        max_formula_error = max(max_formula_error,
                                float(np.max(np.abs(fused - (0.9 * new_t3 + 0.1 * combined)))))
        max_timestamp_error = max(max_timestamp_error, float(not (
            np.array_equal(new_time, old_time) and np.array_equal(new_time, amr_time)
            and np.array_equal(new_time, old_amr_time) and np.array_equal(new_time, combined_time)
            and np.array_equal(new_time, fused_time))))

    metrics = pd.read_csv(args.output_root / "metrics_per_signal.csv")
    valid = metrics.groupby("variant").aff_f1.apply(lambda values: int(np.isfinite(values).sum()))
    excluded = metrics.groupby("variant").excluded_reason.apply(
        lambda values: int((values == "single_class_ground_truth").sum()))
    empty = metrics[(metrics.predicted_intervals == 0) & metrics.aff_f1.notna()]
    fixed = pd.read_csv(args.output_root / "reference_path_summary.csv")
    fixed = fixed[fixed.variant != "AMR_native"]
    checks = {
        "signals": len(signals),
        "score_files": len(list(score_root.glob("*/*/*.npz"))),
        "maximum_T3_source_score_error": max_t3_error,
        "maximum_AMR_source_score_error": max_amr_error,
        "maximum_primary_fusion_formula_error": max_formula_error,
        "timestamp_mismatch_count": int(max_timestamp_error),
        "all_variants_aff_valid_488": bool(valid.eq(488).all()),
        "all_variants_exclude_same_four": bool(excluded.eq(4).all()),
        "empty_prediction_rows": len(empty),
        "all_empty_predictions_aff_f1_zero": bool(empty.aff_f1.eq(0).all()),
        "all_E_variants_fixed_reference": bool(fixed.reference_change_fraction.eq(0).all()),
        "all_E_variants_fixed_path": bool(fixed.path_change_fraction.eq(0).all()),
    }
    passed = (checks["signals"] == 492 and checks["score_files"] == 6396
              and max_t3_error == 0 and max_amr_error == 0 and max_formula_error < 1e-12
              and checks["timestamp_mismatch_count"] == 0
              and all(value for key, value in checks.items() if key.startswith("all_")))
    result = {"status": "passed" if passed else "failed", **checks}
    (args.output_root / "final_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
