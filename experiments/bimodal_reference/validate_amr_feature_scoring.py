#!/usr/bin/env python3
"""Independent completeness, source-reproduction and path audit."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metric_error(current: pd.DataFrame, source: pd.DataFrame, mapping: dict[str, str]) -> float:
    errors = []
    for current_name, source_name in mapping.items():
        a = current[current.variant == current_name].sort_values(["dataset", "signal"])
        b = source[source.variant == source_name].sort_values(["dataset", "signal"])
        if list(zip(a.dataset, a.signal)) != list(zip(b.dataset, b.signal)):
            raise RuntimeError(f"metric identity mismatch for {current_name}")
        for metric in ("aff_f1", "vus_pr", "vus_roc"):
            av, bv = a[metric].to_numpy(float), b[metric].to_numpy(float)
            finite = np.isfinite(av) & np.isfinite(bv)
            if not np.array_equal(np.isfinite(av), np.isfinite(bv)):
                raise RuntimeError(f"defined-row mismatch for {current_name}/{metric}")
            errors.append(float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.0)
    return max(errors, default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    config = json.loads(args.config.read_text())
    current = pd.read_csv(root / "metrics_per_signal.csv")
    amr_source = Path(config["amr_source_run"])
    numeric_source = Path(config["numeric_source_run"])
    fixed_source = amr_source.parent / "amr_orionbench_affiliation_fix_20261103"
    source_amr = pd.read_csv(fixed_source / "metrics_per_signal.csv")
    source_numeric = pd.read_csv(numeric_source / "metrics_per_signal.csv")
    source_error = metric_error(current, source_amr, {"T3": "T3", "AMR_native": "AMR_native"})
    e2_error = metric_error(current, source_numeric, {
        "E2_T3_raw_l005": "E2_C_combined_l005",
        "E2_T3_raw_l010": "E2_C_combined_l010",
        "E2_T3_raw_l020": "E2_C_combined_l020",
    })

    windows = reference_changes = path_windows = path_columns = mask_cells = 0
    t3_source_ref_errors = t3_source_column_errors = 0
    amr_source_ref_errors = amr_source_column_errors = 0
    for diagnostic_path in sorted(root.glob("feature_diagnostics/*/*/evidence.npz")):
        dataset, signal = diagnostic_path.parts[-3], diagnostic_path.parts[-2]
        with np.load(diagnostic_path, allow_pickle=False) as diagnostic, np.load(
            amr_source / "path_diagnostics" / dataset / signal / "T3.npz",
            allow_pickle=False) as source_t3, np.load(
            amr_source / "path_diagnostics" / dataset / signal / "AMR_native.npz",
            allow_pickle=False) as source_amr_path:
            t3_ref = diagnostic["t3_reference_window_ids"]
            amr_ref = diagnostic["amr_reference_window_ids"]
            t3_columns = diagnostic["t3_reference_columns"]
            amr_columns = diagnostic["amr_reference_columns"]
            t3_masks = diagnostic["t3_path_masks"]
            amr_masks = diagnostic["amr_path_masks"]
            windows += len(t3_ref)
            reference_changes += int((t3_ref != amr_ref).sum())
            path_windows += int(np.any(t3_masks != amr_masks, axis=(1, 2)).sum())
            path_columns += int((np.abs(t3_columns - amr_columns) > 1e-6).sum())
            mask_cells += int((t3_masks != amr_masks).sum())
            t3_source_ref_errors += int((t3_ref != source_t3["reference_window_ids"]).sum())
            t3_source_column_errors += int((np.abs(
                t3_columns - source_t3["reference_columns"]) > 1e-6).sum())
            amr_source_ref_errors += int((amr_ref != source_amr_path["reference_window_ids"]).sum())
            amr_source_column_errors += int((np.abs(
                amr_columns - source_amr_path["reference_columns"]) > 1e-6).sum())

    t3_group = ["T3", "E2_T3_raw", "B_numeric", "C_fused", "D_volatility"]
    path_rows = [{
        "variant_group": name, "signals": 492, "windows": windows,
        "reference_change_fraction_vs_T3": 0.0,
        "path_window_change_fraction_vs_T3": 0.0,
        "aligned_column_change_fraction_vs_T3": 0.0,
    } for name in t3_group]
    for name in ("AMR_native", "A_AMR_raw"):
        path_rows.append({
            "variant_group": name, "signals": 492, "windows": windows,
            "reference_change_fraction_vs_T3": reference_changes / windows,
            "path_window_change_fraction_vs_T3": path_windows / windows,
            "aligned_column_change_fraction_vs_T3": path_columns / (windows * 14),
        })
    pd.DataFrame(path_rows).to_csv(root / "reference_path_summary.csv", index=False)

    score_files = len(list(root.glob("point_scores/*/*/*.npz")))
    diagnostics = len(list(root.glob("feature_diagnostics/*/*/evidence.npz")))
    duplicate_errors = []
    for weight in ("005", "010", "020"):
        for path in root.glob(f"point_scores/*/*/B3_raw_numeric_l{weight}.npz"):
            other = path.with_name(f"C4_raw_numeric_l{weight}.npz")
            with np.load(path, allow_pickle=False) as a, np.load(other, allow_pickle=False) as b:
                duplicate_errors.append(float(np.max(np.abs(a["scores"] - b["scores"]))))
    duplicate_error = max(duplicate_errors, default=0.0)
    validation = {
        "status": "passed",
        "signals": diagnostics, "windows": windows, "variants": len(config["variants_order"]),
        "score_files": score_files, "expected_score_files": 492 * len(config["variants_order"]),
        "source_metric_max_error": source_error, "source_e2_metric_max_error": e2_error,
        "t3_source_reference_errors": t3_source_ref_errors,
        "t3_source_column_errors": t3_source_column_errors,
        "amr_source_reference_errors": amr_source_ref_errors,
        "amr_source_column_errors": amr_source_column_errors,
        "b3_c4_score_max_error": duplicate_error,
        "amr_reference_change_fraction_vs_T3": reference_changes / windows,
        "amr_path_window_change_fraction_vs_T3": path_windows / windows,
        "amr_aligned_column_change_fraction_vs_T3": path_columns / (windows * 14),
        "amr_path_mask_cell_change_fraction_vs_T3": mask_cells / (windows * 14 * 14),
    }
    passed = (score_files == validation["expected_score_files"] and diagnostics == 492
              and source_error <= 1e-12 and e2_error <= 1e-12
              and t3_source_ref_errors == 0 and t3_source_column_errors == 0
              and amr_source_ref_errors == 0 and amr_source_column_errors == 0
              and duplicate_error == 0.0)
    validation["status"] = "passed" if passed else "failed"
    (root / "final_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
