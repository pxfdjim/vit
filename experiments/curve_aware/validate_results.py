#!/usr/bin/env python3
"""Validate completeness, finiteness, cache integrity, and baseline reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.curve_aware.runner import load_json, publish_json_no_replace


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    previous_roots = list(args.previous_root)
    metric_paths = sorted((args.output_root / "metrics").glob("*/*/*.json"))
    score_paths = sorted((args.output_root / "point_scores").glob("*/*/*.npz"))
    expected_signals = sum(
        int(config["expected_signal_counts"][dataset]) for dataset in config["datasets"]
    )
    expected = expected_signals * len(config["variants_order"])
    if len(metric_paths) != expected or len(score_paths) != expected:
        raise RuntimeError(f"expected {expected} metrics/scores, found {len(metric_paths)}/{len(score_paths)}")

    alpha_set = {float(value) for value in config["evaluation"]["alphas"]}
    keys = set()
    maximum_stable_difference = 0.0
    compared_baselines = 0
    for metric_path in metric_paths:
        document = json.loads(metric_path.read_text(encoding="utf-8"))
        key = (document["dataset"], document["signal"], document["variant"])
        if key in keys or document.get("status") != "success":
            raise RuntimeError(f"duplicate or failed metric: {metric_path}")
        keys.add(key)
        if {float(row["alpha"]) for row in document["alpha_metrics"]} != alpha_set:
            raise RuntimeError(f"alpha mismatch: {metric_path}")
        numeric = [row[field] for row in document["alpha_metrics"] for field in ("threshold", "precision", "recall", "f1")]
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise RuntimeError(f"non-finite metric: {metric_path}")
        score_path = args.output_root / "point_scores" / key[0] / key[1] / f"{key[2]}.npz"
        with np.load(score_path, allow_pickle=False) as stored:
            scores = stored["scores"].astype(np.float64)
            timestamps = stored["timestamps"]
        if len(scores) != int(document["n_points"]) or len(timestamps) != len(scores) or not np.isfinite(scores).all():
            raise RuntimeError(f"invalid point scores: {score_path}")
        if previous_roots and key[2] == "A_stable_final":
            candidates = [
                root / "point_scores" / key[0] / key[1] / filename
                for root in previous_roots
                for filename in ("A_stable_median.npz", "A_stable_vit4ts.npz", "A_stable_final.npz")
            ]
            previous = next((path for path in candidates if path.is_file()), None)
            if previous is None:
                raise FileNotFoundError(f"missing previous A score for {key[0]}/{key[1]}")
            with np.load(previous, allow_pickle=False) as stored:
                old_scores = stored["scores"].astype(np.float64)
            if len(old_scores) != len(scores):
                raise RuntimeError(f"previous A length mismatch: {previous}")
            maximum_stable_difference = max(maximum_stable_difference, float(np.max(np.abs(scores - old_scores))))
            compared_baselines += 1

    results = pd.read_csv(args.output_root / "results.csv")
    if len(results) != expected * len(alpha_set) or not np.isfinite(results[["alpha", "precision", "recall", "f1", "threshold"]].to_numpy()).all():
        raise RuntimeError("results.csv completeness/finiteness failed")
    completion = pd.read_csv(args.output_root / "completion.csv")
    if not completion["complete"].all():
        raise RuntimeError("completion.csv contains an incomplete group")
    candidates = pd.read_csv(args.output_root / "candidate_pool_audit.csv", keep_default_na=False)
    expected_paired_rows = expected_signals * sum(
        config["variants"][name]["family"] != "A" for name in config["variants_order"]
    )
    if len(candidates) != expected_paired_rows:
        raise RuntimeError(f"expected {expected_paired_rows} candidate audit rows, found {len(candidates)}")
    for column in ("min_windows", "max_windows"):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    if (
        candidates[["min_windows", "max_windows"]].isna().any().any()
        or (candidates.min_windows < 1).any()
        or (candidates.max_windows > int(config["model"]["candidate_pool"]["max_common_visual_windows"])).any()
        or (candidates.min_windows > candidates.max_windows).any()
        or (candidates.pool_digest.astype(str).str.len() == 0).any()
    ):
        raise RuntimeError("candidate-pool bounds/digests failed")
    pool_groups = candidates.groupby(["dataset", "signal"], as_index=False).agg(
        variants=("model", "nunique"), digests=("pool_digest", "nunique")
    )
    expected_paired_variants = sum(config["variants"][name]["family"] != "A" for name in config["variants_order"])
    if not ((pool_groups.variants == expected_paired_variants) & (pool_groups.digests == 1)).all():
        raise RuntimeError("B--E did not report one shared candidate-pool identity per signal")
    if previous_roots and (compared_baselines != expected_signals or maximum_stable_difference > 1e-6):
        raise RuntimeError(
            f"A reproduction failed: compared {compared_baselines}/{expected_signals}, "
            f"maximum absolute difference {maximum_stable_difference}"
        )

    payload = {
        "status": "passed",
        "metric_files": len(metric_paths),
        "point_score_files": len(score_paths),
        "result_rows": len(results),
        "complete_model_signal_pairs": len(keys),
        "alphas": sorted(alpha_set, reverse=True),
        "all_point_scores_finite_and_aligned": True,
        "all_metric_values_finite": True,
        "expected_signals": expected_signals,
        "all_candidate_pools_valid": True,
        "all_paired_variants_share_candidate_pool": True,
        "previous_roots": [str(root) for root in previous_roots],
        "maximum_absolute_A_score_difference_vs_previous_run": maximum_stable_difference if previous_roots else None,
        "baseline_signals_compared": compared_baselines if previous_roots else None,
        "stable_reproduction_tolerance": 1e-6 if previous_roots else None,
        "stable_reproduction_passed": (compared_baselines == expected_signals and maximum_stable_difference <= 1e-6) if previous_roots else None,
    }
    if not publish_json_no_replace(args.output_root / "validation.json", payload):
        raise FileExistsError(args.output_root / "validation.json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=HERE / "config.json")
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--previous-root", type=Path, action="append", default=[], help="Previous score root for A reproduction; repeat for split original/Yahoo runs")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
