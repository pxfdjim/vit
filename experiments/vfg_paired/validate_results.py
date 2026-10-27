#!/usr/bin/env python3
"""Validate completeness, finiteness, cache integrity, and Stable reproduction."""

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

from experiments.vfg_paired.runner import load_json, publish_json_no_replace


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    metric_paths = sorted((args.output_root / "metrics").glob("*/*/*.json"))
    score_paths = sorted((args.output_root / "point_scores").glob("*/*/*.npz"))
    expected = 125 * len(config["variants_order"])
    if len(metric_paths) != expected or len(score_paths) != expected:
        raise RuntimeError(f"expected {expected} metrics/scores, found {len(metric_paths)}/{len(score_paths)}")

    alpha_set = {float(value) for value in config["evaluation"]["alphas"]}
    keys = set()
    maximum_stable_difference = 0.0
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
        if args.previous_root and key[2] == "A_stable_median":
            previous = args.previous_root / "point_scores" / key[0] / key[1] / "A_stable_vit4ts.npz"
            with np.load(previous, allow_pickle=False) as stored:
                old_scores = stored["scores"].astype(np.float64)
            maximum_stable_difference = max(maximum_stable_difference, float(np.max(np.abs(scores - old_scores))))

    results = pd.read_csv(args.output_root / "results.csv")
    if len(results) != expected * len(alpha_set) or not np.isfinite(results[["alpha", "precision", "recall", "f1", "threshold"]].to_numpy()).all():
        raise RuntimeError("results.csv completeness/finiteness failed")
    completion = pd.read_csv(args.output_root / "completion.csv")
    if not completion["complete"].all():
        raise RuntimeError("completion.csv contains an incomplete group")
    candidates = pd.read_csv(args.output_root / "candidate_pool_audit.csv")
    if len(candidates) != 125 or candidates.max_common_visual_pool.max() > 32 or candidates.min_common_visual_pool.min() < 5:
        raise RuntimeError("candidate-pool audit failed")

    payload = {
        "status": "passed",
        "metric_files": len(metric_paths),
        "point_score_files": len(score_paths),
        "result_rows": len(results),
        "complete_model_signal_pairs": len(keys),
        "alphas": sorted(alpha_set, reverse=True),
        "all_point_scores_finite_and_aligned": True,
        "all_metric_values_finite": True,
        "all_125_candidate_pools_valid": True,
        "maximum_absolute_A_score_difference_vs_previous_run": maximum_stable_difference if args.previous_root else None,
        "stable_reproduction_tolerance": 1e-6 if args.previous_root else None,
        "stable_reproduction_passed": maximum_stable_difference <= 1e-6 if args.previous_root else None,
    }
    if not publish_json_no_replace(args.output_root / "validation.json", payload):
        raise FileExistsError(args.output_root / "validation.json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=HERE / "config.json")
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--previous-root", type=Path)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
