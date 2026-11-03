#!/usr/bin/env python3
"""Collect training, gradient, parameter, inference, and similarity artifacts."""
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

from src.models.amr_training_ablation import AblationReferenceModel, trainable_parameter_counts


def checkpoint_path(config, output_root, variant, dataset):
    if variant in {"Full", "C1_full"}:
        return Path(config["full_checkpoint_root"]) / f"{dataset}.pt"
    if variant == "C4_numeric_only":
        variant = "T2_numeric_ssl"
    return output_root / "checkpoints" / variant / f"{dataset}.pt"


def payload_for(config, output_root, variant, dataset):
    return torch.load(checkpoint_path(config, output_root, variant, dataset),
                      map_location="cpu", weights_only=True)


def numeric_state(payload):
    return {name.removeprefix("numeric."): value for name, value in payload["state_dict"].items()
            if name.startswith("numeric.")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("initial", "all"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    variants = config["initial_variants"] if args.stage == "initial" else config["all_variants"]
    learned = [name for name in variants if name not in {"T3", "E2", "B3"}]
    train_rows, parameter_rows, gradient_rows, equality_rows = [], [], [], []
    canonical_seen = set()
    for dataset in config["datasets"]:
        for variant in learned:
            canonical = "Full" if variant == "C1_full" else (
                "T2_numeric_ssl" if variant == "C4_numeric_only" else variant)
            payload = payload_for(config, args.output_root, variant, dataset)
            if canonical not in {"Full"}:
                objective = payload["objective"]
                history = payload["history"]
                final_loss = history[-1]["total"] if history else np.nan
                row = {"dataset": dataset, "variant": variant, "canonical_checkpoint": canonical,
                       "objective": objective, "architecture": payload["model_config"]["architecture"],
                       "train_windows": payload["train_windows"], "signals": payload["signals"],
                       "requested_epochs": payload["requested_epochs"],
                       "completed_epochs": payload["completed_epochs"],
                       "optimizer_updates": payload["optimizer_updates"],
                       "final_loss": final_loss, "training_seconds": payload["elapsed_seconds"],
                       "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
                       "labels_used": payload["labels_used"]}
                for group in ("numeric_transformer", "cross_attention", "fusion",
                              "reconstruction_head", "all"):
                    values = [epoch.get(f"grad_norm_{group}", np.nan) for epoch in history]
                    values = [value for value in values if np.isfinite(value)]
                    row[f"mean_epoch_grad_norm_{group}"] = float(np.mean(values)) if values else 0.0
                train_rows.append(row)
                counts = payload["parameter_counts"]
                parameter_rows.append({"dataset": dataset, "variant": variant,
                                       "canonical_checkpoint": canonical, **counts})
                for audit in payload["gradient_path_audit"]:
                    gradient_rows.append({"dataset": dataset, "variant": variant,
                                          "audit_state": "initialization", **audit})
            else:
                history = payload["history"]
                train_rows.append({
                    "dataset": dataset, "variant": variant, "canonical_checkpoint": "Full",
                    "objective": "cross_modal_alignment", "architecture": "cross",
                    "train_windows": payload["train_windows"], "signals": payload["signals"],
                    "requested_epochs": 16, "completed_epochs": len(history),
                    "optimizer_updates": np.nan, "final_loss": history[-1]["total"],
                    "training_seconds": payload["elapsed_seconds"],
                    "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
                    "labels_used": payload["labels_used"],
                    **{f"mean_epoch_grad_norm_{group}": np.nan for group in
                       ("numeric_transformer", "cross_attention", "fusion",
                        "reconstruction_head", "all")},
                })
                model = AblationReferenceModel("cross")
                model.load_state_dict(payload["state_dict"], strict=True)
                parameter_rows.append({"dataset": dataset, "variant": variant,
                                       "canonical_checkpoint": "Full",
                                       **trainable_parameter_counts(model)})
        if args.stage == "all":
            t1 = numeric_state(payload_for(config, args.output_root, "T1_random", dataset))
            l5 = numeric_state(payload_for(config, args.output_root, "L5_no_alignment", dataset))
            keys_equal = set(t1) == set(l5)
            maximum = max(float((t1[name] - l5[name]).abs().max()) for name in t1) if keys_equal else np.inf
            equality_rows.append({"dataset": dataset, "comparison": "T1_random_vs_L5_no_alignment",
                                  "state_keys_equal": keys_equal, "numeric_state_max_abs_diff": maximum})

    inference_rows, similarity_rows = [], []
    manifest_paths = sorted((args.output_root / "inference_manifests").glob(f"*_{args.stage}.json"))
    for path in manifest_paths:
        manifest = json.loads(path.read_text())
        if manifest["status"] != "passed":
            raise RuntimeError(f"failed inference manifest: {path}")
        for record in manifest["records"]:
            if record["status"] != "passed":
                continue
            for variant, seconds in record["numeric_encoding_seconds"].items():
                inference_rows.append({"dataset": record["dataset"], "signal": record["signal"],
                                       "variant": variant, "windows": record["windows"],
                                       "numeric_encoding_seconds": seconds,
                                       "total_signal_seconds": record["elapsed_seconds"]})
        similarity_rows.extend(manifest["feature_similarity"])
    train = pd.DataFrame(train_rows); parameters = pd.DataFrame(parameter_rows)
    gradients = pd.DataFrame(gradient_rows); equality = pd.DataFrame(equality_rows)
    inference = pd.DataFrame(inference_rows); similarity = pd.DataFrame(similarity_rows)
    train.to_csv(args.output_root / "training_summary.csv", index=False)
    parameters.to_csv(args.output_root / "parameter_counts.csv", index=False)
    gradients.to_csv(args.output_root / "gradient_path_audit.csv", index=False)
    if len(equality): equality.to_csv(args.output_root / "checkpoint_equality_audit.csv", index=False)
    inference.to_csv(args.output_root / "inference_runtime_per_signal.csv", index=False)
    similarity.to_csv(args.output_root / "feature_similarity_per_signal.csv", index=False)
    train.groupby("variant", as_index=False).agg(
        datasets=("dataset", "count"), total_training_seconds=("training_seconds", "sum"),
        mean_training_seconds=("training_seconds", "mean"),
        mean_final_loss=("final_loss", "mean"), optimizer_updates=("optimizer_updates", "sum"),
    ).to_csv(args.output_root / "training_summary_by_variant.csv", index=False)
    if len(inference):
        inference.groupby("variant", as_index=False).agg(
            signals=("signal", "count"), total_numeric_encoding_seconds=("numeric_encoding_seconds", "sum"),
            mean_numeric_encoding_seconds=("numeric_encoding_seconds", "mean"),
        ).to_csv(args.output_root / "inference_runtime_by_variant.csv", index=False)
    if len(similarity):
        similarity.groupby("variant", as_index=False).agg(
            signals=("signal", "count"), numeric_cka_with_full=("numeric_cka_with_full", "mean"),
            numeric_visual_cosine=("numeric_visual_cosine", "mean"),
            numeric_fused_cosine=("numeric_fused_cosine", "mean"),
            fused_visual_cosine=("fused_visual_cosine", "mean"),
        ).to_csv(args.output_root / "feature_similarity_by_variant.csv", index=False)
    print(json.dumps({"training_rows": len(train), "gradient_rows": len(gradients),
                      "inference_rows": len(inference), "similarity_rows": len(similarity)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
