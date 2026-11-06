#!/usr/bin/env python3
"""Validate the streamlined Detect-only APRA against immutable E6 history."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.apra import APRA, LossWeights, detection_loss, load_apra_checkpoint, reversible_warp


DATASETS = ("realTraffic", "realTweets", "SMAP", "MSL", "artificialWithAnomaly",
            "realAdExchange", "realAWSCloudwatch", "YAHOOA1", "YAHOOA2",
            "YAHOOA3", "YAHOOA4")
SEEDS = (20261104, 20261105, 20261106)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_historical_module(path: Path):
    name = "_historical_e1_apra_for_detect_only_validation"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import historical module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def flatten_outputs(output: dict, prefix: str = "") -> dict[str, torch.Tensor]:
    flattened = {}
    for key, value in output.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_outputs(value, name))
        elif torch.is_tensor(value):
            flattened[name] = value
    return flattened


def module_equivalence(checkpoint: Path, historical_module_path: Path) -> dict:
    historical_module = load_historical_module(historical_module_path)
    current, payload = load_apra_checkpoint(checkpoint)
    legacy = payload["model_config"]
    historical = historical_module.APRA(
        hidden=int(legacy.get("hidden", 192)), heads=int(legacy.get("heads", 4)),
        layers=int(legacy.get("layers", 2)))
    historical.load_state_dict(payload["state_dict"], strict=True)
    current.eval()
    historical.eval()

    generator = torch.Generator().manual_seed(61699)
    tokens = [torch.randn(3, 14, 192, generator=generator) for _ in range(4)]
    current_output = flatten_outputs(current.forward_tokens(*tokens))
    historical_output = flatten_outputs(historical.forward_tokens(*tokens))
    if current_output.keys() != historical_output.keys():
        raise RuntimeError("module output keys changed")
    output_error = max(float((current_output[key] - historical_output[key]).abs().max())
                       for key in current_output)

    altered = torch.zeros(3, 14, dtype=torch.bool)
    altered[:, 4:9] = True
    synthetic_tokens = [tokens[0] + .2, tokens[1] - .15, tokens[2], tokens[3]]
    warped_numeric, target = reversible_warp(tokens[0], torch.tensor([1.0, -.7, .4]))
    warped_tokens = [warped_numeric, target @ tokens[1], tokens[2], tokens[3]]
    bases = (torch.linspace(0, 1, 42).reshape(3, 14),
             torch.linspace(.1, 1.1, 42).reshape(3, 14))

    def loss_and_gradients(model, module):
        model.zero_grad(set_to_none=True)
        clean = model.forward_tokens(*tokens)
        synthetic = model.forward_tokens(*synthetic_tokens)
        warped = model.forward_tokens(*warped_tokens)
        weights = module.LossWeights()
        loss = module.detection_loss(clean, synthetic, warped, *bases, altered, weights)
        loss.backward()
        gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                     for name, parameter in model.named_parameters()}
        return loss.detach(), gradients

    current_loss, current_gradients = loss_and_gradients(current, sys.modules[APRA.__module__])
    historical_loss, historical_gradients = loss_and_gradients(historical, historical_module)
    loss_error = float((current_loss - historical_loss).abs())
    gradient_error = 0.0
    if current_gradients.keys() != historical_gradients.keys():
        raise RuntimeError("gradient parameter keys changed")
    for name in current_gradients:
        left, right = current_gradients[name], historical_gradients[name]
        if (left is None) != (right is None):
            raise RuntimeError(f"gradient reachability changed: {name}")
        if left is not None:
            gradient_error = max(gradient_error, float((left - right).abs().max()))
    return {"output_max_abs_error": output_error, "loss_max_abs_error": loss_error,
            "gradient_max_abs_error": gradient_error,
            "output_tensors": len(current_output),
            "gradient_tensors": sum(value is not None for value in current_gradients.values())}


def copy_reference_tables(source: Path, output: Path) -> None:
    selected_variants = {"B3", *(f"E1_existing_s{seed}" for seed in SEEDS),
                         *(f"E6_detect_only_s{seed}" for seed in SEEDS)}
    for filename in ("metrics_by_scope.csv", "metrics_by_dataset.csv", "event_counts_all.csv"):
        frame = pd.read_csv(source / filename)
        frame[frame.variant.isin(selected_variants)].to_csv(output / filename, index=False)
    for filename in ("multiseed_summary.csv", "multiseed_by_dataset.csv"):
        frame = pd.read_csv(source / filename)
        frame[frame.variant.isin({"E1_existing", "E6_detect_only"})].to_csv(
            output / filename, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-module", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    historical_root = Path(config["historical_e6_root"])
    checkpoint_root = historical_root / "checkpoints/E1_no_path"
    score_root = historical_root / "point_scores"
    mechanism_root = Path(config["mechanism_result_root"])

    checkpoint_rows = []
    checkpoint_error = 0.0
    parameter_counts = set()
    state_key_counts = set()
    for seed in SEEDS:
        for dataset in DATASETS:
            path = checkpoint_root / f"seed_{seed}" / f"{dataset}.pt"
            model, payload = load_apra_checkpoint(path)
            state = model.state_dict()
            original = payload["state_dict"]
            error = max(float((state[key] - original[key]).abs().max()) for key in state)
            checkpoint_error = max(checkpoint_error, error)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            parameter_counts.add(parameter_count)
            state_key_counts.add(len(state))
            if payload["active_losses"] != ["detect"] or payload["labels_used"]:
                raise RuntimeError(f"historical E6 integrity failure: {path}")
            checkpoint_rows.append({"seed": seed, "dataset": dataset, "source_path": str(path),
                "file_sha256": sha256(path), "state_sha256": state_digest(state),
                "strict_load_max_abs_error": error, "state_tensors": len(state),
                "parameters": parameter_count, "optimizer_updates": payload["optimizer_updates"],
                "training_seconds": payload["elapsed_seconds"],
                "active_losses": ",".join(payload["active_losses"])})
    pd.DataFrame(checkpoint_rows).to_csv(args.output / "checkpoint_manifest.csv", index=False)

    score_rows = []
    score_error = 0.0
    timestamp_errors = 0
    total_values = 0
    source_files = sorted(score_root.glob("*/*/E1_no_path_s*.npz"))
    for source in source_files:
        dataset, signal = source.parents[1].name, source.parent.name
        seed = int(source.stem.rsplit("s", 1)[1])
        reused = mechanism_root / "point_scores" / dataset / signal / f"E6_detect_only_s{seed}.npz"
        with np.load(source) as left, np.load(reused) as right:
            error = float(np.max(np.abs(left["scores"] - right["scores"])))
            timestamps_equal = bool(np.array_equal(left["timestamps"], right["timestamps"]))
            points = len(left["scores"])
        score_error = max(score_error, error)
        timestamp_errors += int(not timestamps_equal)
        total_values += points
        score_rows.append({"seed": seed, "dataset": dataset, "signal": signal,
            "source_path": str(source), "validated_reuse_path": str(reused),
            "points": points, "score_max_abs_error": error,
            "timestamps_equal": timestamps_equal})
    pd.DataFrame(score_rows).to_csv(args.output / "point_score_manifest.csv", index=False)
    copy_reference_tables(mechanism_root, args.output)

    module = module_equivalence(
        checkpoint_root / f"seed_{SEEDS[0]}" / "MSL.pt", args.historical_module)
    source_validation = json.loads((mechanism_root / "validation.json").read_text())
    checks = {
        "checkpoint_strict_load_max_abs_error": checkpoint_error,
        "parameter_counts": sorted(parameter_counts),
        "state_dict_tensor_counts": sorted(state_key_counts),
        "module_output_max_abs_error": module["output_max_abs_error"],
        "detect_loss_max_abs_error": module["loss_max_abs_error"],
        "detect_gradient_max_abs_error": module["gradient_max_abs_error"],
        "point_score_max_abs_error": score_error,
        "timestamp_mismatches": timestamp_errors,
        "source_mechanism_reuse_max_abs_error": source_validation["checks"][
            "reproduction_max_abs_errors"]["reused_E6"],
    }
    passed = (all(value == 0 for key, value in checks.items()
                  if key.endswith("error") or key.endswith("errors") or key == "timestamp_mismatches")
              and checks["parameter_counts"] == [2059973]
              and checks["state_dict_tensor_counts"] == [73]
              and len(checkpoint_rows) == 33 and len(score_rows) == 1476)
    validation = {"status": "passed" if passed else "failed", "checks": checks,
        "module_equivalence": module, "checkpoints": len(checkpoint_rows),
        "point_score_files": len(score_rows), "signals": len({(row['dataset'], row['signal']) for row in score_rows}),
        "seeds": list(SEEDS), "point_score_values": total_values,
        "historical_e6_root": str(historical_root),
        "historical_e1_preserved_at": config["e1_result_root"],
        "labels_used_for_training_calibration_or_selection": False,
        "new_training_performed": False}
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2))
    (args.output / "reused_artifacts.json").write_text(json.dumps({
        "checkpoints": str(checkpoint_root), "point_scores": str(score_root),
        "validated_mechanism_results": str(mechanism_root)}, indent=2))
    print(json.dumps(validation, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
