#!/usr/bin/env python3
"""Train or materialize one dataset-specific AMR ablation checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.amr_training_ablation import (
    AblationReferenceModel, AlignmentLossConfig, alignment_loss, gradient_norms,
    masked_numeric_batch, masked_reconstruction_loss, trainable_parameter_counts,
)


def gradient_audit(model, raw, visual, objective, loss_config, mask_fraction):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(raw.device) if raw.is_cuda else None
    model.train()
    rows = []
    if objective == "numeric_reconstruction":
        masked, target, mask = masked_numeric_batch(raw, mask_fraction)
        loss = masked_reconstruction_loss(model(masked), target, mask)
        model.zero_grad(set_to_none=True); loss.backward()
        rows.append({"loss_group": "numeric_reconstruction", "loss": float(loss.detach()),
                     **gradient_norms(model)})
    elif loss_config.active:
        outputs = model(raw, visual)
        _, groups, _ = alignment_loss(outputs, loss_config)
        active = {
            "patch": loss_config.patch, "temporal_relation": loss_config.temporal_relation,
            "transition_geometry": loss_config.transition_geometry,
            "reference": loss_config.reference,
        }
        selected = [(name, value) for name, value in groups.items() if active[name]]
        for index, (name, loss) in enumerate(selected):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=index + 1 < len(selected))
            rows.append({"loss_group": name, "loss": float(loss.detach()),
                         **gradient_norms(model)})
    else:
        rows.append({"loss_group": "none", "loss": 0.0,
                     **{name: 0.0 for name in ("numeric_transformer", "cross_attention",
                                               "fusion", "reconstruction_head", "other", "all")}})
    model.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, raw.device)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    spec = config["training_variants"][args.variant]
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cached.get("labels_read") is not False or cached.get("uses_all_windows") is not True:
        raise ValueError("training cache violates unlabeled/all-window protocol")
    architecture = spec["architecture"]
    model = AblationReferenceModel(architecture=architecture).to(args.device)
    loss_config = AlignmentLossConfig(**spec.get("alignment_losses", {
        "patch": False, "temporal_relation": False,
        "transition_geometry": False, "reference": False}))
    objective = spec["objective"]
    if objective == "cross_modal_alignment" and not loss_config.active:
        objective = "no_active_objective"
    if objective not in {"cross_modal_alignment", "numeric_reconstruction",
                          "no_active_objective", "untrained_random"}:
        raise ValueError(f"unsupported objective: {objective}")

    dataset = TensorDataset(cached["raw"], cached["visual"].float())
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]),
                        shuffle=True, generator=generator, drop_last=True, num_workers=0)
    if len(loader) == 0:
        raise ValueError("dataset has fewer windows than one batch")
    first_raw, first_visual = next(iter(DataLoader(dataset,
        batch_size=int(config["training"]["batch_size"]), shuffle=False, drop_last=True)))
    first_raw, first_visual = first_raw.to(args.device), first_visual.to(args.device)
    audit = gradient_audit(model, first_raw, first_visual, objective, loss_config,
                           float(config["numeric_self_supervision"]["mask_fraction"]))

    optimizer = None
    if objective in {"cross_modal_alignment", "numeric_reconstruction"}:
        optimizer = torch.optim.AdamW(model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]))
    history, peak, updates = [], 0, 0
    started = time.perf_counter()
    epochs = int(config["training"]["epochs"])
    # Random and no-objective controls are materialized without pretending that
    # data passes are optimization.  Their requested budget is recorded below.
    if optimizer is not None:
        for epoch in range(epochs):
            model.train(); sums: dict[str, float] = {}; batches = 0
            for raw, visual in loader:
                raw, visual = raw.to(args.device), visual.to(args.device)
                augmented = (raw * (0.9 + 0.2 * torch.rand(raw.shape[0], 1, device=args.device))
                             + 0.01 * torch.randn_like(raw))
                if objective == "numeric_reconstruction":
                    masked, target, mask = masked_numeric_batch(
                        augmented, float(config["numeric_self_supervision"]["mask_fraction"]))
                    output = model(masked)
                    loss = masked_reconstruction_loss(output, target, mask)
                    parts = {"numeric_reconstruction": loss}
                else:
                    output = model(augmented, visual)
                    loss, _, diagnostics = alignment_loss(output, loss_config)
                    assert loss is not None
                    parts = diagnostics
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norms = gradient_norms(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                updates += 1
                for name, value in {"total": loss, **parts}.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach())
                for name, value in norms.items():
                    sums[f"grad_norm_{name}"] = sums.get(f"grad_norm_{name}", 0.0) + value
                batches += 1
            row = {"epoch": epoch + 1, **{name: value / batches for name, value in sums.items()}}
            history.append(row); print(json.dumps(row), flush=True)
            if torch.cuda.is_available():
                peak = max(peak, torch.cuda.max_memory_allocated(args.device))
    elapsed = time.perf_counter() - started
    payload = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": {"hidden": 192, "heads": 4, "layers": 2,
                         "architecture": architecture},
        "variant": args.variant, "dataset": cached["dataset"], "objective": objective,
        "alignment_losses": loss_config.__dict__, "history": history,
        "gradient_path_audit": audit, "parameter_counts": trainable_parameter_counts(model),
        "train_windows": len(dataset), "signals": cached["signals"],
        "elapsed_seconds": elapsed, "peak_cuda_memory_bytes": peak,
        "optimizer_updates": updates, "requested_epochs": epochs,
        "completed_epochs": len(history), "checkpoint_selection": "fixed_final_epoch",
        "labels_used": False, "validation_split_used": False,
        "source_cache": str(args.cache), "seed": seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "elapsed_seconds": elapsed,
                      "optimizer_updates": updates, "objective": objective}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
