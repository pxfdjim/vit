#!/usr/bin/env python3
"""Train one APRA+TREM variant with the validated APRA-only objectives."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.apra_acef import LossWeights, detection_loss, path_loss, reversible_warp
from src.models.apra_trem import (
    APRATREM, b3_token_evidence, gradient_norms, load_apra_only,
)


@torch.no_grad()
def projected_visual(model, visual, device, batch_size=256):
    return torch.cat([
        model.visual(visual[start:start + batch_size].float().to(device)).half()
        for start in range(0, len(visual), batch_size)
    ])


def reference_raw(raw: torch.Tensor, ids: torch.Tensor,
                  columns: torch.Tensor) -> torch.Tensor:
    patches = raw[ids].reshape(len(ids), 14, 16)
    return patches.gather(1, columns[..., None].expand(-1, -1, 16))


def max_change(module: torch.nn.Module, initial: dict[str, torch.Tensor]) -> float:
    return max(float((value.detach().cpu() - initial[name]).abs().max())
               for name, value in module.state_dict().items())


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
    spec = config["variants"][args.variant]
    seed = int(config["primary_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cached["labels_read"] or cached["seed"] != seed:
        raise RuntimeError("cache integrity")
    model = APRATREM(
        use_local=spec["use_local"], use_multiscale=spec["use_multiscale"],
        trem_width=int(config["trem"]["width"]),
        residual_limit=float(config["trem"]["residual_limit"]),
    ).to(device)
    apra_checkpoint = Path(config["apra_run"]) / "checkpoints/APRA_only" / f"{cached['dataset']}.pt"
    load_apra_only(model, str(apra_checkpoint))
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)
    initial_numeric = {name: value.detach().cpu().clone() for name, value in model.numeric.state_dict().items()}
    initial_aligner = {name: value.detach().cpu().clone() for name, value in model.aligner.state_dict().items()}
    initial_apra_head = {name: value.detach().cpu().clone() for name, value in model.apra_head.state_dict().items()}
    raw = cached["raw"].to(device)
    synthetic_raw = cached["synthetic_raw"].to(device)
    altered = cached["patch_mask"].to(device)
    reference_ids = cached["reference_ids"].to(device)
    reference_columns = cached["reference_columns"].to(device)
    visual = projected_visual(model, cached["visual"], device)
    synthetic_visual = projected_visual(model, cached["synthetic_visual"], device)
    loss_config = config["loss_weights"]
    weights = LossWeights(path=loss_config["path"], explain=0.0,
                          detect=loss_config["detect"],
                          detect_margin=loss_config["detect_margin"],
                          gamma=loss_config["gamma"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        torch.arange(len(raw)), batch_size=config["training"]["batch_size"],
        shuffle=True, generator=generator, drop_last=True, num_workers=0,
    )

    def batch_outputs(indices):
        ids = reference_ids[indices]
        columns = reference_columns[indices]
        expanded = columns[..., None]
        nq = model.numeric(raw[indices])
        ns = model.numeric(synthetic_raw[indices])
        nr_full = model.numeric(raw[ids])
        nr = nr_full.gather(1, expanded.expand(-1, -1, nr_full.shape[-1]))
        vq = visual[indices].float()
        vs = synthetic_visual[indices].float()
        vr_full = visual[ids].float()
        vr = vr_full.gather(1, expanded.expand(-1, -1, vr_full.shape[-1]))
        rq = raw[indices].reshape(len(indices), 14, 16)
        rs = synthetic_raw[indices].reshape(len(indices), 14, 16)
        rr = reference_raw(raw, ids, columns)
        clean_base = b3_token_evidence(nq, vq, nr, vr, rq, rr)
        synthetic_base = b3_token_evidence(ns, vs, nr, vr, rs, rr)
        clean = model.forward_tokens(nq, vq, nr, vr, clean_base)
        synthetic = model.forward_tokens(ns, vs, nr, vr, synthetic_base)
        warped_n, target = reversible_warp(nq)
        warped_v = target @ vq
        warped_raw = target @ rq
        warped_base = b3_token_evidence(warped_n, warped_v, nr, vr, warped_raw, rr)
        warped = model.forward_tokens(warped_n, warped_v, nr, vr, warped_base)
        return clean, synthetic, warped, target, clean_base, synthetic_base

    def losses(indices):
        clean, synthetic, warped, target, clean_base, synthetic_base = batch_outputs(indices)
        parts = {
            "path": path_loss(warped, target),
            "detect": detection_loss(
                clean, synthetic, warped, clean_base, synthetic_base,
                altered[indices], weights,
            ),
        }
        return weights.path * parts["path"] + weights.detect * parts["detect"], parts

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    first = next(iter(DataLoader(
        torch.arange(len(raw)), batch_size=config["training"]["batch_size"],
        shuffle=False, drop_last=True,
    ))).to(device)
    model.eval()
    with torch.no_grad():
        initial_output, _, _, _, _, _ = batch_outputs(first)
        zero_equivalence = float((initial_output["delta"] - initial_output["apra_delta"]).abs().max())
        zero_initialized = max(float(p.abs().max()) for p in model.trem.output.parameters()) == 0.0
    if zero_equivalence != 0 or not zero_initialized:
        raise RuntimeError("TREM must preserve APRA exactly before training")

    # Probe one update so gradients behind the zero output layer become observable.
    saved = copy.deepcopy(model.state_dict())
    model.train(); probe, _ = losses(first)
    optimizer.zero_grad(set_to_none=True); probe.backward(); optimizer.step()
    _, probe_parts = losses(first)
    model.zero_grad(set_to_none=True); probe_parts["detect"].backward()
    detect_after_one_step = gradient_norms(model)
    model.load_state_dict(saved)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    individual = {}
    for name in ("path", "detect"):
        model.zero_grad(set_to_none=True); _, parts = losses(first); parts[name].backward()
        individual[name] = gradient_norms(model)
    model.zero_grad(set_to_none=True)
    if individual["path"]["apra"] <= 0:
        raise RuntimeError("L_path does not update APRA")
    if detect_after_one_step["apra"] <= 0 or detect_after_one_step["trem"] <= 0:
        raise RuntimeError("L_detect does not update APRA and TREM")
    if any(individual[name]["visual"] != 0 for name in individual):
        raise RuntimeError("frozen visual projection received gradient")

    history = []
    updates = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(config["training"]["epochs"]):
        model.train(); sums = {}; batches = 0
        for indices in loader:
            indices = indices.to(device)
            total, parts = losses(indices)
            optimizer.zero_grad(set_to_none=True); total.backward()
            norms = gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["training"]["gradient_clip"],
            )
            optimizer.step(); updates += 1; batches += 1
            for name, value in {"total": total, **parts}.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            for name, value in norms.items():
                sums[f"grad_{name}"] = sums.get(f"grad_{name}", 0.0) + value
        row = {"epoch": epoch + 1, **{key: value / batches for key, value in sums.items()}}
        history.append(row); print(json.dumps(row), flush=True)
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.inference_mode():
        clean, _, _, _, _, _ = batch_outputs(first)
        altered_alignment = {key: value for key, value in clean["alignment"].items()}
        altered_alignment["structural_numeric"] = altered_alignment["structural_numeric"] + .1
        changed = model.trem(altered_alignment, clean["apra_delta"], clean["b3_evidence"])["trem_delta"]
        residual_input_effect = float((changed - clean["trem_delta"]).abs().mean())
        reliability_alignment = {key: value for key, value in clean["alignment"].items()}
        reliability_alignment["reliability"] = 1.0 - reliability_alignment["reliability"]
        changed = model.trem(reliability_alignment, clean["apra_delta"], clean["b3_evidence"])["trem_delta"]
        reliability_effect = float((changed - clean["trem_delta"]).abs().mean())
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": {
            "use_local": spec["use_local"], "use_multiscale": spec["use_multiscale"],
            "hidden": 192, "heads": 4, "layers": 2,
            "trem_width": config["trem"]["width"],
            "residual_limit": config["trem"]["residual_limit"],
        },
        "dataset": cached["dataset"], "variant": args.variant, "seed": seed,
        "history": history, "gradient_audit_by_loss": individual,
        "detect_gradient_after_one_step": detect_after_one_step,
        "zero_initialized_trem": zero_initialized,
        "initial_equivalence_max_error": zero_equivalence,
        "residual_input_effect": residual_input_effect,
        "reliability_input_effect": reliability_effect,
        "numeric_max_abs_change": max_change(model.numeric, initial_numeric),
        "apra_max_abs_change": max(max_change(model.aligner, initial_aligner),
                                   max_change(model.apra_head, initial_apra_head)),
        "parameter_counts": {
            "all": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "numeric": sum(p.numel() for p in model.numeric.parameters()),
            "apra": sum(p.numel() for p in model.aligner.parameters()) + sum(p.numel() for p in model.apra_head.parameters()),
            "trem": sum(p.numel() for p in model.trem.parameters()),
        },
        "elapsed_seconds": elapsed, "optimizer_updates": updates,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "labels_used": False, "validation_split_used": False,
        "checkpoint_selection": "fixed_final_epoch",
        "source_cache": str(args.cache), "apra_initialization": str(apra_checkpoint),
        "active_losses": ["path", "detect"], "loss_weights": config["loss_weights"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "elapsed_seconds": elapsed,
                      "updates": updates}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

