#!/usr/bin/env python3
"""Train the streamlined APRA model with L_detect only."""
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.apra import (APRA, LossWeights, detection_loss, gradient_norms,
                             load_numeric_from_full, reversible_warp)


def load_frozen_visual(model: APRA, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {name.removeprefix("visual."): value for name, value in payload["state_dict"].items()
             if name.startswith("visual.")}
    model.visual.load_state_dict(state, strict=True)
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def projected_visual(model: APRA, visual: torch.Tensor, device: torch.device,
                     batch_size: int = 256) -> torch.Tensor:
    return torch.cat([model.visual(visual[start:start + batch_size].float().to(device)).half()
                      for start in range(0, len(visual), batch_size)])


def base_proxy(nq, vq, nr, vr):
    numeric = 1 - F.cosine_similarity(nq, nr, dim=-1)
    visual = 1 - F.cosine_similarity(vq, vr, dim=-1)
    return .5 * numeric.detach() + .5 * visual.detach()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cached["labels_read"] or cached["seed"] != seed:
        raise RuntimeError("cache integrity failure")

    model = APRA(**config["model_config"]).to(device)
    full_checkpoint = Path(config["full_checkpoint_root"]) / f"{cached['dataset']}.pt"
    load_numeric_from_full(model, full_checkpoint)
    visual_checkpoint = (Path(config["rcca_run"]) / "checkpoints/M1_full" /
                         f"seed_{seed}" / f"{cached['dataset']}.pt")
    load_frozen_visual(model, visual_checkpoint)
    initial_numeric = {name: value.detach().cpu().clone()
                       for name, value in model.numeric.state_dict().items()}

    raw = cached["raw"].to(device)
    synthetic_raw = cached["synthetic_raw"].to(device)
    altered = cached["patch_mask"].to(device)
    reference_ids = cached["reference_ids"].to(device)
    reference_columns = cached["reference_columns"].to(device)
    visual = projected_visual(model, cached["visual"], device)
    synthetic_visual = projected_visual(model, cached["synthetic_visual"], device)
    weights = LossWeights(**config["loss_weights"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(torch.arange(len(raw)), batch_size=config["training"]["batch_size"],
                        shuffle=True, generator=generator, drop_last=True, num_workers=0)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])

    def batch_outputs(indices):
        ids = reference_ids[indices]
        columns = reference_columns[indices][..., None]
        nq = model.numeric(raw[indices])
        ns = model.numeric(synthetic_raw[indices])
        nr_full = model.numeric(raw[ids])
        nr = nr_full.gather(1, columns.expand(-1, -1, nr_full.shape[-1]))
        vq = visual[indices].float()
        vs = synthetic_visual[indices].float()
        vr_full = visual[ids].float()
        vr = vr_full.gather(1, columns.expand(-1, -1, vr_full.shape[-1]))
        clean = model.forward_tokens(nq, vq, nr, vr)
        synthetic = model.forward_tokens(ns, vs, nr, vr)
        warped_n, target = reversible_warp(nq)
        warped = model.forward_tokens(warped_n, target @ vq, nr, vr)
        return (clean, synthetic, warped, base_proxy(nq, vq, nr, vr),
                base_proxy(ns, vs, nr, vr))

    def losses(indices):
        clean, synthetic, warped, clean_base, synthetic_base = batch_outputs(indices)
        detect = detection_loss(clean, synthetic, warped, clean_base, synthetic_base,
                                altered[indices], weights)
        return detect, {"detect": detect}

    first = next(iter(DataLoader(torch.arange(len(raw)),
                                 batch_size=config["training"]["batch_size"],
                                 shuffle=False, drop_last=True))).to(device)
    saved = copy.deepcopy(model.state_dict())
    model.train()
    probe, _ = losses(first)
    optimizer.zero_grad(set_to_none=True)
    probe.backward()
    optimizer.step()
    _, probe_parts = losses(first)
    model.zero_grad(set_to_none=True)
    probe_parts["detect"].backward()
    detect_after_one_step = gradient_norms(model)
    model.load_state_dict(saved)
    optimizer = torch.optim.AdamW(trainable, lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])

    model.zero_grad(set_to_none=True)
    _, audit_parts = losses(first)
    audit_parts["detect"].backward()
    gradient_audit = {"detect": gradient_norms(model)}
    model.zero_grad(set_to_none=True)
    if detect_after_one_step["apra"] <= 0 or detect_after_one_step["scoring_head"] <= 0:
        raise RuntimeError("L_detect gradient does not reach all APRA modules")

    history = []
    updates = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(config["training"]["epochs"]):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        for indices in loader:
            indices = indices.to(device)
            total, parts = losses(indices)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norms = gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(trainable, config["training"]["gradient_clip"])
            optimizer.step()
            updates += 1
            batches += 1
            for name, value in {"total": total, **parts}.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            for name, value in norms.items():
                sums[f"grad_{name}"] = sums.get(f"grad_{name}", 0.0) + value
        row = {"epoch": epoch + 1, **{key: value / batches for key, value in sums.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)

    elapsed = time.perf_counter() - started
    numeric_change = max(float((value.detach().cpu() - initial_numeric[name]).abs().max())
                         for name, value in model.numeric.state_dict().items())
    output = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": config["model_config"],
        "legacy_model_config": {"aligner": "apra", "fusion": "apra_head", "reject": True,
                                **config["model_config"]},
        "dataset": cached["dataset"], "variant": "E6_Detect_only", "seed": seed,
        "history": history, "gradient_audit_by_loss": gradient_audit,
        "detect_gradient_after_one_step": detect_after_one_step,
        "numeric_max_abs_change": numeric_change,
        "parameter_counts": {"all": sum(p.numel() for p in model.parameters()),
                             "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                             "numeric": sum(p.numel() for p in model.numeric.parameters()),
                             "apra": sum(p.numel() for p in model.aligner.parameters()),
                             "scoring_head": sum(p.numel() for p in model.fusion.parameters())},
        "elapsed_seconds": elapsed, "optimizer_updates": updates,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "labels_used": False, "validation_split_used": False,
        "checkpoint_selection": "fixed_final_epoch", "source_cache": str(args.cache),
        "numeric_initialization": str(full_checkpoint), "visual_initialization": str(visual_checkpoint),
        "loss_weights": config["loss_weights"], "active_losses": ["detect"],
        "active_sublosses": ["hard_ranking", "unchanged", "reversible_warp"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(output, temporary)
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
