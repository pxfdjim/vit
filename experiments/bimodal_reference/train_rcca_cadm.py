#!/usr/bin/env python3
"""Train one dataset/variant/seed RCCA-CADM checkpoint."""
from __future__ import annotations

import argparse
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

from src.models.rcca_cadm import (
    LossWeights, RCCACADM, gradient_norms, load_numeric_from_full,
    reference_conditioned_loss,
)


def model_outputs(model, nq, vq, nr, vr):
    output = model.forward_tokens(nq, vq, nr, vr)
    output.update({"numeric_query": nq, "visual_query": vq,
                   "numeric_reference": nr, "visual_reference": vr})
    return output


def projected_visual(model, visual, device, batch_size=256):
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(visual), batch_size):
            pieces.append(model.visual(visual[start:start + batch_size].float().to(device)).half())
    return torch.cat(pieces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    spec = config["variants"][args.variant]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cached["labels_read"] or cached["seed"] != args.seed:
        raise RuntimeError("synthetic cache integrity failure")
    model = RCCACADM(spec["architecture"]).to(device)
    full_checkpoint = Path(config["full_checkpoint_root"]) / f"{cached['dataset']}.pt"
    load_numeric_from_full(model, str(full_checkpoint))
    initial_numeric = {name: value.detach().cpu().clone()
                       for name, value in model.numeric.state_dict().items()}

    raw = cached["raw"].to(device)
    synthetic_raw = cached["synthetic_raw"].to(device)
    mask = cached["patch_mask"].to(device)
    reference_ids = cached["reference_ids"].to(device)
    reference_columns = cached["reference_columns"].to(device)
    visual = projected_visual(model, cached["visual"], device)
    synthetic_visual = projected_visual(model, cached["synthetic_visual"], device)
    weights = LossWeights(**config["loss_weights"])

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(torch.arange(len(raw)), batch_size=config["training"]["batch_size"],
                        shuffle=True, generator=generator, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])

    def evaluate_batch(indices):
        ref_ids = reference_ids[indices]
        columns = reference_columns[indices][..., None]
        nq = model.numeric(raw[indices]); ns = model.numeric(synthetic_raw[indices])
        nr_full = model.numeric(raw[ref_ids])
        nr = nr_full.gather(1, columns.expand(-1, -1, nr_full.shape[-1]))
        vq = visual[indices].float(); vs = synthetic_visual[indices].float()
        vr_full = visual[ref_ids].float()
        vr = vr_full.gather(1, columns.expand(-1, -1, vr_full.shape[-1]))
        clean = model_outputs(model, nq, vq, nr, vr)
        synthetic = model_outputs(model, ns, vs, nr, vr)
        return clean, synthetic, nr, vr

    first = next(iter(DataLoader(torch.arange(len(raw)),
                                 batch_size=config["training"]["batch_size"],
                                 shuffle=False, drop_last=True))).to(device)
    model.train(); clean, synthetic, nr, vr = evaluate_batch(first)
    audit_loss, audit_parts = reference_conditioned_loss(
        clean, synthetic, mask[first], weights,
        use_synthetic=bool(spec["synthetic"]), use_ranking=bool(spec["ranking"]))
    model.zero_grad(set_to_none=True); audit_loss.backward()
    gradient_audit = {"total_loss": float(audit_loss.detach()), **gradient_norms(model),
                      **{f"loss_{k}": float(v.detach()) for k, v in audit_parts.items()}}
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        base = evaluate_batch(first)[0]["logits"]
        rolled_nr, rolled_vr = nr.roll(1, 0), vr.roll(1, 0)
        replaced = model.forward_tokens(clean["numeric_query"], clean["visual_query"],
                                        rolled_nr, rolled_vr)["logits"]
        query_changed = model.forward_tokens(
            clean["numeric_query"].roll(1, 0), clean["visual_query"].roll(1, 0),
            nr, vr)["logits"]
        sensitivity = {
            "reference_replacement_mean_abs_logit_change": float((base - replaced).abs().mean()),
            "query_replacement_mean_abs_logit_change": float((base - query_changed).abs().mean()),
        }
    if spec["architecture"] != "no_rcca" and sensitivity[
            "reference_replacement_mean_abs_logit_change"] <= 1e-7:
        raise RuntimeError("RCCA failed reference sensitivity gate")

    history, updates = [], 0
    started = time.perf_counter(); peak = 0
    for epoch in range(config["training"]["epochs"]):
        model.train(); sums = {}; batches = 0
        for indices in loader:
            indices = indices.to(device)
            clean, synthetic, _, _ = evaluate_batch(indices)
            loss, parts = reference_conditioned_loss(
                clean, synthetic, mask[indices], weights,
                use_synthetic=bool(spec["synthetic"]), use_ranking=bool(spec["ranking"]))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norms = gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step(); updates += 1; batches += 1
            for name, value in {"total": loss, **parts}.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            for name, value in norms.items():
                sums[f"grad_{name}"] = sums.get(f"grad_{name}", 0.0) + value
        row = {"epoch": epoch + 1, **{name: value / batches for name, value in sums.items()}}
        history.append(row); print(json.dumps(row), flush=True)
        peak = max(peak, torch.cuda.max_memory_allocated(device))
    elapsed = time.perf_counter() - started
    final_numeric_change = max(float((value.detach().cpu() - initial_numeric[name]).abs().max())
                               for name, value in model.numeric.state_dict().items())
    if final_numeric_change <= 0:
        raise RuntimeError("numeric encoder did not update")
    payload = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": {"architecture": spec["architecture"], "hidden": 192,
                         "heads": 4, "layers": 2},
        "dataset": cached["dataset"], "variant": args.variant, "seed": args.seed,
        "history": history, "gradient_audit": gradient_audit,
        "sensitivity_audit": sensitivity,
        "numeric_initialization": str(full_checkpoint),
        "numeric_max_abs_change": final_numeric_change,
        "parameter_counts": {
            "all": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "numeric": sum(p.numel() for p in model.numeric.parameters()),
            "rcca": sum(p.numel() for p in getattr(model, "rcca", []).parameters())
                    if hasattr(model, "rcca") else 0,
            "cadm": sum(p.numel() for p in model.cadm.parameters()),
        },
        "elapsed_seconds": elapsed, "optimizer_updates": updates,
        "peak_cuda_memory_bytes": peak, "labels_used": False,
        "validation_split_used": False, "checkpoint_selection": "fixed_final_epoch",
        "source_cache": str(args.cache), "loss_weights": config["loss_weights"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp"); torch.save(payload, temporary)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "elapsed_seconds": elapsed,
                      "updates": updates, **sensitivity}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

