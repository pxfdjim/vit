#!/usr/bin/env python3
"""Audit APRA losses on identical final checkpoints, batches and parameters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.apra_acef import (  # noqa: E402
    APRAACEF, LossWeights, detection_loss, path_loss,
    selective_alignment_loss, selective_time_warp,
)


GROUPS = {
    "aligner": lambda n: n.startswith("aligner."),
    "residual_module": lambda n: n.startswith("fusion.input.") or n.startswith("fusion.temporal."),
    "scoring_head": lambda n: n.startswith("fusion.out."),
    "numeric": lambda n: n.startswith("numeric."),
}


def projected_visual(model, visual, device, batch_size=256):
    with torch.no_grad():
        return torch.cat([model.visual(visual[s:s+batch_size].float().to(device)).half()
                          for s in range(0, len(visual), batch_size)])


def base_proxy(nq, vq, nr, vr):
    return (.5 * (1-F.cosine_similarity(nq, nr, dim=-1)) +
            .5 * (1-F.cosine_similarity(vq, vr, dim=-1))).detach()


def gradients(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    result = {}
    for group, predicate in GROUPS.items():
        pieces, parameter_nonzero, parameter_total, element_nonzero, element_total = [], 0, 0, 0, 0
        for name, parameter in model.named_parameters():
            if not predicate(name):
                continue
            parameter_total += 1
            gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach()
            nonzero = int(torch.count_nonzero(gradient).item())
            parameter_nonzero += int(nonzero > 0)
            element_nonzero += nonzero
            element_total += gradient.numel()
            pieces.append(gradient.reshape(-1).float().cpu())
        vector = torch.cat(pieces) if pieces else torch.zeros(0)
        result[group] = {
            "vector": vector,
            "norm": float(vector.norm()),
            "parameter_coverage": parameter_nonzero / max(parameter_total, 1),
            "element_coverage": element_nonzero / max(element_total, 1),
        }
    return result


def cosine(left, right):
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator) if denominator > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    seed = int(config["primary_seed"])
    device = torch.device(args.device)
    rows, cosine_rows = [], []
    for dataset in config["datasets"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        checkpoint_path = Path(config["selective_run"]) / "checkpoints/E2_select" / f"seed_{seed}" / f"{dataset}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = checkpoint["model_config"]
        model = APRAACEF(**model_config).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.train()
        cache_path = Path(config["rcca_training_cache"]) / f"{dataset}.pt"
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["labels_read"]:
            raise RuntimeError("training cache contains labels")
        raw = cached["raw"].to(device); synthetic_raw = cached["synthetic_raw"].to(device)
        altered = cached["patch_mask"].to(device)
        reference_ids = cached["reference_ids"].to(device)
        reference_columns = cached["reference_columns"].to(device)
        visual = projected_visual(model, cached["visual"], device)
        synthetic_visual = projected_visual(model, cached["synthetic_visual"], device)
        indices = torch.arange(config["training"]["batch_size"], device=device)
        ids = reference_ids[indices]; columns = reference_columns[indices][..., None]
        nq = model.numeric(raw[indices]); ns = model.numeric(synthetic_raw[indices])
        nr_full = model.numeric(raw[ids]); nr = nr_full.gather(1, columns.expand(-1, -1, nr_full.shape[-1]))
        vq = visual[indices].float(); vs = synthetic_visual[indices].float()
        vr_full = visual[ids].float(); vr = vr_full.gather(1, columns.expand(-1, -1, vr_full.shape[-1]))
        clean = model.forward_tokens(nq, vq, nr, vr)
        synthetic = model.forward_tokens(ns, vs, nr, vr)
        generator = torch.Generator(device=device).manual_seed(seed + 17041)
        count = len(raw)
        modes = torch.arange(count, device=device) % 3
        amplitudes = .35 + .65 * torch.rand(count, generator=generator, device=device)
        amplitudes *= torch.where(torch.rand(count, generator=generator, device=device) < .5, -1., 1.)
        centers = .3 + .4 * torch.rand(count, generator=generator, device=device)
        warped_n, target, _ = selective_time_warp(nq, modes[indices], amplitudes[indices], centers[indices])
        warped = model.forward_tokens(warped_n, target @ vq, nr, vr)
        perturbations = json.loads(cached["perturbations_json"])
        reject_kinds = set(config["selective_supervision"]["reject_perturbations"])
        structural_rows = torch.tensor([row["kind"] in reject_kinds for row in perturbations], device=device)
        structural_reject = altered[indices] & structural_rows[indices, None]
        select, _ = selective_alignment_loss(
            clean, synthetic, warped, target, altered[indices], structural_reject)
        losses = {
            "path": path_loss(warped, target),
            "select": select,
            "detect": detection_loss(
                clean, synthetic, warped, base_proxy(nq, vq, nr, vr),
                base_proxy(ns, vs, nr, vr), altered[indices],
                LossWeights(**config["loss_weights"])),
        }
        by_loss = {name: gradients(model, value) for name, value in losses.items()}
        for loss_name, groups in by_loss.items():
            for group, values in groups.items():
                rows.append({"dataset": dataset, "checkpoint": str(checkpoint_path),
                             "batch_start": 0, "batch_size": len(indices), "loss": loss_name,
                             "loss_value": float(losses[loss_name].detach()), "group": group,
                             "gradient_norm": values["norm"],
                             "parameter_coverage": values["parameter_coverage"],
                             "element_coverage": values["element_coverage"]})
        for group in GROUPS:
            for left, right in (("path", "detect"), ("select", "detect"), ("path", "select")):
                cosine_rows.append({"dataset": dataset, "group": group, "left": left, "right": right,
                                    "cosine": cosine(by_loss[left][group]["vector"],
                                                     by_loss[right][group]["vector"])})
        del model, cached, visual, synthetic_visual
        torch.cuda.empty_cache()
        print(json.dumps({"dataset": dataset, "status": "passed"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows); frame.to_csv(args.output, index=False)
    cosines = pd.DataFrame(cosine_rows); cosines.to_csv(args.output.with_name("gradient_direction_cosines.csv"), index=False)
    summary = frame.groupby(["loss", "group"], as_index=False).agg(
        gradient_norm_mean=("gradient_norm", "mean"), gradient_norm_median=("gradient_norm", "median"),
        parameter_coverage_mean=("parameter_coverage", "mean"),
        element_coverage_mean=("element_coverage", "mean"))
    summary.to_csv(args.output.with_name("gradient_path_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
