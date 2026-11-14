#!/usr/bin/env python3
"""Train on frozen B3 plus the calibrated final point score.

Frozen B3 counterfactuals and T3 paths must be built before training. Only
the model parameters are optimized; evaluation labels are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model import load_model_checkpoint, gradient_norms, reversible_warp


@torch.no_grad()
def projected_visual(model, visual: torch.Tensor, device: torch.device,
                     batch_size: int = 256) -> torch.Tensor:
    return torch.cat([model.visual(visual[start:start + batch_size].float().to(device)).half()
                      for start in range(0, len(visual), batch_size)])


def restore_point(local: torch.Tensor) -> torch.Tensor:
    """Differentiable 14-column to point restoration for covered windows."""
    windows = local.shape[0]
    interpolated = F.interpolate(local.double()[:, None], size=224, mode="linear",
                                 align_corners=False)[:, 0]
    total = 224 + 56 * (windows - 1)
    positions = torch.arange(224, device=local.device)[None] + \
        56 * torch.arange(windows, device=local.device)[:, None]
    point = local.new_zeros(total, dtype=torch.float64)
    counts = local.new_zeros(total, dtype=torch.float64)
    point.scatter_add_(0, positions.reshape(-1), interpolated.reshape(-1))
    counts.scatter_add_(0, positions.reshape(-1), torch.ones_like(interpolated).reshape(-1))
    return point / counts.clamp_min(1)


def calibrated_delta(local: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Apply the same signed positive-median calibration as inference."""
    point = restore_point(local)
    detached = point.detach().abs()
    positive = detached[torch.isfinite(detached) & (detached > 1e-12)]
    if not positive.numel():
        return point, 1.0
    reference = base.detach()
    reference = reference[torch.isfinite(reference) & (reference > 1e-8)]
    base_median = float(torch.quantile(reference, .5)) if reference.numel() else 1.0
    scale = float(np.clip(base_median / max(float(torch.quantile(positive, .5)), 1e-12),
                          1e-4, 1e4))
    return point * scale, scale


def final_score(local: torch.Tensor, base: torch.Tensor, gamma: float = .05):
    delta, scale = calibrated_delta(local, base)
    return base + gamma * delta, scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--frozen-b3-root", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="check one signal and gradients without training")
    args = parser.parse_args()
    if args.output.exists() and not args.smoke:
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    training_data = torch.load(args.training_data, map_location="cpu", weights_only=False)
    if training_data["labels_read"] or int(training_data["seed"]) != seed:
        raise RuntimeError("invalid unlabeled synthetic cache")
    dataset = training_data["dataset"]
    initial_checkpoint = args.initial_checkpoint or Path(config["initial_checkpoint_root"]) / f"{dataset}.pt"
    frozen_root = args.frozen_b3_root or Path(config["frozen_b3_root"])
    model, original = load_model_checkpoint(initial_checkpoint, device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    clean_raw = training_data["raw"].to(device)
    synthetic_raw = training_data["synthetic_raw"].to(device)
    clean_visual = projected_visual(model, training_data["visual"], device)
    synthetic_visual = projected_visual(model, training_data["synthetic_visual"], device)
    records = json.loads(training_data["records_json"])
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["signal"], []).append(index)
    frozen = {}
    for signal, indices in groups.items():
        path = frozen_root / dataset / f"{signal}.npz"
        with np.load(path, allow_pickle=False) as row:
            if not np.array_equal(row["source_indices"], indices):
                raise RuntimeError(f"source index mismatch: {signal}")
            if dataset in {"MSL", "SMAP"} and bool(row["historical_anchor_used"]):
                raise RuntimeError(f"official Train cache used Test B3 anchor: {signal}")
            frozen[signal] = {
                name: torch.from_numpy(row[name].copy()).to(device)
                for name in ("clean_score", "synthetic_score", "point_mask",
                             "clean_reference_ids", "clean_reference_columns",
                             "synthetic_reference_ids", "synthetic_reference_columns")}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])

    def forward_one(signal: str):
        indices = torch.tensor(groups[signal], device=device, dtype=torch.long)
        row = frozen[signal]
        nq = model.numeric(clean_raw[indices])
        ns = model.numeric(synthetic_raw[indices])
        vq = clean_visual[indices].float()
        vs = synthetic_visual[indices].float()
        def output(numeric, visual, prefix):
            ids = row[f"{prefix}_reference_ids"].long()
            columns = row[f"{prefix}_reference_columns"].long()[..., None]
            nr_full, vr_full = numeric[ids], visual[ids]
            nr = nr_full.gather(1, columns.expand(-1, -1, nr_full.shape[-1]))
            vr = vr_full.gather(1, columns.expand(-1, -1, vr_full.shape[-1]))
            return model.forward_tokens(numeric, visual, nr, vr)
        clean = output(nq, vq, "clean")
        synthetic = output(ns, vs, "synthetic")
        warped_n, target = reversible_warp(nq)
        warped_v = target @ vq
        ids = row["clean_reference_ids"].long()
        columns = row["clean_reference_columns"].long()[..., None]
        nr_full, vr_full = nq[ids], vq[ids]
        nr = nr_full.gather(1, columns.expand(-1, -1, nr_full.shape[-1]))
        vr = vr_full.gather(1, columns.expand(-1, -1, vr_full.shape[-1]))
        warped = model.forward_tokens(warped_n, warped_v, nr, vr)
        gamma = float(config["loss_weights"]["gamma"])
        clean_score, clean_scale = final_score(clean["delta"], row["clean_score"], gamma)
        synthetic_score, synthetic_scale = final_score(synthetic["delta"], row["synthetic_score"], gamma)
        warped_delta, _ = calibrated_delta(warped["delta"], row["clean_score"])
        clean_delta, _ = calibrated_delta(clean["delta"], row["clean_score"])
        mask = row["point_mask"].bool()
        base_difference = row["synthetic_score"] - row["clean_score"]
        base_positive = row["clean_score"][row["clean_score"] > 1e-8]
        base_median = (torch.quantile(base_positive, .5) if base_positive.numel()
                       else row["clean_score"].new_tensor(1.0)).clamp_min(1e-4)
        margin = (float(config["loss_weights"]["margin_fraction"]) * base_median).detach()
        difficulty = (margin - base_difference).relu()
        hard_weight = (1 + 2 * difficulty / difficulty.mean().clamp_min(1e-8)).detach()
        score_difference = synthetic_score - clean_score
        ranking = (hard_weight[mask] * F.softplus(
            (margin - score_difference[mask]) / margin)).mean()
        unchanged = (F.smooth_l1_loss(score_difference[~mask] / margin,
                                      torch.zeros_like(score_difference[~mask]))
                     if (~mask).any() else ranking.new_zeros(()))
        warp = F.smooth_l1_loss(gamma * (warped_delta - clean_delta) / margin,
                                torch.zeros_like(clean_delta))
        total = (ranking + float(config["loss_weights"]["unchanged"]) * unchanged
                 + float(config["loss_weights"]["reversible_warp"]) * warp)
        return total, {"ranking": ranking, "unchanged": unchanged, "warp": warp,
                       "clean_scale": clean_scale, "synthetic_scale": synthetic_scale,
                       "margin": float(margin)}

    first = next(iter(groups))
    model.train()
    smoke_loss, smoke = forward_one(first)
    optimizer.zero_grad(set_to_none=True)
    smoke_loss.backward()
    gradient = gradient_norms(model)
    if not np.isfinite(float(smoke_loss.detach())) or gradient["alignment"] <= 0 or gradient["scoring_head"] <= 0:
        raise RuntimeError(f"invalid gradient smoke: {gradient}")
    print(json.dumps({"stage": "smoke_passed", "dataset": dataset,
                      "signal": first, "loss": float(smoke_loss.detach()),
                      "gradients": gradient, "scales": [smoke["clean_scale"],
                                                    smoke["synthetic_scale"]]}), flush=True)
    if args.smoke:
        return 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(config["training"]["epochs"]):
        model.train()
        order = rng.permutation(list(groups))
        sums = {name: 0.0 for name in ("total", "ranking", "unchanged", "warp")}
        for signal in order:
            total, parts = forward_one(str(signal))
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config["training"]["gradient_clip"])
            optimizer.step()
            sums["total"] += float(total.detach())
            for name in ("ranking", "unchanged", "warp"):
                sums[name] += float(parts[name].detach())
        row = {"epoch": epoch + 1,
               **{name: value / len(groups) for name, value in sums.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    parameter_change = max(float((state[name] - initial[name]).abs().max()) for name in state)
    output = {"state_dict": state, "model_config": original["model_config"],
              "legacy_model_config": original.get("legacy_model_config", {}),
              "dataset": dataset, "variant": "Model", "seed": seed,
              "objective": "frozen_b3_plus_calibrated_apra_point_score",
              "history": history, "gradient_audit_by_loss": {"final_score": gradient},
              "parameter_counts": original.get("parameter_counts", {}),
              "parameter_change_from_E6": parameter_change,
              "elapsed_seconds": time.perf_counter() - started,
              "labels_used": False, "checkpoint_selection": "fixed_final_epoch",
              "source_training_data": str(args.training_data),
              "source_frozen_b3": str(frozen_root / dataset),
              "loss_weights": config["loss_weights"],
              "training_config": config["training"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(output, temporary)
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
