#!/usr/bin/env python3
"""Train one dataset-native AMR checkpoint without labels or model selection."""
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

from src.models.bimodal_reference import AlignedMultimodalReference, ModelConfig, patch_alignment_loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cached.get("labels_read") is not False or cached.get("uses_all_windows") is not True:
        raise ValueError("training cache violates the preregistered unlabeled/all-window protocol")
    dataset = TensorDataset(cached["raw"], cached["visual"].float())
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=generator, drop_last=True, num_workers=0)
    if len(loader) == 0:
        raise ValueError("dataset has fewer windows than one training batch")
    model = AlignedMultimodalReference().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    history, peak = [], 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); sums: dict[str, float] = {}; batches = 0
        for raw, visual in loader:
            raw, visual = raw.to(args.device), visual.to(args.device)
            raw = raw * (0.9 + 0.2 * torch.rand(raw.shape[0], 1, device=args.device)) + 0.01 * torch.randn_like(raw)
            output = model(raw, visual)
            loss, parts = patch_alignment_loss(output, include_reference=True)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for name, value in {"total": loss, **parts}.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            batches += 1
        row = {"epoch": epoch + 1, **{name: value / batches for name, value in sums.items()}}
        history.append(row); print(json.dumps(row), flush=True)
        if torch.cuda.is_available():
            peak = max(peak, torch.cuda.max_memory_allocated(args.device))
    payload = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": ModelConfig().__dict__, "dataset": cached["dataset"],
        "training": vars(args) | {"cache": str(args.cache), "output": str(args.output)},
        "history": history, "train_windows": len(dataset), "signals": cached["signals"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak, "checkpoint_selection": "fixed_final_epoch",
        "labels_used": False, "validation_split_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output, payload["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
