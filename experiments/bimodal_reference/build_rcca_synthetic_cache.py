#!/usr/bin/env python3
"""Build one immutable RCCA training cache with paired synthetic CLIP features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from collections import Counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.curve_aware_clip import create_openai_block6_encoder, encode_block6
from src.models.bimodal_reference_vit4ts import _visual_costs
from src.models.numeric_temporal_matching import match_paths
from src.models.vfg_paired_vit4ts import _patch_tokens, select_common_visual_windows


class WindowRenderer:
    """Reuse the exact ViT4TS Matplotlib canvas for independent windows."""
    def __init__(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(2.24, 2.24), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        axis = self.figure.add_subplot(111)
        (self.line,) = axis.plot(np.arange(224), np.zeros(224), "-", linewidth=1,
                                 marker="*", markersize=0.1, color="black")
        axis.set_xlim((0, 223)); axis.set_ylim((0, 1))
        axis.set_xticks([]); axis.set_yticks([])
        self.figure.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
        axis.margins(0, 0)

    def __call__(self, windows: np.ndarray) -> torch.Tensor:
        rendered = []
        for values in windows:
            self.line.set_ydata(values)
            self.canvas.draw()
            rgba = np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)
            rendered.append(np.moveaxis(rgba[..., :3], -1, 0).astype(np.float32) / 255.0)
        return torch.from_numpy(np.stack(rendered))


def perturb_window(raw: torch.Tensor, spec: dict, generator: torch.Generator):
    result = raw.clone()
    span = int(torch.randint(int(spec["min_patch_span"]),
                             int(spec["max_patch_span"]) + 1, (1,), generator=generator))
    start_patch = int(torch.randint(0, 15 - span, (1,), generator=generator))
    start, end = start_patch * 16, (start_patch + span) * 16
    segment = result[start:end]
    scale = float(torch.std(raw).clamp_min(0.25))
    kind_index = int(torch.randint(0, len(spec["types"]), (1,), generator=generator))
    kind = spec["types"][kind_index]

    def uniform(bounds):
        value = torch.rand((), generator=generator).item()
        return float(bounds[0] + value * (bounds[1] - bounds[0]))

    sign = -1.0 if torch.rand((), generator=generator).item() < 0.5 else 1.0
    length = end - start
    if kind == "amplitude_jump":
        segment.add_(sign * uniform(spec["amplitude_range"]) * scale)
    elif kind == "trend_shift":
        segment.add_(torch.linspace(0, sign * uniform(spec["trend_range"]) * scale, length))
    elif kind == "frequency_change":
        cycles = uniform(spec["frequency_cycles_range"])
        phase = uniform((0.0, 2.0 * np.pi))
        wave = torch.sin(torch.linspace(phase, phase + 2 * np.pi * cycles, length))
        segment.add_(wave * scale)
    elif kind == "volatility_burst":
        noise = torch.randn(length, generator=generator)
        segment.add_(noise * uniform(spec["volatility_range"]) * scale)
    elif kind == "morphology":
        mix = uniform(spec["morphology_mix_range"])
        transformed = torch.flip(segment.clone(), dims=(0,)).roll(max(1, length // 5))
        segment.copy_((1.0 - mix) * segment + mix * transformed)
    else:
        raise ValueError(kind)
    mask = torch.zeros(14, dtype=torch.bool); mask[start_patch:start_patch + span] = True
    return result, mask, kind, start_patch, span


def visual_window(raw: np.ndarray, stats: dict) -> np.ndarray:
    residual = raw * float(stats["scale"]) + float(stats["center"])
    lo, hi = float(stats["visual_min"]), float(stats["visual_max"])
    if hi <= lo:
        return np.zeros_like(raw)
    return np.clip((residual - lo) / (hi - lo), 0.0, 1.0)


def retain_reference_eligible_windows(base: dict):
    """Drop training signals for which overlap exclusion leaves no legal reference."""
    records = json.loads(base["records_json"])
    counts = Counter(record["signal"] for record in records)
    # With four-window overlap exclusion, every query has a legal same-signal
    # reference only when there are at least eight ordered windows.
    retained = torch.tensor([counts[record["signal"]] >= 8 for record in records])
    excluded = {signal: count for signal, count in counts.items() if count < 8}
    if not retained.any():
        raise RuntimeError("no windows retain a legal non-overlapping T3 reference")
    filtered_records = [record for record, keep in zip(records, retained.tolist()) if keep]
    filtered = dict(base)
    filtered["raw"] = base["raw"][retained]
    filtered["visual"] = base["visual"][retained]
    filtered["records_json"] = json.dumps(filtered_records, separators=(",", ":"))
    filtered["signals"] = len({record["signal"] for record in filtered_records})
    return filtered, excluded


def reference_paths(base: dict, device: torch.device):
    records = json.loads(base["records_json"])
    visual = _patch_tokens(base["visual"])
    ids = torch.empty(len(records), dtype=torch.long)
    columns = torch.empty(len(records), 14, dtype=torch.long)
    by_signal: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_signal.setdefault(record["signal"], []).append(index)
    for signal, global_indices in by_signal.items():
        local = visual[global_indices].to(device)
        pool = select_common_visual_windows(local)
        costs = _visual_costs(local.reshape(len(local), 14, 14, -1), pool)
        matched = match_paths(costs, pool, band=2, penalty_scale=0.1)
        mapping = torch.tensor(global_indices)
        ids[global_indices] = mapping[matched.reference_ids.cpu()]
        columns[global_indices] = matched.reference_columns.round().long().cpu()
        if torch.any((ids[global_indices] - torch.tensor(global_indices)).abs() < 4):
            raise RuntimeError(f"{signal}: T3 reference violates overlap exclusion")
    return ids, columns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--render-batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    base_path = Path(config["base_training_cache_root"]) / f"{args.dataset}.pt"
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    if base["labels_read"] or not base["uses_all_windows"]:
        raise RuntimeError("base cache violates unlabeled protocol")
    base, excluded_short_signals = retain_reference_eligible_windows(base)
    records = json.loads(base["records_json"])
    stats = json.loads(Path(config["normalization_stats"]).read_text())
    generator = torch.Generator().manual_seed(args.seed)
    synthetic, masks, perturbations = [], [], []
    for index, (raw, record) in enumerate(zip(base["raw"], records)):
        altered, mask, kind, start, span = perturb_window(raw, config["synthetic"], generator)
        synthetic.append(altered); masks.append(mask)
        perturbations.append({"index": index, "signal": record["signal"], "kind": kind,
                              "start_patch": start, "patch_span": span})
    synthetic_raw = torch.stack(synthetic)
    patch_mask = torch.stack(masks)
    if torch.any(patch_mask.sum(1) < config["synthetic"]["min_patch_span"]):
        raise RuntimeError("invalid perturbation masks")

    device = torch.device(args.device)
    reference_ids, reference_columns = reference_paths(base, device)
    encoder = create_openai_block6_encoder(config["clip_checkpoint"], device)
    renderer = WindowRenderer()
    visual_pieces = []
    started = time.perf_counter()
    for start in range(0, len(synthetic_raw), args.render_batch_size):
        stop = min(start + args.render_batch_size, len(synthetic_raw))
        normalized_images = np.stack([
            visual_window(synthetic_raw[index].numpy(),
                          stats[args.dataset][records[index]["signal"]])
            for index in range(start, stop)
        ])
        images = renderer(normalized_images)
        tokens = encode_block6(encoder, images, device=device,
                               batch_size=args.clip_batch_size)
        visual_pieces.append(tokens.half())
        print(json.dumps({"dataset": args.dataset, "seed": args.seed,
                          "encoded": stop, "windows": len(synthetic_raw),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
    synthetic_visual = torch.cat(visual_pieces)
    payload = {
        "dataset": args.dataset, "seed": args.seed,
        "raw": base["raw"], "visual": base["visual"],
        "synthetic_raw": synthetic_raw, "synthetic_visual": synthetic_visual,
        "patch_mask": patch_mask, "reference_ids": reference_ids,
        "reference_columns": reference_columns,
        "records_json": base["records_json"],
        "perturbations_json": json.dumps(perturbations, separators=(",", ":")),
        "windows": len(synthetic_raw), "signals": base["signals"],
        "labels_read": False, "uses_all_windows": True,
        "t3_reference_fixed": True, "dtw_path_fixed": True,
        "renderer": "reused_ViT4TS_Matplotlib_canvas_224x224",
        "clip": "frozen_OpenAI_ViT-B-16_Block6",
        "elapsed_seconds": time.perf_counter() - started,
        "source_base_cache": str(base_path),
        "excluded_short_signals": excluded_short_signals,
        "excluded_short_windows": int(sum(excluded_short_signals.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "shape": list(synthetic_visual.shape),
                      "elapsed_seconds": payload["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
