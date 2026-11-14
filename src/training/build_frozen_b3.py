#!/usr/bin/env python3
"""Reconstruct frozen B3 on paired clean/synthetic window streams.

This runs the original CLIP, T3, raw numeric and AMR numeric scoring functions.
The source synthetic windows are independently perturbed; their stitched point
stream is a controlled counterfactual, not an observed time series.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
# The frozen B3 counterfactual uses the original, immutable scoring implementation.
SOURCE = ROOT / ".worktrees/rcca-cadm-e6aad7d"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "src"))

import open_clip
from models.clip_vision import multi_scale_aggregation
from models.model_utils import patch_scale
from experiments.bimodal_reference.build_rcca_synthetic_cache import WindowRenderer, visual_window
from src.models.bimodal_reference import load_model
from src.models.bimodal_reference_vit4ts import _calibrate, _token_costs, _visual_costs
from src.models.fixed_reference_numeric_evidence import build_numeric_evidence, score_on_fixed_paths
from src.models.numeric_temporal_matching import match_paths
from src.models.temporal_vit4ts import _column_map
from src.models.vfg_paired_vit4ts import (
    _patch_tokens, point_scores_from_maps, select_common_visual_windows, stable_scale_map)



def encoder(device: torch.device):
    checkpoint = ROOT / "data/model_cache/ViT-B-16.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(checkpoint), vision_cfg={"output_tokens": True},
        force_quick_gelu=True, weights_only=False)
    clip = clip.to(device).eval()
    for parameter in clip.parameters():
        parameter.requires_grad_(False)
    return clip, patch_scale(image_size=(224, 224))


@torch.inference_mode()
def final_features(clip, mask, renderer, windows, signal_stats, device, batch_size=32):
    pieces = {name: [] for name in ("large", "mid", "patch")}
    large_mask = mid_mask = None
    for start in range(0, len(windows), batch_size):
        values = np.stack([visual_window(row.numpy(), signal_stats)
                           for row in windows[start:start + batch_size]])
        images = renderer(values).to(device)
        _, patches = clip.encode_image(images)
        large, mid, large_mask, mid_mask = multi_scale_aggregation(patches, 16, mask)
        for name, tensor in (("large", large), ("mid", mid), ("patch", patches)):
            pieces[name].append(tensor.detach().cpu().float())
    return {name: torch.cat(pieces[name]) for name in pieces} | {
        "large_mask": large_mask.cpu(), "mid_mask": mid_mask.cpu()}


@torch.inference_mode()
def b3_score(raw, block, final, amr, device):
    patch = _patch_tokens(block).float().to(device)
    final_patch = _patch_tokens(final["patch"]).to(device)
    pool = select_common_visual_windows(final_patch)
    visual = _visual_costs(patch.reshape(len(patch), 14, 14, -1), pool)
    path = match_paths(visual, pool, band=2, penalty_scale=.1)
    raw_cost = build_numeric_evidence(raw, pool)
    numeric_tokens = []
    for start in range(0, len(raw), 32):
        numeric_tokens.append(amr(raw[start:start + 32].to(device),
                                  patch[start:start + 32])["numeric"])
    numeric_tokens = torch.cat(numeric_tokens)
    numeric_cost, _ = _calibrate(visual, _token_costs(numeric_tokens, pool))
    t3_local = .5 * path.scores
    large_map = stable_scale_map(final["large"], final["large_mask"], device=device)
    mid_map = stable_scale_map(final["mid"], final["mid_mask"], device=device)
    t3_map = (large_map + mid_map + _column_map(t3_local).cpu().double()) / 3.0
    length = 224 + 56 * (len(raw) - 1)
    t3 = np.asarray(point_scores_from_maps(t3_map, full_length=length, device=device), np.float64)
    raw_local = score_on_fixed_paths(raw_cost.combined, path)
    numeric_local = score_on_fixed_paths(numeric_cost, path)
    def restore(local):
        return np.asarray(point_scores_from_maps(_column_map(local.double()),
                            full_length=length, device=device), np.float64)
    def calibrate(local):
        values = restore(local)
        positive = lambda x: x[np.isfinite(x) & (x > 1e-8)]
        # Official NASA Train contains constant signals. Their evidence can
        # have no strictly positive samples; use the inference default rather
        # than taking the median of an empty array and propagating NaNs.
        t3_positive = positive(t3)
        value_positive = positive(values)
        t3_median = float(np.median(t3_positive)) if t3_positive.size else 1.0
        value_median = float(np.median(value_positive)) if value_positive.size else 1.0
        scale = float(np.clip(t3_median / max(value_median, 1e-8), 1e-4, 1e4))
        return values * scale
    score = .9 * t3 + .05 * calibrate(raw_local) + .05 * calibrate(numeric_local)
    if not np.isfinite(score).all():
        raise RuntimeError("frozen B3 reconstruction contains nonfinite scores")
    return score, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--signal")
    parser.add_argument("--only-clean", action="store_true")
    parser.add_argument("--debug-clean-output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cache_root = Path(config["synthetic_training_cache_root"])
    cache_path = args.training_data or cache_root / f"{args.dataset}.pt"
    output_root = args.output_root or Path(config["frozen_b3_root"])
    stats_path = Path(config["normalization_stats"])
    full_root = Path(config["full_checkpoint_root"])
    historical = Path(config["feature_scoring_run"])
    device = torch.device(args.device)
    torch.set_num_threads(1)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["labels_read"]:
        raise RuntimeError("training cache contains labels")
    records = json.loads(cache["records_json"])
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[row["signal"]].append(index)
    stats = json.loads(stats_path.read_text())[args.dataset]
    clip, mask = encoder(device)
    amr, _ = load_model(full_root / f"{args.dataset}.pt", device)
    amr.eval()
    for parameter in amr.parameters():
        parameter.requires_grad_(False)
    renderer = WindowRenderer()
    destination = output_root / args.dataset
    destination.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    for signal, indices in groups.items():
        if args.signal and signal != args.signal:
            continue
        target = destination / f"{signal}.npz"
        if target.exists() and not args.only_clean:
            continue
        clean_raw = cache["raw"][indices]
        clean_block = cache["visual"][indices]
        clean_final = final_features(clip, mask, renderer, clean_raw, stats[signal], device)
        clean_score, clean_path = b3_score(clean_raw, clean_block, clean_final, amr, device)
        reference = historical / "point_scores" / args.dataset / signal / "B3_raw_numeric_l010.npz"
        clean_error = None
        historical_clean = None
        # NASA caches contain official Train windows; historical B3 files are
        # official Test scores for the same signal names and must never enter
        # training, even as an unlabeled anchor.
        same_stream = (records[indices[0]].get("source") ==
                       "complete_unlabeled_sequence")
        if same_stream and reference.is_file() and records[indices[0]]["window_index"] == 0:
            with np.load(reference, allow_pickle=False) as stored:
                historical_clean = stored["scores"][:len(clean_score)].astype(np.float64)
                clean_error = float(np.max(np.abs(clean_score - historical_clean)))
        if args.only_clean:
            if args.debug_clean_output is not None:
                np.savez_compressed(args.debug_clean_output, reconstructed=clean_score,
                                    historical=historical_clean)
            print(json.dumps({"dataset": args.dataset, "signal": signal,
                              "clean_error": clean_error,
                              "elapsed": time.perf_counter() - start_time}), flush=True)
            continue
        synthetic_raw = cache["synthetic_raw"][indices]
        synthetic_block = cache["synthetic_visual"][indices]
        synthetic_final = final_features(clip, mask, renderer, synthetic_raw, stats[signal], device)
        synthetic_score, synthetic_path = b3_score(
            synthetic_raw, synthetic_block, synthetic_final, amr, device)
        # Preserve the immutable historical clean B3 exactly where it exists.
        # The paired synthetic change comes from the reconstructed frozen B3.
        if historical_clean is not None:
            synthetic_score += historical_clean - clean_score
            clean_score = historical_clean
        if not np.isfinite(clean_score).all() or not np.isfinite(synthetic_score).all():
            raise RuntimeError(f"nonfinite frozen B3 score: {args.dataset}/{signal}")
        patch_mask = cache["patch_mask"][indices].numpy().astype(bool)
        point_mask = np.zeros(len(clean_score), dtype=bool)
        for window, row in enumerate(patch_mask):
            for patch in np.flatnonzero(row):
                begin = window * 56 + int(patch) * 16
                point_mask[begin:begin + 16] = True
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, clean_score=clean_score,
            synthetic_score=synthetic_score, point_mask=point_mask,
            clean_reference_ids=clean_path.reference_ids.cpu().numpy(),
            clean_reference_columns=clean_path.reference_columns.round().long().cpu().numpy(),
            synthetic_reference_ids=synthetic_path.reference_ids.cpu().numpy(),
            synthetic_reference_columns=synthetic_path.reference_columns.round().long().cpu().numpy(),
            source_indices=np.asarray(indices, dtype=np.int32),
            source_split=np.asarray(records[indices[0]].get("source", "unknown")),
            historical_anchor_used=np.asarray(historical_clean is not None),
            clean_reproduction_error=np.nan if clean_error is None else clean_error)
        temporary.replace(target)
        print(json.dumps({"dataset": args.dataset, "signal": signal,
                          "windows": len(indices), "clean_error": clean_error,
                          "elapsed": time.perf_counter() - start_time}), flush=True)


if __name__ == "__main__":
    main()
