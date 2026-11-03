#!/usr/bin/env python3
"""Extract frozen CLIP final/Block-6 features for NASA native splits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import (
    NASA, evaluation_frame, load_stats, normalization_start, signal_names,
    train_frame, visual_values,
)
from experiments.curve_aware.extract_visual_cache import render_images
from experiments.temporal_matching.runner import publish_json_no_replace
from src.models.curve_aware_clip import (
    FrozenCLIPBlock6Encoder, build_intermediate_payload,
    curve_coordinates_from_plot_transform, encode_block6, expected_window_count,
    input_digest, intermediate_cache_path, publish_torch_no_replace,
    validate_intermediate_cache, visual_windows,
)
from src.models.vfg_vit4ts import VFGViT4TSBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--final-cache-root", type=Path, required=True)
    parser.add_argument("--block-cache-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "evaluation"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    tasks = [(d, s) for d in ("MSL", "SMAP") for s in signal_names(args.data_root, d)]
    tasks = tasks[args.worker_index::args.num_workers]
    stats = load_stats(args.stats)
    config = {"model": {"window_size": 224, "step_size": 56, "patch_size": 16,
                         "image_size": [224, 224], "batch_size": args.batch_size}}
    # One verified OpenAI CLIP instance supplies both final and Block-6 tokens.
    os.environ["CLIP_PRETRAINED"] = "openai"
    final_backend = VFGViT4TSBackend(config, args.device, args.final_cache_root)
    block_encoder = FrozenCLIPBlock6Encoder(final_backend.visual_encoder.clip_encoder).to(args.device)
    records, failures = [], 0
    for ordinal, (dataset, signal) in enumerate(tasks, 1):
        started = time.perf_counter()
        try:
            frame = train_frame(args.data_root, dataset, signal) if args.split == "train" else evaluation_frame(args.data_root, dataset, signal)
            raw = frame.value.to_numpy(np.float64)
            start = normalization_start(dataset, stats[dataset][signal], args.split)
            visual = visual_values(raw, stats[dataset][signal], start)
            digest = input_digest(raw)
            context_dataset = f"{dataset}__train" if args.split == "train" else dataset
            context = {"dataset": context_dataset, "signal": signal}
            count = expected_window_count(len(raw))
            if args.split == "evaluation":
                _, _ = final_backend._load_or_extract_visual(visual, context, digest)
                final_path = final_backend._cache_path(context, digest)
            else:
                final_path = Path("training_split_has_no_final_score_cache")
            block_path = intermediate_cache_path(args.block_cache_root, context, digest)
            if block_path.is_file():
                cached = torch.load(block_path, map_location="cpu", weights_only=True)
                validate_intermediate_cache(cached, digest=digest, expected_windows=count, path=block_path)
                status = "validated_existing"
            else:
                windows = visual_windows(visual)
                images = render_images(visual)
                tokens = encode_block6(block_encoder, images, device=args.device, batch_size=args.batch_size)
                payload = build_intermediate_payload(
                    tokens, curve_coordinates_from_plot_transform(windows),
                    digest=digest, source_final_cache=final_path,
                )
                publish_torch_no_replace(block_path, payload)
                cached = torch.load(block_path, map_location="cpu", weights_only=True)
                validate_intermediate_cache(cached, digest=digest, expected_windows=count, path=block_path)
                status = "published"
            records.append({"dataset": dataset, "signal": signal, "split": args.split,
                            "points": len(raw), "windows": count, "digest": digest,
                            "final_cache": str(final_path) if args.split == "evaluation" else None,
                            "block_cache": str(block_path), "status": status,
                            "elapsed_seconds": time.perf_counter() - started})
            print(f"[{ordinal}/{len(tasks)}] {args.split} {dataset}/{signal}: {status}", flush=True)
        except Exception as error:
            failures += 1
            records.append({"dataset": dataset, "signal": signal, "split": args.split,
                            "status": "failed", "error": f"{type(error).__name__}: {error}"})
            print(f"[{ordinal}/{len(tasks)}] failed {dataset}/{signal}: {error}", flush=True)
    manifest = {"status": "passed" if failures == 0 else "failed", "split": args.split,
                "worker_index": args.worker_index, "num_workers": args.num_workers,
                "records": records, "failures": failures}
    path = args.manifest_dir / f"{args.split}_worker_{args.worker_index:02d}_of_{args.num_workers:02d}.json"
    publish_json_no_replace(path, manifest)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
