#!/usr/bin/env python3
"""Build one all-window unlabeled AMR training cache per OrionBench dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import (
    NASA, STEP, WINDOW, load_stats, normalized, signal_names, train_frame,
)
from src.models.curve_aware_clip import (
    expected_window_count, input_digest, intermediate_cache_path,
    validate_intermediate_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--block-cache-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stats = load_stats(args.stats)
    raws, visuals, records = [], [], []
    for signal in signal_names(args.data_root, args.dataset):
        frame = train_frame(args.data_root, args.dataset, signal)
        values = frame.value.to_numpy(np.float64)
        standardized = torch.from_numpy(normalized(values, stats[args.dataset][signal], 0))
        raw_windows = standardized.unfold(0, WINDOW, STEP).contiguous()
        count = expected_window_count(len(values))
        if len(raw_windows) != count:
            raise RuntimeError(f"{args.dataset}/{signal}: numeric window count mismatch")
        context = {"dataset": f"{args.dataset}__train" if args.dataset in NASA else args.dataset,
                   "signal": signal}
        digest = input_digest(values)
        paths = [intermediate_cache_path(root, context, digest) for root in args.block_cache_root]
        path = next((item for item in paths if item.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"missing Block-6 cache: {paths}")
        cached = torch.load(path, map_location="cpu", weights_only=True)
        validate_intermediate_cache(cached, digest=digest, expected_windows=count, path=path)
        visual = cached["intermediate_patch"].half()
        if len(visual) != len(raw_windows):
            raise RuntimeError(f"{args.dataset}/{signal}: visual window count mismatch")
        raws.append(raw_windows); visuals.append(visual)
        records.extend({"dataset": args.dataset, "signal": signal, "window_index": i,
                        "source": "NASA_official_Train" if args.dataset in NASA else "complete_unlabeled_sequence"}
                       for i in range(count))
        print(args.dataset, signal, count, flush=True)
    payload = {"raw": torch.cat(raws), "visual": torch.cat(visuals),
               "records_json": json.dumps(records, separators=(",", ":")),
               "dataset": args.dataset, "signals": len(signal_names(args.data_root, args.dataset)),
               "windows": sum(len(x) for x in raws), "labels_read": False,
               "uses_all_windows": True, "window_size": WINDOW, "step_size": STEP}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output, payload["raw"].shape, payload["visual"].shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
