"""Capture the per-window top-25% anomaly vectors for the four datasets whose
window-level cache is missing (artificialWithAnomaly, realTraffic, realTweets,
MSL), using the untouched ViT4TS first-stage pipeline at window=224/stride=56.

This is a re-run of the original baseline inference, nothing else: the model,
weights (OpenAI CLIP ViT-B/16), preprocessing and window protocol are exactly
the stored 224/56 baseline.  For every sequence the captured per-window
vectors are re-stitched with the official overlap-mean and required to
reproduce the cached 224/56 baseline score before they are accepted.

Outputs under data/runs/vit4ts_failure_audit/windows/<dataset>/<signal>.npz
  vectors   [n_windows, 224] per-window 1-D top-25%-mean anomaly vectors
  timestamps
  reproduce_max_abs_diff   max |re-stitched score - cached baseline score|
"""
import argparse
import ast
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/models/lib"))
from models.model_utils import aggregate_anomaly_map, align_anomaly_vector
import models.vit4ts as vit
from failure_audit import DATASETS, WINDOW, STRIDE, load_index_and_labels

CAPTURE_DATASETS = ("artificialWithAnomaly", "realTraffic", "realTweets", "MSL")
OUT = ROOT / "data/runs/vit4ts_failure_audit/windows"


def overlap_mean(vectors, stride):
    n, width = vectors.shape
    sums = np.zeros(stride * (n - 1) + width)
    counts = np.zeros(len(sums), dtype=int)
    for i, row in enumerate(vectors):
        start = i * stride
        sums[start:start + width] += row
        counts[start:start + width] += 1
    return sums / np.maximum(counts, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if os.getenv("CLIP_PRETRAINED") != "openai":
        raise RuntimeError("Set CLIP_PRETRAINED=openai")

    index, labels = load_index_and_labels()
    tasks = [(ds, name) for ds in CAPTURE_DATASETS for name in index[ds]]
    assigned = [task for i, task in enumerate(tasks) if i % args.workers == args.worker]
    print("START", args.worker, "gpu", os.getenv("CUDA_VISIBLE_DEVICES"),
          "assigned", len(assigned), flush=True)

    model = vit.ViT4TS(window_size=WINDOW, window_step_ratio=4.0,
                       image_size=(224, 224), verbose=False)
    assert model.model.pretrained_tag == "openai"
    original_stitch = vit.stitch_anomaly_maps

    for ds, signal in assigned:
        target = OUT / ds / (signal + ".npz")
        if target.exists():
            print("SKIP", ds, signal, flush=True)
            continue
        frame = pd.read_csv(ROOT / "data" / ds / (signal + ".csv"))
        frame = frame.sort_values("timestamp")
        timestamps = frame.timestamp.to_numpy()
        capture = {}

        def stitch(maps, ratio, percent):
            capture["vectors"] = np.stack(
                [aggregate_anomaly_map(m, percent) for m in maps])
            return original_stitch(maps, ratio, percent)

        vit.stitch_anomaly_maps = stitch
        try:
            official, observed = model.predict_scores(frame)
        finally:
            vit.stitch_anomaly_maps = original_stitch
        if not np.array_equal(observed, timestamps):
            raise RuntimeError(f"Timestamp mismatch: {ds}/{signal}")

        vectors = capture["vectors"]
        n_windows = len(vectors)
        stitched = overlap_mean(vectors, STRIDE)
        rebuilt = align_anomaly_vector(stitched, len(frame), WINDOW,
                                       STRIDE, n_windows)
        # compare with the stored 224/56 baseline score
        baseline = ROOT / "data/runs/20261016_vit4ts_final7/scores" / ds / signal / "window_224.npz"
        with np.load(baseline) as cached:
            stored = np.asarray(cached["top"], dtype=float)
            cached_ts = cached["timestamps"]
        if not np.array_equal(timestamps, cached_ts):
            raise RuntimeError(f"Cache timestamp mismatch: {ds}/{signal}")
        diff = float(np.max(np.abs(rebuilt - stored)))
        if diff > 1e-6:
            raise RuntimeError(f"Baseline reproduction failed: {ds}/{signal}, "
                               f"max_abs_diff={diff}")

        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp.npz")
        np.savez_compressed(temp, vectors=vectors, timestamps=timestamps,
                            reproduce_max_abs_diff=diff,
                            n_windows=n_windows)
        temp.replace(target)
        print("CAPTURED", ds, signal, "windows", n_windows,
              "reproduce_max_abs_diff", diff, flush=True)
    print("WORKER_DONE", args.worker, flush=True)


if __name__ == "__main__":
    main()
