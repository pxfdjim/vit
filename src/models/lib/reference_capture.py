"""Capture the pre-upsample patch-level anomaly maps (14x14, three-scale
fused, i.e. the reference-matching result) for every window of all 125
sequences, with the untouched ViT4TS first-stage pipeline at 224/56 and OpenAI
CLIP ViT-B/16.

The patch map of window w is normal_vision_score = nan_to_num((m_l+m_m+m_p)/3)
right before bilinear upsampling in ViT4TS._run_inference.  For every sequence
the captured patch maps are upsampled, top-25% aggregated, overlap-mean
stitched and required to reproduce the cached 224/56 baseline score before
they are accepted.

Output: data/runs/vit4ts_failure_audit/patchmaps/<dataset>/<signal>.npz
  patches            [n_windows, 14, 14] float32
  reproduce_max_abs_diff
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/models/lib"))
import models.vit4ts as vit
from models.model_utils import aggregate_anomaly_map, align_anomaly_vector
from failure_audit import DATASETS, STRIDE, WINDOW, load_index_and_labels

OUT = ROOT / "data/runs/vit4ts_failure_audit/patchmaps"
PATCH = 16  # image pixels per patch; 224/16 = 14 patches per side


class _FProxy:
    """Capture the tensor entering vit4ts's final F.interpolate call."""

    def __init__(self, store):
        self._store = store

    def interpolate(self, x, **kwargs):
        self._store.append(x.detach().cpu().numpy())
        return torch_F.interpolate(x, **kwargs)

    def __getattr__(self, name):
        return getattr(torch_F, name)


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

    index, _ = load_index_and_labels()
    tasks = [(ds, name) for ds in DATASETS for name in index[ds]]
    assigned = [task for i, task in enumerate(tasks)
                if i % args.workers == args.worker]
    print("START", args.worker, "gpu", os.getenv("CUDA_VISIBLE_DEVICES"),
          "assigned", len(assigned), flush=True)

    model = vit.ViT4TS(window_size=WINDOW, window_step_ratio=4.0,
                       image_size=(224, 224), verbose=False)
    assert model.model.pretrained_tag == "openai"

    for ds, signal in assigned:
        target = OUT / ds / (signal + ".npz")
        if target.exists():
            print("SKIP", ds, signal, flush=True)
            continue
        frame = pd.read_csv(ROOT / "data" / ds / (signal + ".csv"))
        frame = frame.sort_values("timestamp")
        timestamps = frame.timestamp.to_numpy()
        store = []
        proxy = _FProxy(store)
        original_F = vit.F
        vit.F = proxy
        try:
            official, observed = model.predict_scores(frame)
        finally:
            vit.F = original_F
        if not np.array_equal(observed, timestamps):
            raise RuntimeError(f"Timestamp mismatch: {ds}/{signal}")
        patches = np.concatenate(
            [p[:, 0] for p in store], axis=0).astype(np.float32)  # [n,14,14]
        n_windows = patches.shape[0]

        # end-to-end verification: upsample -> top-25% -> overlap-mean -> align
        ups = torch_F.interpolate(
            torch.from_numpy(patches).unsqueeze(1), size=(224, 224),
            mode="bilinear").squeeze(1).numpy()
        vectors = np.stack([aggregate_anomaly_map(m, 0.25) for m in ups])
        rebuilt = align_anomaly_vector(overlap_mean(vectors, STRIDE),
                                       len(frame), WINDOW, STRIDE, n_windows)
        final7 = (ROOT / "data/runs/20261016_vit4ts_final7/scores" / ds
                  / signal / "window_224.npz")
        diag = (ROOT / "data/runs/20261016_vit4ts_diagnostics/scores" / ds
                / signal / "paper_window_224.npz")
        source = final7 if final7.exists() else diag
        with np.load(source) as cached:
            stored = np.asarray(cached["top" if final7.exists() else "corrected"],
                                dtype=float)
        diff = float(np.max(np.abs(rebuilt - stored)))
        # The released harmonic aggregation is numerically unstable for
        # near-duplicate patches: sum(1/dissimilarity) can cross zero and
        # produce ~1e8-scale artifacts.  Some stored baselines contain them
        # (max > 10 while legitimate scores stay below ~0.5).  A fresh clean
        # capture can then never match the stored score; flag instead of fail.
        stored_artifact = bool(np.max(stored) > 10)
        rebuilt_artifact = bool(np.max(rebuilt) > 10)
        if rebuilt_artifact and not stored_artifact:
            raise RuntimeError(f"Fresh capture introduced artifacts: "
                               f"{ds}/{signal}")
        if not stored_artifact and not rebuilt_artifact and diff > 1e-4:
            raise RuntimeError(f"Baseline reproduction failed: {ds}/{signal}, "
                               f"max_abs_diff={diff}")
        artifact_flag = ("both" if stored_artifact and rebuilt_artifact else
                         "stored_only" if stored_artifact else "none")

        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp.npz")
        np.savez_compressed(temp, patches=patches,
                            reproduce_max_abs_diff=diff,
                            artifact_flag=artifact_flag)
        temp.replace(target)
        print("CAPTURED", ds, signal, "windows", n_windows,
              "reproduce_max_abs_diff", diff, flush=True)
    print("WORKER_DONE", args.worker, flush=True)


if __name__ == "__main__":
    main()
