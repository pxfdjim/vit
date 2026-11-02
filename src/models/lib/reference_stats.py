"""Reference-matching statistics for real TP/FN events and FP intervals.

For every span (true event or predicted FP interval at the dataset's best
alpha) we compare the patch-level reference-matching dissimilarity inside the
span against the same sequence's background patch dissimilarity (all patches
of windows whose column range does not intersect any labeled truth interval).

Inputs: cached patchmaps (reference_capture.py, verified against the stored
baseline), failure_cases.csv spans, cached per-window vectors.
Output: data/runs/vit4ts_failure_audit/reference_stats.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/models/lib"))
from failure_audit import DATASETS, STRIDE, WINDOW, load_index_and_labels, \
    load_window_vectors

OUT = ROOT / "data/runs/vit4ts_failure_audit"
PATCH = 16


def main():
    index, labels = load_index_and_labels()
    cases = pd.read_csv(OUT / "failure_cases.csv")
    rows = []
    for dataset in DATASETS:
        for signal in index[dataset]:
            path = OUT / "patchmaps" / dataset / (signal + ".npz")
            if not path.exists():
                continue
            with np.load(path) as cached:
                patches = cached["patches"]  # [n,14,14]
            if float(np.abs(patches).max()) > 10:
                continue  # patch maps contaminated by the harmonic
                          # instability (5 SMAP sequences); excluded here
            vectors = load_window_vectors(dataset, signal)
            n_windows, P = patches.shape[0], patches.shape[-1]
            truth = [(int(a), int(b)) for a, b in labels[signal]]
            ts = None
            # raw index spans from failure_cases (i0/i1 already raw indices)
            ev = cases[(cases.dataset == dataset) & (cases.signal == signal)]
            truth_mask = np.zeros(len(vectors) * 0 + 1, dtype=bool)  # placeholder
            T = WINDOW + (n_windows - 1) * STRIDE
            # background: windows whose range misses every truth interval;
            # approximate truth in raw indices via window overlap
            anomalous_window = np.zeros(n_windows, dtype=bool)
            for w in range(n_windows):
                lo, hi = w * STRIDE, w * STRIDE + WINDOW
                if any(not (hi <= a or lo > b) for a, b in truth):
                    anomalous_window[w] = True
            background = patches[~anomalous_window].ravel()
            if background.size < 100:
                background = patches.ravel()
            bg_p50, bg_p95 = np.quantile(background, [0.50, 0.95])

            for _, event in ev.iterrows():
                i0, i1 = int(event.i0), int(event.i1)
                w_min = max(0, -(-(i0 - WINDOW + 1) // STRIDE))
                w_max = min(n_windows - 1, i1 // STRIDE)
                vals, k = [], 0
                for w in range(w_min, w_max + 1):
                    j0 = max(i0 - w * STRIDE, 0)
                    j1 = min(i1 - w * STRIDE, WINDOW - 1)
                    if j1 < j0:
                        continue
                    k += 1
                    p0, p1 = j0 // PATCH, min(j1 // PATCH, P - 1)
                    vals.append(patches[w, :, p0:p1 + 1].ravel())
                if not vals:
                    continue
                vals = np.concatenate(vals)
                rows.append({
                    "dataset": dataset, "signal": signal,
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "covering_windows": k,
                    "patch_mean": round(float(vals.mean()), 6),
                    "patch_p90": round(float(np.quantile(vals, 0.9)), 6),
                    "bg_p50": round(float(bg_p50), 6),
                    "bg_p95": round(float(bg_p95), 6),
                    "mean_over_bg_p50": round(float(vals.mean() / bg_p50), 4),
                    "frac_above_bg_p95": round(
                        float(np.mean(vals > bg_p95)), 4)})
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "reference_stats.csv", index=False)
    summary = {}
    for group, sub in stats.groupby("event_type"):
        summary[group] = {
            "n": int(len(sub)),
            "mean_over_bg_p50_median": round(sub.mean_over_bg_p50.median(), 4),
            "mean_over_bg_p50_p25": round(sub.mean_over_bg_p50.quantile(.25), 4),
            "mean_over_bg_p50_p75": round(sub.mean_over_bg_p50.quantile(.75), 4),
            "frac_above_bg_p95_median": round(sub.frac_above_bg_p95.median(), 4)}
    (OUT / "reference_stats_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
