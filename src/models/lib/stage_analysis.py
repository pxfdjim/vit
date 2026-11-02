"""Stage-wise decomposition of real FN events and FP intervals, plus
consistency/persistence probes and two single-factor consistency-aware
stitching variants (min / median over covering windows).

All scores come from the cached 224/56 baseline and captured per-window
vectors (verified against the baseline); no new model inference.

Stages of the official 1-D recovery chain, measured at the dataset's best
fixed alpha against the official EWMA threshold:
  per-window : max over covering windows of the captured per-window top-25%
               vector inside the event range
  raw        : max of the overlap-mean stitched score before EWMA
  ewma       : max of the official EWMA score (this is what the threshold
               is compared against)

Outputs under data/runs/vit4ts_failure_audit:
  stage_decomposition.csv   one row per real event (TP/FN) and FP interval
  stitch_probes.csv         per dataset: F1 at each alpha for mean/min/median
  consistency_stats.json    aggregate comparison FN vs TP vs FP
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/models/lib"))
from models.model_utils import (align_anomaly_vector, compute_detection_intervals)
from evaluation.evaluate import evaluate_intervals, window_wise_metrics
from failure_audit import (ALPHAS, DATASETS, STRIDE, WINDOW,
                           load_index_and_labels, load_scores,
                           load_window_vectors)

OUT = ROOT / "data/runs/vit4ts_failure_audit"


def stitch(vectors, stride, how):
    n, width = vectors.shape
    total = stride * (n - 1) + width
    if how == "mean":
        sums = np.zeros(total)
        counts = np.zeros(total, dtype=int)
        for i, row in enumerate(vectors):
            start = i * stride
            sums[start:start + width] += row
            counts[start:start + width] += 1
        return sums / np.maximum(counts, 1)
    if how == "min":
        out = np.full(total, np.inf)
    elif how == "max":
        out = np.full(total, -np.inf)
    else:  # median: stack per position
        buckets = [[] for _ in range(total)]
        for i, row in enumerate(vectors):
            start = i * stride
            for j, v in enumerate(row):
                buckets[start + j].append(v)
        return np.array([np.median(b) if b else np.nan for b in buckets])
    for i, row in enumerate(vectors):
        start = i * stride
        out[start:start + width] = (np.minimum if how == "min"
                                    else np.maximum)(
            out[start:start + width], row)
    return out


def aligned(stitched, length, n_windows):
    return align_anomaly_vector(stitched, length, WINDOW, STRIDE, n_windows)


def ewma_of(scores):
    span = max(1, int(len(scores) * 0.01))
    return pd.Series(scores).ewm(span=span).mean().values


def main():
    index, labels = load_index_and_labels()
    best_alpha = json.loads((OUT / "audit_summary.json").read_text())["best_alpha"]
    cases = pd.read_csv(OUT / "failure_cases.csv")

    rows, probe_rows = [], []
    for dataset in DATASETS:
        for signal in index[dataset]:
            alpha = best_alpha[dataset]
            top, q25, ts, _ = load_scores(dataset, signal)
            truth = [(int(a), int(b)) for a, b in labels[signal]]
            vectors = load_window_vectors(dataset, signal)
            n_windows = len(vectors)
            T = len(top)
            mean_st = aligned(stitch(vectors, STRIDE, "mean"), T, n_windows)
            if float(np.max(np.abs(mean_st - top))) > 1e-9:
                raise RuntimeError(f"reproduction failed: {dataset}/{signal}")
            min_st = aligned(stitch(vectors, STRIDE, "min"), T, n_windows)
            med_st = aligned(stitch(vectors, STRIDE, "median"), T, n_windows)
            for how, score in (("mean", mean_st), ("min", min_st),
                               ("median", med_st)):
                for a in ALPHAS:
                    intervals, _, _ = compute_detection_intervals(score, alpha=a)
                    predicted = [[int(ts[i]), int(ts[j])] for i, j in intervals]
                    metric = evaluate_intervals(truth, predicted)
                    probe_rows.append({"dataset": dataset, "signal": signal,
                                       "stitch": how, "alpha": a,
                                       "f1": metric["F1"]})

            ewma = ewma_of(top)
            intervals0, thr, _ = compute_detection_intervals(top, alpha=alpha)
            raw_scores = top  # stored top is pre-EWMA
            # covering windows helper
            def perwindow_max(i0, i1):
                w_min = max(0, -(-(i0 - WINDOW + 1) // STRIDE))
                w_max = min(n_windows - 1, i1 // STRIDE)
                best = 0.0
                count = w_max - w_min + 1
                above = 0
                for w in range(w_min, w_max + 1):
                    j0 = max(i0 - w * STRIDE, 0)
                    j1 = min(i1 - w * STRIDE, WINDOW - 1)
                    if j1 < j0:
                        continue
                    v = float(vectors[w, j0:j1 + 1].max())
                    best = max(best, v)
                    if v > thr:
                        above += 1
                return best, count, above / max(count, 1)

            # anomaly mask for background stats
            mask = np.zeros(T, dtype=bool)
            for a, b in truth:
                mask[np.searchsorted(ts, a, "left"):
                     np.searchsorted(ts, b, "right")] = True
            p90 = np.quantile(top[~mask], 0.90) if (~mask).any() else np.nan

            def longest_run(seg, level):
                flag = seg > level
                best = cur = 0
                for f in flag:
                    cur = cur + 1 if f else 0
                    best = max(best, cur)
                return best

            ev = cases[(cases.dataset == dataset) & (cases.signal == signal)]
            for _, event in ev.iterrows():
                i0, i1 = int(event.i0), int(event.i1)
                pw, k, frac_above = perwindow_max(i0, i1)
                raw = float(raw_scores[i0:i1 + 1].max())
                ew = float(ewma[i0:i1 + 1].max())
                rows.append({
                    "dataset": dataset, "signal": signal,
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "duration_points": event.duration_points,
                    "covering_windows": k,
                    "windows_above_thr_frac": round(frac_above, 3),
                    "perwindow_over_thr": round(pw / thr, 4) if thr else None,
                    "raw_over_thr": round(raw / thr, 4) if thr else None,
                    "ewma_over_thr": round(ew / thr, 4) if thr else None,
                    "overlap_dilution": round(pw / raw, 4) if raw > 0 else None,
                    "ewma_dilution": round(raw / ew, 4) if ew > 0 else None,
                    "runlen_above_p90": longest_run(top[i0:i1 + 1], p90),
                    "runlen_ratio": round(
                        longest_run(top[i0:i1 + 1], p90)
                        / max(event.duration_points, 1), 3)})

    stage = pd.DataFrame(rows)
    stage.to_csv(OUT / "stage_decomposition.csv", index=False)
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(OUT / "stitch_probes.csv", index=False)

    def stats(frame, types, col):
        sub = frame[frame.event_type.isin(types)]
        return {"n": int(len(sub)), col: round(float(sub[col].median()), 4)
                if len(sub) else None}

    fn = stage[stage.event_type == "FN"]
    tp = stage[stage.event_type == "TP"]
    fp = stage[stage.event_type == "FP"]
    summary = {
        "stage_medians": {
            group: {col: round(float(sub[col].median()), 4)
                    for col in ("perwindow_over_thr", "raw_over_thr",
                                "ewma_over_thr", "overlap_dilution",
                                "ewma_dilution", "windows_above_thr_frac",
                                "runlen_ratio")}
            for group, sub in (("FN", fn), ("TP", tp), ("FP", fp))},
        "stage_survival": {
            group: {"perwindow_above_thr": int((sub.perwindow_over_thr > 1).sum()),
                    "raw_above_thr": int((sub.raw_over_thr > 1).sum()),
                    "ewma_above_thr": int((sub.ewma_over_thr > 1).sum()),
                    "n": int(len(sub))}
            for group, sub in (("FN", fn), ("TP", tp), ("FP", fp))},
        "stitch_probe_f1max": {
            how: {ds: round(float(gg[gg.dataset == ds]
                                  .groupby("alpha").f1.mean().max()), 4)
                  for ds in DATASETS}
            for how, gg in probes.groupby("stitch")},
    }
    (OUT / "consistency_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
