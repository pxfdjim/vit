"""Follow-up controlled experiments on the real failure cases of the fixed
ViT4TS first-stage baseline (224/56, OpenAI CLIP ViT-B/16).

Starting point: data/runs/vit4ts_failure_audit (failure_cases.csv etc.), where
62 real FN events and 348 real FP events were audited.  All experiments below
change exactly ONE factor of the official scoring/threshold pipeline and keep
everything else identical (window 224/stride 56, EWMA + Gaussian mu+z*sigma at
the dataset's best fixed alpha, paper F1-max protocol, no model change, no
synthetic injections, no VLM/LLM):

  base    official pipeline, reproduces the stored 224/56 baseline predictions
  no_ewma smoothing=False in compute_detection_intervals
  max_st  overlap-max stitching instead of overlap-mean (from the captured
          per-window top-25% vectors, verified to re-produce the baseline)
  mad_thr method='median' (median + z*MAD) instead of mean + z*std

Outputs under data/runs/vit4ts_failure_audit:
  event_followup.csv      per true-event: mode classification and per-experiment
                          detection outcomes at the best alpha
  experiment_metrics.csv  per dataset per experiment: TP/FN/FP and F1 at every
                          alpha, F1-max (paper protocol)
  separability.csv       per sequence score-distribution diagnostics
  followup_summary.json   aggregate numbers used by diagnostic_report.md
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
from failure_audit import (ALPHAS, DATASETS, DIAG, FINAL, STRIDE, WINDOW,
                           load_index_and_labels, load_scores,
                           load_window_vectors)

OUT = ROOT / "data/runs/vit4ts_failure_audit"


def stitch(vectors, stride, how="mean"):
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
    out = np.full(total, -np.inf)
    for i, row in enumerate(vectors):
        start = i * stride
        out[start:start + width] = np.maximum(out[start:start + width], row)
    return out


def aligned(stitched, length, n_windows):
    return align_anomaly_vector(stitched, length, WINDOW, STRIDE, n_windows)


def detect(score, alpha, smoothing=True, method="mean"):
    intervals, threshold, ewma = compute_detection_intervals(
        score, alpha=alpha, smoothing=smoothing, method=method)
    return intervals, threshold, ewma


def main():
    index, labels = load_index_and_labels()
    best_alpha = json.loads((OUT / "audit_summary.json").read_text())["best_alpha"]

    cases = pd.read_csv(OUT / "failure_cases.csv")
    true_cases = cases[cases.event_type.isin(["TP", "FN"])]

    event_rows, exp_rows, sep_rows = [], [], []
    for dataset in DATASETS:
        for signal in index[dataset]:
            alpha = best_alpha[dataset]
            top, q25, ts, source = load_scores(dataset, signal)
            truth = [(int(a), int(b)) for a, b in labels[signal]]
            vectors = load_window_vectors(dataset, signal)
            n_windows = len(vectors)
            # sanity: overlap-mean re-stitching must reproduce the baseline
            rebuilt = aligned(stitch(vectors, STRIDE, "mean"), len(top), n_windows)
            if float(np.max(np.abs(rebuilt - top))) > 1e-9:
                raise RuntimeError(f"reproduction failed: {dataset}/{signal}")
            max_score = aligned(stitch(vectors, STRIDE, "max"), len(top), n_windows)

            variants = {"base": (top, True, "mean"),
                        "no_ewma": (top, False, "mean"),
                        "max_st": (max_score, True, "mean"),
                        "mad_thr": (top, True, "median")}

            per_variant = {}
            for name, (score, smoothing, method) in variants.items():
                for a in ALPHAS:
                    intervals, threshold, ewma = detect(
                        score, a, smoothing=smoothing, method=method)
                    predicted = [[int(ts[i]), int(ts[j])] for i, j in intervals]
                    counts = window_wise_metrics(truth, predicted)
                    metric = evaluate_intervals(truth, predicted)
                    per_variant[(name, a)] = {
                        "predicted": predicted, "threshold": threshold,
                        "ewma": np.asarray(ewma, dtype=float), "counts": counts,
                        "f1": metric["F1"]}
                exp_rows.append({
                    "dataset": dataset, "signal": signal, "experiment": name,
                    **{f"f1_alpha_{a}": per_variant[(name, a)]["f1"] for a in ALPHAS}})

            # per true-event outcomes at the best alpha
            _, _, base_ewma = variants["base"][0], None, None
            intervals0, threshold0, ewma0 = detect(top, alpha)
            _, threshold_noe, _ = detect(top, alpha, smoothing=False)
            _, threshold_mad, _ = detect(top, alpha, method="median")
            _, threshold_max, _ = detect(max_score, alpha)
            ev = true_cases[(true_cases.dataset == dataset)
                            & (true_cases.signal == signal)]
            for _, event in ev.iterrows():
                i0, i1 = int(event.i0), int(event.i1)
                perwin = event.perwindow_score_max
                raw = float(top[i0:i1 + 1].max())
                ewma_seg = float(ewma0[i0:i1 + 1].max())
                ratios = {
                    "perwindow_over_thr": (perwin / threshold0
                                           if pd.notna(perwin) and threshold0 else None),
                    "raw_over_thr": (raw / threshold0 if threshold0 else None),
                    "ewma_over_thr": (ewma_seg / threshold0 if threshold0 else None),
                    "ewma_attenuation": (ewma_seg / raw if raw > 0 else None)}
                outcome = {}
                for name in variants:
                    predicted = per_variant[(name, alpha)]["predicted"]
                    outcome[f"detected_{name}"] = int(any(
                        not (pe < int(ts[event.i0]) or ps > int(ts[event.i1]))
                        for ps, pe in predicted))
                event_rows.append({
                    "dataset": dataset, "signal": signal,
                    "event_id": event.event_id, "event_type": event.event_type,
                    "duration_points": event.duration_points,
                    "mode": ("S" if pd.notna(perwin) and perwin > threshold0
                             else "W" if pd.notna(perwin) else None),
                    "raw_score_max": round(raw, 6),
                    "ewma_score_max": round(ewma_seg, 6),
                    **{k: (round(v, 4) if v is not None else None)
                       for k, v in ratios.items()},
                    "thr_base": round(float(threshold0), 6),
                    "thr_no_ewma": round(float(threshold_noe), 6),
                    "thr_mad": round(float(threshold_mad), 6),
                    "thr_max_st": round(float(threshold_max), 6),
                    **outcome})

            # separability diagnostics of the baseline score (EWMA'd)
            mask = np.zeros(len(top), dtype=bool)
            for a, b in truth:
                mask[np.searchsorted(ts, a, "left"):
                     np.searchsorted(ts, b, "right")] = True
            if mask.any() and (~mask).any():
                normal = ewma0[~mask]
                anomalous = ewma0[mask]
                p99 = np.quantile(normal, 0.99)
                sep_rows.append({
                    "dataset": dataset, "signal": signal, "alpha": alpha,
                    "anom_points": int(mask.sum()),
                    "anom_below_normal_p99_frac": round(
                        float(np.mean(anomalous < p99)), 4),
                    "normal_above_anom_max_frac": round(
                        float(np.mean(normal > anomalous.max())), 4),
                    "point_exceedance_rate": round(
                        float(np.mean(ewma0 > threshold0)), 4)})

    events = pd.DataFrame(event_rows)
    experiments = pd.DataFrame(exp_rows)
    separability = pd.DataFrame(sep_rows)
    events.to_csv(OUT / "event_followup.csv", index=False)
    experiments.to_csv(OUT / "experiment_metrics.csv", index=False)
    separability.to_csv(OUT / "separability.csv", index=False)

    # dataset-level F1-max per experiment (paper protocol: fixed-alpha mean
    # signal F1, max over alpha)
    f1max = {}
    for name in ("base", "no_ewma", "max_st", "mad_thr"):
        sub = experiments[experiments.experiment == name]
        f1max[name] = {}
        for dataset in DATASETS:
            ds = sub[sub.dataset == dataset]
            means = {a: ds[f"f1_alpha_{a}"].mean() for a in ALPHAS}
            f1max[name][dataset] = round(max(means.values()), 4)

    fn_events = events[events.event_type == "FN"]
    summary = {
        "protocol": {"single_factor_changes": {
            "no_ewma": "EWMA smoothing off, rest identical",
            "max_st": "overlap-max instead of overlap-mean stitching",
            "mad_thr": "median+z*MAD instead of mean+z*std threshold"},
            "alpha": "dataset best fixed alpha of the baseline F1-max"},
        "fn_modes": {
            "S_perwindow_above_threshold": int((fn_events["mode"] == "S").sum()),
            "S_lost_at_ewma": int(((fn_events["mode"] == "S")
                                   & (fn_events.raw_over_thr > 1)).sum()),
            "S_lost_at_overlap_mean": int(((fn_events["mode"] == "S")
                                          & (fn_events.raw_over_thr <= 1)).sum()),
            "W_perwindow_below_threshold": int((fn_events["mode"] == "W").sum())},
        "fn_recovered_at_best_alpha": {
            name: int(fn_events[f"detected_{name}"].sum()) for name in
            ("no_ewma", "max_st", "mad_thr")},
        "tp_events": int(len(events[events.event_type == "TP"])),
        "fn_events": int(len(fn_events)),
        "dataset_f1max": f1max,
        "separability_median": {
            "anom_below_normal_p99_frac": float(
                separability.anom_below_normal_p99_frac.median()),
            "normal_above_anom_max_frac": float(
                separability.normal_above_anom_max_frac.median()),
            "point_exceedance_rate_vs_alpha": {
                str(a): round(float(separability[separability.alpha == a]
                                    .point_exceedance_rate.median()), 4)
                for a in sorted(separability.alpha.unique())}},
    }
    (OUT / "followup_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
