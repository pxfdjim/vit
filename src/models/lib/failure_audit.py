"""Event-level audit of real false positives and misses of the fixed ViT4TS
first-stage baseline on the full evaluation sequences.

Inputs are the cached 224/56 OpenAI-CLIP scores (no rerun of CLIP, no model
change, no synthetic injections).  For every ground-truth anomaly interval and
every predicted interval of each sequence we record detection status and
features describing amplitude, position, threshold margins, aggregation
variants and (where cached) per-window score attenuation.  The chosen alpha per
dataset follows the paper F1-max protocol: fixed alpha, mean signal F1, max
over {0.1, 0.01, 0.001}.

Outputs (under --output, default data/runs/vit4ts_failure_audit):
  failure_cases.csv     one row per event (true TP/FN and predicted FP)
  per_sequence.csv      sequence-level context (anomalous window fraction etc.)
  audit_summary.json    counts and aggregate statistics used in the report
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.evaluate import evaluate_intervals, window_wise_metrics
from models.model_utils import compute_detection_intervals
from preprocessing.preprocess import preprocess_time_series

DATASETS = ("artificialWithAnomaly", "realAWSCloudwatch", "realAdExchange",
            "realTraffic", "realTweets", "MSL", "SMAP")
ALPHAS = (0.1, 0.01, 0.001)
FINAL = ROOT / "data/runs/20261016_vit4ts_final7"
DIAG = ROOT / "data/runs/20261016_vit4ts_diagnostics"
WINDOW, STRIDE = 224, 56


def load_index_and_labels():
    with (ROOT / "data/datasets.csv").open(newline="") as source:
        index = {dataset: sorted(ast.literal_eval(names))
                 for dataset, names in csv.reader(source)}
    labels = {}
    with (ROOT / "data/anomalies.csv").open() as source:
        next(source)
        for line in source:
            name, intervals = line.strip().split(",", 1)
            labels[name] = ast.literal_eval(intervals.strip('"'))
    return index, labels


def load_scores(dataset, signal):
    """Return (top_224, q25_224, timestamps, source) for one sequence."""
    final = FINAL / "scores" / dataset / signal / "window_224.npz"
    if final.exists():
        with np.load(final) as cached:
            top = np.asarray(cached["top"], dtype=float)
            q25 = np.asarray(cached["q25"], dtype=float)
            ts = cached["timestamps"]
            if np.max(np.abs(top - cached["official"])) > 1e-9:
                raise RuntimeError(f"224 mappings differ: {dataset}/{signal}")
        return top, q25, ts, str(final.relative_to(ROOT))
    old = DIAG / "scores" / dataset / signal / "paper_window_224.npz"
    quant = DIAG / "quantile_diagnostic/scores" / dataset / (signal + ".npz")
    with np.load(old) as cached:
        top = np.asarray(cached["corrected"], dtype=float)
        ts = cached["timestamps"]
        if np.max(np.abs(top - cached["official"])) > 1e-9:
            raise RuntimeError(f"224 mappings differ: {dataset}/{signal}")
    with np.load(quant) as cached:
        q25 = np.asarray(cached["quantile"], dtype=float)
        if np.max(np.abs(np.asarray(cached["source"], dtype=float) - top)) > 1e-9:
            raise RuntimeError(f"q25 source mismatch: {dataset}/{signal}")
    return top, q25, ts, str(old.relative_to(ROOT))


def load_window_vectors(dataset, signal):
    path = DIAG / "cache" / dataset / signal / "paper_window_224.npz"
    if not path.exists():
        path = (ROOT / "data/runs/vit4ts_failure_audit/windows"
                / dataset / (signal + ".npz"))
    if not path.exists():
        return None
    with np.load(path) as cached:
        return np.asarray(cached["vectors"], dtype=float)


def robust_spread(values):
    center = float(np.median(values))
    return center, float(1.4826 * np.median(np.abs(values - center)))


def zscore(values, center, spread):
    if spread > 0:
        return float(np.max(np.abs(values - center)) / spread)
    return 0.0 if np.max(np.abs(values - center)) == 0 else float("inf")


def interval_indices(ts, start, end):
    i0 = int(np.searchsorted(ts, start, side="left"))
    i1 = int(np.searchsorted(ts, end, side="right")) - 1
    return i0, i1


def detection_at(scores, ts, alpha, truth):
    found, _, _ = compute_detection_intervals(scores, alpha=alpha)
    predicted = [[int(ts[a]), int(ts[b])] for a, b in found]
    counts = window_wise_metrics(truth, predicted)
    hit = {tuple(event): any(not (pe < event[0] or ps > event[1])
                             for ps, pe in predicted) for event in truth}
    return predicted, counts, hit


def event_features(dataset, signal, ts, raw, proc, i0, i1):
    """Shared context features of a raw-index interval [i0, i1]."""
    T = len(raw)
    inside = slice(i0, i1 + 1)
    dur = i1 - i0 + 1
    lo, hi = max(0, i0 - 2 * dur), min(T, i1 + 1 + 2 * dur)
    raw_context = np.concatenate([raw[lo:i0], raw[i1 + 1:hi]])
    proc_context = np.concatenate([proc[lo:i0], proc[i1 + 1:hi]])
    if len(raw_context) < 8:
        raw_context = raw
        proc_context = proc
    local_center, local_spread = robust_spread(raw_context)
    g_center, g_spread = robust_spread(raw)
    proc_center, proc_spread = robust_spread(proc_context)
    return {
        "start_ts": int(ts[i0]), "end_ts": int(ts[i1]),
        "i0": i0, "i1": i1, "duration_points": dur,
        "position_frac": round(i0 / max(T - 1, 1), 4),
        "robust_global_z": round(zscore(raw[inside], g_center, g_spread), 3),
        "robust_local_z": round(zscore(raw[inside], local_center, local_spread), 3),
        "processed_local_z": round(zscore(proc[inside], proc_center, proc_spread), 3),
        "processed_range_frac": round(float(proc[inside].max()
                                            - proc[inside].min()), 4),
        "contains_global_extreme": int(np.argmax(raw) in range(i0, i1 + 1)
                                       or np.argmin(raw) in range(i0, i1 + 1)),
    }


def sequence_audit(dataset, signal, signal_labels, rows, seq_rows):
    frame = pd.read_csv(ROOT / "data" / dataset / (signal + ".csv"))
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    ts = frame.timestamp.to_numpy()
    raw = frame.value.to_numpy(dtype=float)
    proc = preprocess_time_series(raw)
    top, q25, cached_ts, source = load_scores(dataset, signal)
    if not (np.array_equal(ts, cached_ts) and len(top) == len(raw)):
        raise RuntimeError(f"alignment mismatch: {dataset}/{signal}")
    truth = [(int(a), int(b)) for a, b in signal_labels]

    # reproduce stored baseline metrics for the same config
    stored = json.loads((FINAL / "results" / dataset / (signal + ".json")).read_text())
    stored_rows = {float(m["alpha"]): m for m in stored["configs"]["top_224"]["metrics"]}
    per_alpha_f1, predictions, ewma_cache = {}, {}, {}
    for alpha in ALPHAS:
        found, threshold, ewma = compute_detection_intervals(top, alpha=alpha)
        predicted = [[int(ts[a]), int(ts[b])] for a, b in found]
        metric = evaluate_intervals(truth, predicted)
        kept = stored_rows[alpha]
        if (abs(metric["F1"] - kept["f1"]) > 1e-9
                or [list(x) for x in predicted] != kept["predicted_intervals"]):
            raise RuntimeError(f"baseline mismatch: {dataset}/{signal}/{alpha}")
        per_alpha_f1[alpha] = metric["F1"]
        predictions[alpha] = predicted
        ewma_cache[alpha] = (np.asarray(ewma, dtype=float), float(threshold))

    q25_hit_rows = {}
    for alpha in ALPHAS:
        _, counts, hit = detection_at(q25, ts, alpha, truth)
        q25_hit_rows[alpha] = (counts, hit)

    vectors = load_window_vectors(dataset, signal)
    T = len(raw)
    n_windows = int((T - WINDOW) / STRIDE) + 1
    covered = WINDOW + (n_windows - 1) * STRIDE

    true_idx = [interval_indices(ts, a, b) for a, b in truth]
    anomalous_points = sum(i1 - i0 + 1 for i0, i1 in true_idx)
    anomalous_windows = 0
    if truth:
        anomaly_mask = np.zeros(T, dtype=bool)
        for i0, i1 in true_idx:
            anomaly_mask[i0:i1 + 1] = True
        for w in range(n_windows):
            if anomaly_mask[w * STRIDE:w * STRIDE + WINDOW].any():
                anomalous_windows += 1
    seq_rows.append({
        "dataset": dataset, "signal": signal, "T": T, "n_true_events": len(truth),
        "anomalous_point_frac": round(anomalous_points / T, 4),
        "anomalous_window_frac": round(anomalous_windows / n_windows, 4),
        "n_windows": n_windows, "covered_points": covered,
        "tail_points": max(T - covered, 0), "score_source": source,
        **{f"f1_alpha_{alpha}": round(per_alpha_f1[alpha], 4) for alpha in ALPHAS},
    })

    best_alpha = None  # filled by caller per dataset
    return {"dataset": dataset, "signal": signal, "ts": ts, "raw": raw,
            "proc": proc, "top": top, "q25": q25, "truth": truth,
            "true_idx": true_idx, "predictions": predictions,
            "ewma_cache": ewma_cache, "q25_hit_rows": q25_hit_rows,
            "vectors": vectors, "n_windows": n_windows, "covered": covered,
            "per_alpha_f1": per_alpha_f1}


def score_features(ctx, i0, i1, alpha, score_kind="top"):
    scores = ctx["top"] if score_kind == "top" else ctx["q25"]
    ewma, threshold = ctx["ewma_cache"][alpha] if score_kind == "top" else (
        None, None)
    if score_kind == "top":
        seg = ewma[i0:i1 + 1]
        rank = float(np.mean(ewma <= seg.max()))
        return {"score_max": round(float(seg.max()), 6),
                "score_mean": round(float(seg.mean()), 6),
                "threshold": round(threshold, 6),
                "margin_to_threshold": round(float(seg.max() - threshold), 6),
                "margin_rel": (round(float((seg.max() - threshold) / threshold), 4)
                               if threshold > 0 else None),
                "score_percentile": round(rank, 4)}
    seg = scores[i0:i1 + 1]
    # q25 threshold at the same alpha (unsmoothed protocol is identical: EWMA
    # is applied inside compute_detection_intervals; recompute here)
    found, threshold, ewma_q = compute_detection_intervals(scores, alpha=alpha)
    rank = float(np.mean(np.asarray(ewma_q) <= np.asarray(ewma_q)[i0:i1 + 1].max()))
    return {"q25_score_max": round(float(seg.max()), 6),
            "q25_threshold": round(float(threshold), 6),
            "q25_margin": round(float(np.asarray(ewma_q)[i0:i1 + 1].max()
                                      - threshold), 6),
            "q25_score_percentile": round(rank, 4)}


def window_attenuation(ctx, i0, i1):
    vectors = ctx["vectors"]
    if vectors is None:
        return {}
    n = len(vectors)
    w_min = max(0, -(-(i0 - WINDOW + 1) // STRIDE))
    w_max = min(n - 1, i1 // STRIDE)
    best, best_pos = 0.0, None
    for w in range(w_min, w_max + 1):
        j0 = max(i0 - w * STRIDE, 0)
        j1 = min(i1 - w * STRIDE, WINDOW - 1)
        if j1 < j0:
            continue
        col = vectors[w, j0:j1 + 1]
        k = int(np.argmax(col))
        if col[k] > best:
            best = float(col[k])
            best_pos = w * STRIDE + j0 + k
    final_max = float(ctx["top"][i0:i1 + 1].max())
    global_window_max = float(vectors.max())
    center = (i0 + i1) / 2
    return {"perwindow_score_max": round(best, 6),
            "final_score_max_raw": round(final_max, 6),
            "stitch_attenuation_ratio": (round(final_max / best, 4)
                                         if best > 0 else None),
            "ratio_to_global_window_max": (round(best / global_window_max, 4)
                                           if global_window_max > 0 else None),
            "best_window_peak_offset_points": (round(best_pos - center, 1)
                                               if best_pos is not None else None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/runs/vit4ts_failure_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    index, labels = load_index_and_labels()
    contexts = []
    seq_rows = []
    for dataset in DATASETS:
        for signal in index[dataset]:
            contexts.append(sequence_audit(dataset, signal, labels[signal],
                                           None, seq_rows))
    per_seq = pd.DataFrame(seq_rows)

    # paper F1-max protocol: fixed alpha, mean signal F1 per dataset, max over alpha
    best_alpha = {}
    for dataset in DATASETS:
        subset = per_seq[per_seq.dataset == dataset]
        means = {alpha: subset[f"f1_alpha_{alpha}"].mean() for alpha in ALPHAS}
        best_alpha[dataset] = max(means, key=means.get)

    rows = []
    for ctx in contexts:
        dataset, signal = ctx["dataset"], ctx["signal"]
        alpha = best_alpha[dataset]
        truth, predictions = ctx["truth"], ctx["predictions"][alpha]
        predicted = predictions
        # ---- true events ----
        for event_id, ((a, b), (i0, i1)) in enumerate(zip(truth, ctx["true_idx"])):
            detected = any(not (pe < a or ps > b) for ps, pe in predicted)
            matched = [interval_indices(ctx["ts"], ps, pe)
                       for ps, pe in predicted if not (pe < a or ps > b)]
            row = {"dataset": dataset, "signal": signal, "alpha": alpha,
                   "event_id": event_id, "event_type": "TP" if detected else "FN",
                   **event_features(dataset, signal, ctx["ts"], ctx["raw"],
                                    ctx["proc"], i0, i1),
                   **score_features(ctx, i0, i1, alpha)}
            for other in ALPHAS:
                row[f"detected_alpha_{other}"] = int(
                    any(not (pe < a or ps > b) for ps, pe in ctx["predictions"][other]))
            for other in ALPHAS:
                row[f"q25_detected_alpha_{other}"] = int(
                    ctx["q25_hit_rows"][other][1][tuple((a, b))])
            row.update({f"q25_{k}": v for k, v in
                        score_features(ctx, i0, i1, alpha, "q25").items()})
            row["q25_detected_same_alpha"] = int(
                ctx["q25_hit_rows"][alpha][1][tuple((a, b))])
            row.update(window_attenuation(ctx, i0, i1))
            if matched:
                cov = sum(min(i1, m[1]) - max(i0, m[0]) + 1 for m in matched)
                row["matched_coverage_frac"] = round(cov / (i1 - i0 + 1), 4)
                centers = [(m[0] + m[1]) / 2 for m in matched]
                row["center_offset_points"] = round(
                    min(centers, key=lambda c: abs(c - (i0 + i1) / 2))
                    - (i0 + i1) / 2, 1)
            rows.append(row)
        # ---- predicted intervals (FP part) ----
        for ps, pe in predicted:
            if any(not (pe < a or ps > b) for a, b in truth):
                continue
            i0, i1 = interval_indices(ctx["ts"], ps, pe)
            row = {"dataset": dataset, "signal": signal, "alpha": alpha,
                   "event_id": None, "event_type": "FP",
                   **event_features(dataset, signal, ctx["ts"], ctx["raw"],
                                    ctx["proc"], i0, i1),
                   **score_features(ctx, i0, i1, alpha)}
            dists = [min(abs(i0 - i1t), abs(i0t - i1)) if i1 < i0t or i1t < i0
                     else 0 for i0t, i1t in ctx["true_idx"]]
            row["distance_to_nearest_true_points"] = (
                round(min(dists), 1) if dists else None)
            row["in_extrapolated_tail"] = int(i0 >= ctx["covered"])
            row.update({f"q25_{k}": v for k, v in
                        score_features(ctx, i0, i1, alpha, "q25").items()})
            row["q25_also_fp"] = int(any(
                not (qe < ps or qs > pe)
                for qs, qe in [[int(ctx["ts"][ia]), int(ctx["ts"][ib])]
                               for ia, ib in compute_detection_intervals(
                                   ctx["q25"], alpha=alpha)[0]]))
            rows.append(row)

    events = pd.DataFrame(rows)
    events.to_csv(args.output / "failure_cases.csv", index=False)
    per_seq["best_alpha"] = per_seq.dataset.map(best_alpha)
    per_seq.to_csv(args.output / "per_sequence.csv", index=False)

    true_rows = events[events.event_type.isin(["TP", "FN"])]
    fn_rows = events[events.event_type == "FN"]
    tp_rows = events[events.event_type == "TP"]
    fp_rows = events[events.event_type == "FP"]

    def dist(frame, column):
        values = frame[column].dropna()
        return {"n": int(len(values)),
                "median": float(values.median()) if len(values) else None,
                "p25": float(values.quantile(0.25)) if len(values) else None,
                "p75": float(values.quantile(0.75)) if len(values) else None}

    summary = {
        "protocol": {"window": WINDOW, "stride": STRIDE, "image": [224, 224],
                     "clip": "OpenAI ViT-B/16", "aggregation": "top_25_percent_mean",
                     "alphas": list(ALPHAS), "f1max": "fixed-alpha mean signal F1, max over alpha",
                     "baseline_reproduction": "per-signal predictions equal final7 stored metrics"},
        "best_alpha": {k: v for k, v in best_alpha.items()},
        "f1max": {ds: round(float(per_seq[per_seq.dataset == ds][
            f"f1_alpha_{best_alpha[ds]}"].mean()), 4) for ds in DATASETS},
        "counts": {
            "sequences": int(len(per_seq)),
            "true_events": int(len(true_rows)),
            "TP": int(len(tp_rows)), "FN": int(len(fn_rows)),
            "FP": int(len(fp_rows)),
            "by_dataset": {ds: {
                "true": int((true_rows.dataset == ds).sum()),
                "TP": int(((tp_rows.dataset == ds)).sum()),
                "FN": int(((fn_rows.dataset == ds)).sum()),
                "FP": int(((fp_rows.dataset == ds)).sum())} for ds in DATASETS},
        },
        "fn_features": {
            "robust_local_z": dist(fn_rows, "robust_local_z"),
            "robust_global_z": dist(fn_rows, "robust_global_z"),
            "duration_points": dist(fn_rows, "duration_points"),
            "margin_rel": dist(fn_rows, "margin_rel"),
            "contains_global_extreme_frac": (
                float(fn_rows.contains_global_extreme.mean()) if len(fn_rows) else None),
            "detected_at_other_alpha": {
                str(a): int(fn_rows[f"detected_alpha_{a}"].sum())
                for a in ALPHAS},
            "recovered_by_q25_same_alpha": int(fn_rows["q25_detected_same_alpha"].sum())
                if len(fn_rows) else 0,
            "stitch_attenuation_ratio": dist(fn_rows, "stitch_attenuation_ratio"),
            "perwindow_vs_threshold": dist(
                fn_rows.assign(ratio=fn_rows.perwindow_score_max
                               / fn_rows.threshold.replace(0, np.nan)),
                "ratio"),
        },
        "tp_features": {
            "robust_local_z": dist(tp_rows, "robust_local_z"),
            "duration_points": dist(tp_rows, "duration_points"),
            "margin_rel": dist(tp_rows, "margin_rel"),
            "contains_global_extreme_frac": (
                float(tp_rows.contains_global_extreme.mean()) if len(tp_rows) else None),
        },
        "fp_features": {
            "duration_points": dist(fp_rows, "duration_points"),
            "in_extrapolated_tail_n": int(fp_rows.in_extrapolated_tail.sum())
                if len(fp_rows) else 0,
            "adjacent_to_true_within_56_n": int(
                (fp_rows.distance_to_nearest_true_points <= STRIDE).sum())
                if len(fp_rows) else 0,
            "distance_to_nearest_true_points": dist(
                fp_rows, "distance_to_nearest_true_points"),
            "contains_global_extreme_frac": (
                float(fp_rows.contains_global_extreme.mean()) if len(fp_rows) else None),
            "robust_local_z": dist(fp_rows, "robust_local_z"),
            "q25_also_fp_frac": (
                float(fp_rows.q25_also_fp.mean()) if len(fp_rows) else None),
        },
        "sequence_reference_context": {
            "anomalous_window_frac_of_fn_sequences": dist(
                per_seq.merge(fn_rows[["dataset", "signal"]].drop_duplicates(),
                              on=["dataset", "signal"], how="inner"),
                "anomalous_window_frac"),
            "anomalous_window_frac_of_all": dist(per_seq, "anomalous_window_frac"),
        },
    }
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"events": len(events), "FN": len(fn_rows), "FP": len(fp_rows),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
