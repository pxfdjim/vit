"""Independent recomputation of the q25-vs-top25 F1 comparison from cached
scores, strictly under the paper F1-max protocol, plus fixed-alpha contrasts.

Protocol notes (arXiv 2506.06836 v2, verbatim):
  - "We then collapse M to a 1-D score s by taking the q-th quantile across
    all rows: s(t)=quantile_q(M_{:,t})."          -> per-column (per time step)
    quantile over the row/patch axis; the released implementation does the 2-D
    overlap average first, matching "average overlapping contributions to form
    a 2-D anomaly map" and then collapses.
  - Appendix A: EWMA smoothing, then Gaussian thresholds
    tau_alpha = mu + z_alpha*sigma, alpha in {0.10, 0.01, 0.001}, and
    "report ... the maximum F1 (F1-max) over these z_alpha values".
  - Table 1 caption: "Each entry reports the maximum F1 score across all
    evaluated thresholds."  The paper does not spell out the per-signal vs
    per-dataset aggregation, so both are reported; the primary table follows
    the fixed-alpha dataset-mean convention established in the earlier audits.

Outputs: data/runs/vit4ts_failure_audit/q25_verification.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/models/lib"))
from models.model_utils import compute_detection_intervals
from evaluation.evaluate import evaluate_intervals, window_wise_metrics
from failure_audit import ALPHAS, DATASETS, load_index_and_labels, load_scores

OUT = ROOT / "data/runs/vit4ts_failure_audit"


def f1_of(score, ts, truth, alpha):
    intervals, _, _ = compute_detection_intervals(score, alpha=alpha)
    predicted = [[int(ts[a]), int(ts[b])] for a, b in intervals]
    metric = evaluate_intervals(truth, predicted)
    counts = window_wise_metrics(truth, predicted)
    return metric["F1"], counts


def main():
    index, labels = load_index_and_labels()
    rows = []
    for dataset in DATASETS:
        for signal in index[dataset]:
            top, q25, ts, source = load_scores(dataset, signal)
            truth = [(int(a), int(b)) for a, b in labels[signal]]
            for method, score in (("top", top), ("q25", q25)):
                entry = {"dataset": dataset, "signal": signal,
                         "method": method, "score_source": source}
                for alpha in ALPHAS:
                    f1, counts = f1_of(score, ts, truth, alpha)
                    entry[f"f1_alpha_{alpha}"] = f1
                    entry[f"tp_alpha_{alpha}"] = counts["TP"]
                    entry[f"fp_alpha_{alpha}"] = counts["FP"]
                    entry[f"fn_alpha_{alpha}"] = counts["FN"]
                # per-signal best alpha (author-repo convention, secondary)
                entry["f1max_per_signal"] = max(
                    entry[f"f1_alpha_{a}"] for a in ALPHAS)
                rows.append(entry)
    per_signal = pd.DataFrame(rows)
    per_signal.to_csv(OUT / "q25_verification.csv", index=False)

    table = {}
    for dataset in DATASETS:
        sub = per_signal[per_signal.dataset == dataset]
        table[dataset] = {}
        for method in ("top", "q25"):
            m = sub[sub.method == method]
            means = {a: m[f"f1_alpha_{a}"].mean() for a in ALPHAS}
            best_alpha = max(means, key=means.get)
            top_best = max(  # best alpha of the top method in this dataset
                {a: sub[(sub.method == "top")][f"f1_alpha_{a}"].mean()
                 for a in ALPHAS}, key=lambda a: sub[(sub.method == "top")][
                     f"f1_alpha_{a}"].mean())
            table[dataset][method] = {
                "best_alpha": best_alpha,
                "f1max_fixed_alpha": round(means[best_alpha], 4),
                **{f"mean_f1_alpha_{a}": round(v, 4) for a, v in means.items()},
                "f1max_per_signal": round(m.f1max_per_signal.mean(), 4),
                "f1_at_top_best_alpha": round(
                    m[f"f1_alpha_{top_best}"].mean(), 4)}
    summary = {"protocol": {
        "paper_quote": "s(t)=quantile_q(M_{:,t}); q=0.25; EWMA then "
                       "tau=mu+z_alpha*sigma; F1-max over alpha in "
                       "{0.1,0.01,0.001}",
        "primary_convention": "fixed alpha, mean signal F1 per dataset, "
                              "max over alpha",
        "secondary_convention": "per-signal best-alpha F1, then mean"},
        "per_dataset": table,
        "macro_fixed_alpha": {m: round(np.mean(
            [table[d][m]["f1max_fixed_alpha"] for d in DATASETS]), 4)
            for m in ("top", "q25")},
        "macro_per_signal": {m: round(np.mean(
            [table[d][m]["f1max_per_signal"] for d in DATASETS]), 4)
            for m in ("top", "q25")}}
    (OUT / "q25_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    # cross-check against the stored final7 comparison table
    stored = pd.read_csv(ROOT / "data/runs/20261016_vit4ts_final7/comparison_7x4.csv")
    mismatches = []
    for dataset in DATASETS:
        for method, config in (("top", "top_224"), ("q25", "q25_224")):
            a = table[dataset][method]["f1max_fixed_alpha"]
            b = float(stored[(stored.dataset == dataset)
                             & (stored.config == config)]
                      .f1max_fixed_alpha.iloc[0])
            if abs(a - round(b, 4)) > 1e-3:
                mismatches.append((dataset, method, a, b))
    summary["stored_comparison_mismatches"] = mismatches
    (OUT / "q25_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
