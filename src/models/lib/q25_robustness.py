"""Robustness of the q25 aggregation to its computation details: quantile
taken on the 14-row patch map (block-expanded) instead of the 224-row
bilinearly-upsampled map.  Everything else identical.

Output: data/runs/vit4ts_failure_audit/q25_robustness.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/models/lib"))
from models.model_utils import align_anomaly_vector, compute_detection_intervals
from evaluation.evaluate import evaluate_intervals
from failure_audit import ALPHAS, DATASETS, STRIDE, WINDOW, \
    load_index_and_labels, load_scores

OUT = ROOT / "data/runs/vit4ts_failure_audit"


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
    index, labels = load_index_and_labels()
    rows = []
    for dataset in DATASETS:
        for signal in index[dataset]:
            top, q25, ts, _ = load_scores(dataset, signal)
            truth = [(int(a), int(b)) for a, b in labels[signal]]
            path = OUT / "patchmaps" / dataset / (signal + ".npz")
            if not path.exists():
                continue
            with np.load(path) as cached:
                patches = cached["patches"]  # [n,14,14]
            if float(np.abs(patches).max()) > 10:
                # harmonic instability destroyed these patch maps; the q224
                # column still uses the stored (artifact-robust) q25 score
                rows.append({"dataset": dataset, "signal": signal,
                             "patch_skipped_artifact": 1})
                continue
            n = patches.shape[0]
            v14 = np.quantile(patches, 0.25, axis=1)  # over rows -> [n,14]
            expanded = np.repeat(v14, WINDOW // 14, axis=1)  # [n,224]
            q25_patch = align_anomaly_vector(
                overlap_mean(expanded, STRIDE), len(top), WINDOW, STRIDE, n)
            entry = {"dataset": dataset, "signal": signal}
            for name, score in (("top", top), ("q224", q25),
                                ("q14_patch", q25_patch)):
                for a in ALPHAS:
                    intervals, _, _ = compute_detection_intervals(score, alpha=a)
                    predicted = [[int(ts[i]), int(ts[j])] for i, j in intervals]
                    entry[f"f1_{name}_alpha_{a}"] = evaluate_intervals(
                        truth, predicted)["F1"]
            rows.append(entry)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q25_robustness.csv", index=False)
    summary = {}
    for name in ("top", "q224", "q14_patch"):
        summary[name] = {}
        for dataset in DATASETS:
            sub = df[df.dataset == dataset]
            means = {a: sub[f"f1_{name}_alpha_{a}"].mean() for a in ALPHAS}
            summary[name][dataset] = round(max(means.values()), 4)
        summary[name]["macro"] = round(np.mean(
            list(summary[name].values())), 4)
    (OUT / "q25_robustness_summary.json").write_text(
        __import__("json").dumps(summary, indent=2) + "\n")
    print(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
