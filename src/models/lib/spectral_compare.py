"""Aggregate spectral_stage1 results and compare with the stage-1 baseline and
the paper's stage-1 (ViT4TS) / stage-2 (VLM4TS) numbers over all 11 datasets.

Usage: python spectral_compare.py --tags residual_k2g2 full_only ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/runs/spectral_stage1"
DATASETS = ["artificialWithAnomaly", "realAWSCloudwatch", "realAdExchange",
            "realTraffic", "realTweets", "MSL", "SMAP",
            "YAHOOA1", "YAHOOA2", "YAHOOA3", "YAHOOA4"]
SHORT = {"artificialWithAnomaly": "Art", "realAWSCloudwatch": "AWS",
         "realAdExchange": "AdEx", "realTraffic": "Traffic",
         "realTweets": "Tweets", "MSL": "MSL", "SMAP": "SMAP",
         "YAHOOA1": "A1", "YAHOOA2": "A2", "YAHOOA3": "A3",
         "YAHOOA4": "A4"}
# paper Table 1 (arXiv 2506.06836v2)
PAPER_STAGE2 = {"artificialWithAnomaly": 0.714, "realAWSCloudwatch": 0.488,
                "realAdExchange": 0.727, "realTraffic": 0.632,
                "realTweets": 0.686, "MSL": 0.619, "SMAP": 0.773,
                "YAHOOA1": 0.733, "YAHOOA2": 0.901, "YAHOOA3": 0.497,
                "YAHOOA4": 0.474}
PAPER_STAGE1 = {"artificialWithAnomaly": 0.545, "realAWSCloudwatch": 0.400,
                "realAdExchange": 0.615, "realTraffic": 0.615,
                "realTweets": 0.597, "MSL": 0.543, "SMAP": 0.726,
                "YAHOOA1": 0.614, "YAHOOA2": 0.892, "YAHOOA3": 0.614,
                "YAHOOA4": 0.565}


def dataset_f1(results_dir, dataset):
    per_alpha = {a: [] for a in (0.1, 0.01, 0.001)}
    n = 0
    for p in sorted(results_dir.glob(dataset + "/*.json")):
        rec = json.loads(p.read_text())
        n += 1
        for r in rec["alpha_rows"]:
            per_alpha[r["alpha"]].append(r["F1"])
    if n == 0:
        return None
    means = {a: float(np.mean(v)) for a, v in per_alpha.items()}
    best = max(means, key=means.get)
    return {"signals": n, "best_alpha": best,
            "f1max": round(means[best], 4), **{f"f1_{a}": round(v, 4)
                                               for a, v in means.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--full-only-tag", default="full_only",
                        help="tag holding the 224/56 single-branch baseline")
    args = parser.parse_args()
    rows = []
    for tag in args.tags:
        d = OUT / tag / "results"
        for ds in DATASETS:
            r = dataset_f1(d, ds)
            if r is None:
                continue
            rows.append({"tag": tag, "dataset": ds, **r})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="dataset", columns="tag", values="f1max")
    table = piv.copy()
    table["paper_stage1"] = [PAPER_STAGE1[d] for d in table.index]
    table["paper_stage2"] = [PAPER_STAGE2[d] for d in table.index]
    if "full_only_q25" in table.columns:
        qcol = table["full_only_q25"]
        merged = [Q25_ONLY.get(d) if Q25_ONLY.get(d) is not None
                  else qcol[d] for d in table.index]
        table.insert(1, "q25_only", merged)
    if args.full_only_tag in table.columns:
        table = table[sorted(table.columns, key=lambda c: (
            c != args.full_only_tag, c))]
    print(table.to_string())
    print("\nmacro (only over rows complete in every column):")
    complete = table.dropna()
    print(f"  rows included: {len(complete)}/11")
    for col in table.columns:
        print(f"  {col:22s} {complete[col].mean():.4f}")
    out = OUT / "comparison.csv"
    table.to_csv(out)
    print("\nwritten", out)


if __name__ == "__main__":
    main()
