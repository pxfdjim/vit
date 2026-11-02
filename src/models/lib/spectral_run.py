"""Run the SpectralViT4TS stage-1 model on the benchmark sequences with the
unchanged evaluation protocol (EWMA + Gaussian alpha thresholds, contextual
F1-max over alpha in {0.1, 0.01, 0.001}, paper aggregation convention).

Outputs under data/runs/spectral_stage1/:
  scores/<dataset>/<signal>.npz   per-point anomaly scores
  results/<dataset>/<signal>.json per-signal metrics and predicted intervals
  metrics_<fuse>.csv              per-dataset summary (filled at the end)
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
from models.model_utils import compute_detection_intervals
from evaluation.evaluate import evaluate_intervals, window_wise_metrics
from failure_audit import load_index_and_labels

OUT = ROOT / "data/runs/spectral_stage1"
ALPHAS = (0.1, 0.01, 0.001)


def evaluate_protocol(score, ts, truth):
    rows = []
    for alpha in ALPHAS:
        intervals, threshold, _ = compute_detection_intervals(score, alpha=alpha)
        predicted = [[int(ts[a]), int(ts[b])] for a, b in intervals]
        counts = window_wise_metrics(truth, predicted)
        metric = evaluate_intervals(truth, predicted)
        rows.append({"alpha": alpha, "threshold": threshold,
                     "predicted_intervals": predicted, **counts, **metric})
    return rows


def save_json(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(item, indent=2, default=float) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True,
                        help="comma-separated dataset names")
    parser.add_argument("--fuse", default="max", choices=["max", "mean", "max_norm", "mean_norm", "max_z", "mean_z", "residual", "residual_add", "residual_rank"])
    parser.add_argument("--bands", default="default",
                        help="'default' = (0,7),(7,28),(28,None); 'none' = "
                             "full band only (baseline validation)")
    parser.add_argument("--agg", default="top25", choices=["top25", "q25"])
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--masked-only", action="store_true")
    parser.add_argument("--matching", default="median", choices=["median", "knn", "hybrid"])
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if os.getenv("CLIP_PRETRAINED") != "openai":
        raise RuntimeError("Set CLIP_PRETRAINED=openai")
    bands = {'default': ((0, 7), (7, 28), (28, None)), 'none': (), 'low': ((0, 7),), 'mid': ((7, 28),), 'high': ((28, None),), 'lowhigh': ((0, 7), (28, None))}[args.bands]
    agg_tag = "" if args.agg == "top25" else "_" + args.agg
    if args.fuse == "residual":
        agg_tag = f"{agg_tag}_k{args.kappa:g}g{args.gain:g}"
        if args.masked_only:
            agg_tag += "m"
    if args.matching in ("knn", "hybrid"):
        tag = args.fuse if args.bands == "default" else ("full_only" if args.bands == "none" else f"{args.fuse}_{args.bands}")
        tag = f"{args.matching}k{args.knn_k}_" + tag + agg_tag
    else:
        tag = args.fuse if args.bands == "default" else ("full_only" if args.bands == "none" else f"{args.fuse}_{args.bands}")
        tag = tag + agg_tag
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    index, labels = load_index_and_labels()
    datasets = args.datasets.split(",")
    tasks = [(ds, s) for ds in datasets for s in index[ds]]
    assigned = [t for i, t in enumerate(tasks) if i % args.workers == args.worker]

    from models.spectral_vit4ts import SpectralViT4TS
    model = SpectralViT4TS(window_size=224, window_step_ratio=4.0,
                           image_size=(224, 224), batch_size=20,
                           bands=bands, fuse=args.fuse, kappa=args.kappa, gain=args.gain, masked_only=args.masked_only,
                             agg=args.agg, matching=args.matching, knn_k=args.knn_k, verbose=False)
    assert model.model.pretrained_tag == "openai"
    print("START worker", args.worker, "gpu",
          os.getenv("CUDA_VISIBLE_DEVICES"), "tasks", len(assigned),
          "fuse", args.fuse, "bands", bands, flush=True)

    for ds, signal in assigned:
        result_path = out / "results" / ds / (signal + ".json")
        if result_path.exists():
            print("SKIP", ds, signal, flush=True)
            continue
        t0 = time.monotonic()
        yahoo_dir = {"YAHOOA1": "Yahoo_S5/A1Benchmark",
                     "YAHOOA2": "Yahoo_S5/A2Benchmark",
                     "YAHOOA3": "Yahoo_S5/A3Benchmark",
                     "YAHOOA4": "Yahoo_S5/A4Benchmark"}
        data_dir = ROOT / "data" / yahoo_dir.get(ds, ds)
        frame = pd.read_csv(data_dir / (signal + ".csv"))
        if "timestamp" not in frame.columns and "timestamps" in frame.columns:
            frame = frame.rename(columns={"timestamps": "timestamp"})
        frame = frame.sort_values("timestamp")
        if ds.startswith("YAHOO"):
            # official evaluation (src/run_experiment.py) scores the Yahoo
            # benchmarks against one-based row positions, matching the
            # index-based labels in data/anomalies.csv
            frame["timestamp"] = np.arange(1, len(frame) + 1)
        scores, ts = model.predict_scores(frame)
        if not np.array_equal(ts, frame.timestamp.to_numpy()):
            raise RuntimeError(f"timestamp mismatch {ds}/{signal}")
        truth = [(int(a), int(b)) for a, b in labels[signal]]
        rows = evaluate_protocol(scores, ts, truth)
        score_path = out / "scores" / ds / (signal + ".npz")
        score_path.parent.mkdir(parents=True, exist_ok=True)
        temp = score_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, scores=scores, timestamps=ts)
        temp.replace(score_path)
        save_json(result_path, {"dataset": ds, "signal": signal,
                                "fuse": args.fuse, "agg": args.agg, "bands": [list(b) for b in bands],
                                "alpha_rows": rows})
        print("DONE", ds, signal, "f1",
              [round(r["F1"], 4) for r in rows],
              round(time.monotonic() - t0, 1), "s", flush=True)

    if args.worker == 0:
        # dataset summary (only after all workers finish is it complete; the
        # supervisor reruns this step)
        rows = []
        for ds in datasets:
            per_alpha = {a: [] for a in ALPHAS}
            for signal in index[ds]:
                p = out / "results" / ds / (signal + ".json")
                if not p.exists():
                    continue
                rec = json.loads(p.read_text())
                for r in rec["alpha_rows"]:
                    per_alpha[r["alpha"]].append(r["F1"])
            if not per_alpha[ALPHAS[0]]:
                continue
            means = {a: float(np.mean(v)) for a, v in per_alpha.items()}
            best = max(means, key=means.get)
            rows.append({"dataset": ds, "signals": len(per_alpha[ALPHAS[0]]),
                         "best_alpha": best,
                         "f1max_fixed_alpha": round(means[best], 4),
                         **{f"mean_f1_alpha_{a}": round(v, 4)
                            for a, v in means.items()}})
        pd.DataFrame(rows).to_csv(out / f"metrics_{tag}.csv", index=False)
        print("SUMMARY", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
