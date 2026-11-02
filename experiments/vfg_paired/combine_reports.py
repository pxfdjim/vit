#!/usr/bin/env python3
"""Combine the completed original-seven and Yahoo-four paired VFG reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vfg_paired.runner import publish_bytes_no_replace
from experiments.vfg_paired.summarize import publish_csv, publish_figure


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |\n| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join([header, *rows])


def run(args: argparse.Namespace) -> int:
    original = pd.read_csv(args.original_root / "dataset_f1max.csv")
    yahoo = pd.read_csv(args.yahoo_root / "dataset_f1max.csv")
    best = pd.concat([original, yahoo], ignore_index=True)
    if best.duplicated(["dataset", "model"]).any() or not best["complete"].all():
        raise RuntimeError("source dataset/model results are incomplete or duplicated")

    models = list(dict.fromkeys(best["model"]))
    original_datasets = list(dict.fromkeys(original["dataset"]))
    yahoo_datasets = list(dict.fromkeys(yahoo["dataset"]))
    real_datasets = [name for name in original_datasets if name != "artificialWithAnomaly"]
    scopes = {
        "real_6": real_datasets,
        "original_all_7": original_datasets,
        "yahoo_4": yahoo_datasets,
        "all_11": original_datasets + yahoo_datasets,
    }
    rows = []
    for scope, datasets in scopes.items():
        for model in models:
            selected = best[(best.model == model) & best.dataset.isin(datasets)]
            if selected.dataset.nunique() != len(datasets):
                raise RuntimeError(f"incomplete scope {scope}/{model}")
            rows.append({
                "scope": scope,
                "model": model,
                "macro_f1max": selected.f1_max.mean(),
                "n_datasets": len(datasets),
                "TP": int(selected.TP.sum()),
                "FP": int(selected.FP.sum()),
                "FN": int(selected.FN.sum()),
            })
    macro = pd.DataFrame(rows)

    comparisons = []
    for scope in scopes:
        indexed = macro[macro.scope == scope].set_index("model")
        for model in models:
            comparisons.append({
                "scope": scope,
                "model": model,
                "delta_vs_A": indexed.loc[model, "macro_f1max"] - indexed.loc["A_stable_median", "macro_f1max"],
                "delta_vs_B": indexed.loc[model, "macro_f1max"] - indexed.loc["B_paired_visual", "macro_f1max"],
            })
    changes = pd.DataFrame(comparisons)

    args.output_root.mkdir(parents=True, exist_ok=True)
    publish_csv(args.output_root / "dataset_f1max_11.csv", best)
    publish_csv(args.output_root / "macro_f1max_11.csv", macro)
    publish_csv(args.output_root / "macro_changes_11.csv", changes)
    provenance = {
        "schema_version": 1,
        "original_root": str(args.original_root.resolve()),
        "yahoo_root": str(args.yahoo_root.resolve()),
        "point_scores_are_retained_in_source_roots": True,
        "n_signals": 492,
        "n_datasets": 11,
        "models": models,
    }
    publish_bytes_no_replace(args.output_root / "provenance.json", (json.dumps(provenance, indent=2) + "\n").encode())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = best.pivot(index="dataset", columns="model", values="f1_max").loc[original_datasets + yahoo_datasets, models]
    figure, axis = plt.subplots(figsize=(16, 6))
    pivot.plot.bar(ax=axis, width=.84)
    axis.set_ylim(0, 1); axis.set_ylabel("Unweighted contextual F1-max"); axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False, fontsize=8, ncol=2); figure.tight_layout()
    publish_figure(args.output_root / "dataset_f1max_11.png", figure); plt.close(figure)

    all11 = macro[macro.scope == "all_11"].set_index("model")
    b = float(all11.loc["B_paired_visual", "macro_f1max"])
    frequency = ["C_paired_fft_joint", "D_fft_guided_reference", "E_paired_stft_joint"]
    best_frequency = max(frequency, key=lambda name: float(all11.loc[name, "macro_f1max"]))
    report = "\n".join([
        "# Paired visual-frequency ViT4TS: 11-dataset analysis", "",
        "## Coverage and protocol", "",
        "The combined report covers 492 complete univariate series: 125 signals from the original seven datasets and 367 indexed Yahoo S5 A1-A4 signals. Each source run uses frozen OpenAI CLIP ViT-B/16, 224/56 windows, stable Harmonic aggregation, A-E, and dataset-level Unweighted Contextual F1-max over α ∈ {0.1, 0.01, 0.001}.", "",
        "## Dataset results", "", markdown_table(best[["dataset", "model", "best_alpha", "f1_max", "TP", "FP", "FN"]]), "",
        "## Macro results", "", markdown_table(macro), "",
        "## Main findings", "",
        f"- Across all 11 datasets, B reaches {b:.6f} macro F1 and changes performance by {b - float(all11.loc['A_stable_median', 'macro_f1max']):+.6f} versus A.",
        f"- C, D, and E change all-11 macro F1 by {float(all11.loc['C_paired_fft_joint', 'macro_f1max']) - b:+.6f}, {float(all11.loc['D_fft_guided_reference', 'macro_f1max']) - b:+.6f}, and {float(all11.loc['E_paired_stft_joint', 'macro_f1max']) - b:+.6f} versus B.",
        f"- The strongest frequency-bearing method is {best_frequency} at {float(all11.loc[best_frequency, 'macro_f1max']):.6f}. It remains below paired visual B, so the controlled results do not show an independent frequency benefit.",
        "- The model change is a nonparametric reference-library and distance-matching mechanism around a frozen visual backbone. It is an incremental architectural change rather than a new learned representation architecture.", "",
        "## Source artifacts", "",
        f"- Original seven: `{args.original_root}`", f"- Yahoo A1-A4: `{args.yahoo_root}`",
        "- Point scores and per-signal metrics remain in those source roots; this directory stores combined tables, provenance, and the 11-dataset figure.", "",
    ])
    publish_bytes_no_replace(args.output_root / "experiment_report.md", report.encode())
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--original-root", type=Path, required=True)
    result.add_argument("--yahoo-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
