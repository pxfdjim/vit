#!/usr/bin/env python3
"""Create controlled frequency comparisons and the final paired-run report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vfg_paired.runner import load_json, publish_bytes_no_replace
from experiments.vfg_paired.summarize import (
    event_changes,
    load_documents,
    publish_csv,
    publish_figure,
)


DEFAULT_CONFIG = HERE / "config.json"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def controlled_comparisons(best: pd.DataFrame, macro: pd.DataFrame, config: dict) -> pd.DataFrame:
    baseline = "B_paired_visual"
    frequency_models = [
        "C_paired_fft_joint",
        "D_fft_guided_reference",
        "E_paired_stft_joint",
    ]
    rows = []
    indexed = best.set_index(["dataset", "model"])
    for scope, datasets in {"real_6": config["real_datasets"], "all_7": config["datasets"]}.items():
        for model in frequency_models:
            left = macro[(macro.scope == scope) & (macro.model == model)].iloc[0]
            right = macro[(macro.scope == scope) & (macro.model == baseline)].iloc[0]
            rows.append({
                "scope": scope, "dataset": "", "model": model,
                "baseline": baseline, "delta_f1max": left.macro_f1max - right.macro_f1max,
                "TP_change": int(left.TP - right.TP), "FP_change": int(left.FP - right.FP),
                "FN_change": int(left.FN - right.FN),
            })
            for dataset in datasets:
                current = indexed.loc[(dataset, model)]
                reference = indexed.loc[(dataset, baseline)]
                rows.append({
                    "scope": "dataset", "dataset": dataset, "model": model,
                    "baseline": baseline, "delta_f1max": current.f1_max - reference.f1_max,
                    "TP_change": int(current.TP - reference.TP), "FP_change": int(current.FP - reference.FP),
                    "FN_change": int(current.FN - reference.FN),
                })
    return pd.DataFrame(rows)


def candidate_audit(documents: list[dict]) -> pd.DataFrame:
    rows = []
    for document in documents:
        if document["variant"] != "B_paired_visual":
            continue
        windows = int(document["backend_metadata"]["n_windows"])
        counts = [sum(abs(query - ref) >= 4 for ref in range(windows)) for query in range(windows)]
        selected = [min(32, count) for count in counts]
        rows.append({
            "dataset": document["dataset"], "signal": document["signal"], "n_windows": windows,
            "min_eligible_nonoverlap_windows": min(counts), "max_eligible_nonoverlap_windows": max(counts),
            "min_common_visual_pool": min(selected), "max_common_visual_pool": max(selected),
            "D_selected_windows": 5,
        })
    frame = pd.DataFrame(rows).sort_values(["dataset", "signal"])
    if len(frame) != 125 or frame.max_common_visual_pool.max() > 32 or frame.min_common_visual_pool.min() < 5:
        raise RuntimeError("candidate-pool audit failed")
    return frame


def comparison_event_summary(documents: list[dict], best: pd.DataFrame, config: dict) -> pd.DataFrame:
    local = dict(config)
    local["comparisons"] = [
        {"name": f"{letter}_minus_B", "left": model, "right": "B_paired_visual"}
        for letter, model in [
            ("C", "C_paired_fft_joint"),
            ("D", "D_fft_guided_reference"),
            ("E", "E_paired_stft_joint"),
        ]
    ]
    _, per_dataset = event_changes(documents, best, local)
    totals = []
    for comparison, group in per_dataset.groupby("comparison", sort=False):
        for scope, datasets in {"real_6": config["real_datasets"], "all_7": config["datasets"]}.items():
            part = group[group.dataset.isin(datasets)]
            totals.append({
                "scope": scope, "comparison": comparison,
                "FN_recovered": int(part.FN_recovered.sum()), "TP_lost": int(part.TP_lost.sum()),
                "net_event_recovery": int(part.net_event_recovery.sum()),
                "FP_change": int(part.FP_change.sum()),
            })
    return pd.DataFrame(totals)


def plots(best: pd.DataFrame, macro: pd.DataFrame, controlled: pd.DataFrame, root: Path, config: dict) -> None:
    order = config["variants_order"]
    pivot = best.pivot(index="dataset", columns="model", values="f1_max").loc[config["datasets"], order]
    figure, axis = plt.subplots(figsize=(13, 5.5))
    pivot.plot.bar(ax=axis, width=.82)
    axis.set_ylabel("Unweighted contextual F1-max")
    axis.set_ylim(0, 1); axis.grid(axis="y", alpha=.25); axis.legend(frameon=False, fontsize=8)
    figure.tight_layout(); publish_figure(root / "plots/dataset_f1max.png", figure); plt.close(figure)

    view = controlled[controlled.scope.isin(["real_6", "all_7"])].pivot(index="model", columns="scope", values="delta_f1max")
    figure, axis = plt.subplots(figsize=(8, 4.6))
    view.plot.bar(ax=axis, color=["#3B82F6", "#F59E0B"])
    axis.axhline(0, color="black", lw=.8); axis.set_ylabel("F1-max change versus paired visual B")
    axis.grid(axis="y", alpha=.25); axis.legend(frameon=False)
    figure.tight_layout(); publish_figure(root / "plots/frequency_vs_paired_visual.png", figure); plt.close(figure)


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    best = pd.read_csv(args.output_root / "dataset_f1max.csv")
    macro = pd.read_csv(args.output_root / "macro_f1max.csv")
    documents = load_documents(args.output_root)
    controlled = controlled_comparisons(best, macro, config)
    candidates = candidate_audit(documents)
    events = comparison_event_summary(documents, best, config)
    publish_csv(args.output_root / "controlled_frequency_vs_B.csv", controlled)
    publish_csv(args.output_root / "candidate_pool_audit.csv", candidates)
    publish_csv(args.output_root / "controlled_frequency_events.csv", events)
    plots(best, macro, controlled, args.output_root, config)

    macro_view = macro[["scope", "model", "macro_f1max", "TP", "FP", "FN"]]
    dataset_view = best.pivot(index="dataset", columns="model", values="f1_max").reset_index()
    controlled_view = controlled[controlled.scope.isin(["real_6", "all_7"])].copy()
    report = "\n".join([
        "# Paired visual-frequency ViT4TS: final analysis", "",
        "## Completion and protocol", "",
        "All 125 series completed for A–E at α ∈ {0.1, 0.01, 0.001}. The evaluation first averages contextual F1 equally across series for each dataset/α, selects the dataset α with the largest mean, and then macro-averages datasets. `real_6` excludes `artificialWithAnomaly`; `all_7` includes it. CLIP features came from the existing frozen OpenAI ViT-B/16 cache. All methods use stable Harmonic aggregation.", "",
        "## Dataset F1-max", "", markdown_table(dataset_view), "",
        "## Macro results and contextual counts", "", markdown_table(macro_view), "",
        "## Frequency methods versus paired visual B", "", markdown_table(controlled_view), "",
        "## Event changes versus paired visual B", "", markdown_table(events), "",
        "## Findings", "",
        "- B improves over A by 0.045362 on the six real datasets and 0.081428 over all seven datasets. This is evidence that the real-window reference library is useful, although B also increases the base-patch reference capacity from 196 median patches to as many as 32×196 real patches.",
        "- C, D, and E change the six-real macro by -0.026969, -0.031283, and -0.009202 versus B. On all seven datasets the changes are -0.023116, -0.026814, and -0.007887. Thus none of the fixed frequency mechanisms improves the controlled paired-visual baseline.",
        "- E is the strongest frequency variant by macro F1. It exceeds whole-window FFT C by 0.017768 on real_6 and 0.015229 on all_7, but remains below B. Local time alignment is less harmful than whole-window FFT here; it does not establish an independent frequency gain.",
        "- D does not beat B or C in macro F1. Its comparison with C/E is partially confounded: D retains 5 windows (980 patches), whereas C/E retain 5 visual patches per query patch before frequency scoring.",
        "- The controlled result does not support the claim that frequency information improves ViT4TS under this fixed design. The most effective tested matching mechanism is paired visual B. Among frequency-bearing methods, E is best.", "",
        "## Interpretation limits", "",
        "- B–E share the same initial visual pool and exclude the current and overlapping windows with |q-r| ≥ 4. The exclusion applies to the base 14×14 patch branch; the retained large/mid branches still use the original all-window median reference, so the full model is transductive.",
        "- Labels enter only contextual F1-max evaluation and post-hoc event/plot analysis. They are not used in candidate selection, distance calibration, λ, K, or frequency construction.",
        "- Runtime tables are cache-only component diagnostics. They exclude CLIP and image rendering, and shared work is amortized differently across variants; they must not be used as an end-to-end speed ranking.",
        "- `candidate_pool_audit.csv` verifies all 125 series, the ≤32 initial pool, ≥5 non-overlapping candidates, and D's fixed 5-window selection.", "",
        "## Artifacts", "",
        "- `results.csv`, `dataset_f1max.csv`, `macro_f1max.csv`: complete evaluation results.",
        "- `controlled_frequency_vs_B.csv`, `controlled_frequency_events.csv`: controlled frequency effects and event accounting.",
        "- `candidate_pool_audit.csv`: per-series candidate counts.",
        "- `point_scores/`: all 625 point-score arrays.",
        "- `plots/dataset_f1max.png`, `plots/frequency_vs_paired_visual.png`, `plots/failures/`: comparison and failure figures.",
        "- `runtime.csv`, `incremental_timing.csv`: cache-only component timing and memory.", "",
    ])
    publish_bytes_no_replace(args.output_root / "final_analysis.md", report.encode("utf-8"))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--output-root", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
