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
    macro_scopes,
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
    for scope, datasets in macro_scopes(config).items():
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


def candidate_audit(documents: list[dict], config: dict) -> pd.DataFrame:
    guided = {
        (document["dataset"], document["signal"]): document["backend_metadata"]
        for document in documents
        if document["variant"] == "D_fft_guided_reference"
    }
    rows = []
    for document in documents:
        if document["variant"] != "B_paired_visual":
            continue
        windows = int(document["backend_metadata"]["n_windows"])
        counts = [sum(abs(query - ref) >= 4 for ref in range(windows)) for query in range(windows)]
        selected = [min(32, count) for count in counts]
        d_metadata = guided[(document["dataset"], document["signal"])]
        rows.append({
            "dataset": document["dataset"], "signal": document["signal"], "n_windows": windows,
            "min_eligible_nonoverlap_windows": min(counts), "max_eligible_nonoverlap_windows": max(counts),
            "min_common_visual_pool": min(selected), "max_common_visual_pool": max(selected),
            "D_target_windows": int(d_metadata.get("fft_guided_target_windows", 5)),
            "D_min_selected_windows": int(d_metadata.get("fft_guided_min_selected_windows", d_metadata.get("fft_guided_selected_windows", 5))),
            "D_max_selected_windows": int(d_metadata.get("fft_guided_max_selected_windows", d_metadata.get("fft_guided_selected_windows", 5))),
        })
    frame = pd.DataFrame(rows).sort_values(["dataset", "signal"])
    expected = sum(int(config["expected_signal_counts"][name]) for name in config["datasets"])
    if (
        len(frame) != expected
        or frame.max_common_visual_pool.max() > 32
        or frame.min_common_visual_pool.min() < 1
        or frame.D_min_selected_windows.min() < 1
        or frame.D_max_selected_windows.max() > 5
    ):
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
        for scope, datasets in macro_scopes(config).items():
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

    scope_names = list(macro_scopes(config))
    view = controlled[controlled.scope.isin(scope_names)].pivot(index="model", columns="scope", values="delta_f1max")
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
    candidates = candidate_audit(documents, config)
    events = comparison_event_summary(documents, best, config)
    publish_csv(args.output_root / "controlled_frequency_vs_B.csv", controlled)
    publish_csv(args.output_root / "candidate_pool_audit.csv", candidates)
    publish_csv(args.output_root / "controlled_frequency_events.csv", events)
    plots(best, macro, controlled, args.output_root, config)

    macro_view = macro[["scope", "model", "macro_f1max", "TP", "FP", "FN"]]
    dataset_view = best.pivot(index="dataset", columns="model", values="f1_max").reset_index()
    scope_names = list(macro_scopes(config))
    controlled_view = controlled[controlled.scope.isin(scope_names)].copy()
    expected_signals = sum(int(config["expected_signal_counts"][name]) for name in config["datasets"])
    primary_scope = scope_names[-1]
    primary = macro[macro.scope == primary_scope].set_index("model")
    b_minus_a = float(primary.loc["B_paired_visual", "macro_f1max"] - primary.loc["A_stable_median", "macro_f1max"])
    frequency_delta = {
        model: float(primary.loc[model, "macro_f1max"] - primary.loc["B_paired_visual", "macro_f1max"])
        for model in ["C_paired_fft_joint", "D_fft_guided_reference", "E_paired_stft_joint"]
    }
    best_frequency = max(frequency_delta, key=lambda name: float(primary.loc[name, "macro_f1max"]))
    ragged = candidates[candidates.D_min_selected_windows < candidates.D_target_windows]
    report = "\n".join([
        "# Paired visual-frequency ViT4TS: final analysis", "",
        "## Completion and protocol", "",
        f"All {expected_signals} configured series completed for A–E at α ∈ {{0.1, 0.01, 0.001}}. The evaluation first averages contextual F1 equally across series for each dataset/α, selects the dataset α with the largest mean, and then macro-averages datasets in the configured scopes ({', '.join(scope_names)}). CLIP features use frozen OpenAI ViT-B/16. All methods use stable Harmonic aggregation.", "",
        "## Dataset F1-max", "", markdown_table(dataset_view), "",
        "## Macro results and contextual counts", "", markdown_table(macro_view), "",
        "## Frequency methods versus paired visual B", "", markdown_table(controlled_view), "",
        "## Event changes versus paired visual B", "", markdown_table(events), "",
        "## Findings", "",
        f"- In `{primary_scope}`, paired visual B changes macro F1 by {b_minus_a:+.6f} versus stable median A. B also increases the base-patch reference capacity from 196 median patches to as many as 32×196 real patches.",
        f"- Against the controlled paired-visual B baseline, C, D, and E change `{primary_scope}` macro F1 by {frequency_delta['C_paired_fft_joint']:+.6f}, {frequency_delta['D_fft_guided_reference']:+.6f}, and {frequency_delta['E_paired_stft_joint']:+.6f}.",
        f"- The strongest frequency-bearing method is `{best_frequency}`. " + ("It exceeds B in this scope." if frequency_delta[best_frequency] > 0 else "It remains below B in this scope, so these fixed experiments do not establish an independent frequency gain."),
        f"- D uses up to five frequency-selected windows. {len(ragged)} series required fewer than five for at least one query; all retained references remain non-overlapping.", "",
        "## Interpretation limits", "",
        "- B–E share the same initial visual pool and exclude the current and overlapping windows with |q-r| ≥ 4. The exclusion applies to the base 14×14 patch branch; the retained large/mid branches still use the original all-window median reference, so the full model is transductive.",
        "- Labels enter only contextual F1-max evaluation and post-hoc event/plot analysis. They are not used in candidate selection, distance calibration, λ, K, or frequency construction.",
        "- Runtime tables are cache-only component diagnostics. They exclude CLIP and image rendering, and shared work is amortized differently across variants; they must not be used as an end-to-end speed ranking.",
        f"- `candidate_pool_audit.csv` verifies all {expected_signals} series, the ≤32 initial pool, at least one non-overlapping candidate, and D's 1–5-window selection.", "",
        "## Artifacts", "",
        "- `results.csv`, `dataset_f1max.csv`, `macro_f1max.csv`: complete evaluation results.",
        "- `controlled_frequency_vs_B.csv`, `controlled_frequency_events.csv`: controlled frequency effects and event accounting.",
        "- `candidate_pool_audit.csv`: per-series candidate counts.",
        f"- `point_scores/`: all {expected_signals * len(config['variants_order'])} point-score arrays.",
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
