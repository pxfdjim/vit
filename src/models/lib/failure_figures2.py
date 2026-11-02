"""Figures for the q25 verification, stage decomposition and consistency
analyses.  Output: data/runs/vit4ts_failure_audit/case_figures/"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/runs/vit4ts_failure_audit"
FIG = OUT / "case_figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})

q25 = json.loads((OUT / "q25_summary.json").read_text())["per_dataset"]
stage = pd.read_csv(OUT / "stage_decomposition.csv")
ref = pd.read_csv(OUT / "reference_stats.csv")
DATASETS = ["artificialWithAnomaly", "realAWSCloudwatch", "realAdExchange",
            "realTraffic", "realTweets", "MSL", "SMAP"]
SHORT = {"artificialWithAnomaly": "artificial", "realAWSCloudwatch": "AWS",
         "realAdExchange": "AdExchange", "realTraffic": "Traffic",
         "realTweets": "Tweets", "MSL": "MSL", "SMAP": "SMAP"}
COLOR = {"FN": "#d62728", "TP": "#2ca02c", "FP": "#1f77b4"}
# sequences whose stored baseline is destroyed by the harmonic-aggregation
# numerical instability; their thresholds/margins are meaningless
ARTIFACT_SEQS = {("SMAP", s) for s in ("A-1", "D-8", "G-2", "P-4", "R-1")}
_kept = lambda df: df[[k not in ARTIFACT_SEQS
                       for k in zip(df.dataset, df.signal)]]
stage = _kept(stage)
ref = _kept(ref)


def fig_q25():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(DATASETS))
    w = 0.27
    top_own = [q25[d]["top"]["f1max_fixed_alpha"] for d in DATASETS]
    q25_own = [q25[d]["q25"]["f1max_fixed_alpha"] for d in DATASETS]
    q25_fixed = [q25[d]["q25"]["f1_at_top_best_alpha"] for d in DATASETS]
    ax.bar(x - w, top_own, w, label="top25% mean @ own $\\alpha^*$", color="#7f7f7f")
    ax.bar(x, q25_own, w, label="row-q25 @ own $\\alpha^*$", color="#1f77b4")
    ax.bar(x + w, q25_fixed, w, label="row-q25 @ top's $\\alpha^*$ (shared)",
           color="#aec7e8")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[d] for d in DATASETS])
    ax.set_ylabel("dataset F1-max")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    ax.set_title("q25 vs top25% aggregation: own-alpha vs shared-alpha "
                 "(paper protocol, cached scores)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "q25_alpha_interaction.png")
    plt.close(fig)


def fig_stage_funnel():
    stages = ["perwindow_over_thr", "raw_over_thr", "ewma_over_thr"]
    labels = ["per-window\n(before stitch)", "overlap-mean\n(before EWMA)",
              "EWMA\n(official score)"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for et in ("FN", "TP", "FP"):
        sub = stage[stage.event_type == et]
        med = [sub[s].median() for s in stages]
        p25 = [sub[s].quantile(0.25) for s in stages]
        p75 = [sub[s].quantile(0.75) for s in stages]
        axes[0].plot(range(3), med, "-o", color=COLOR[et], label=et)
        axes[0].fill_between(range(3), p25, p75, color=COLOR[et], alpha=0.15)
    axes[0].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("event score / official threshold")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Median score at each recovery stage\n"
                      "(IQR shaded)", fontsize=8)
    for et in ("FN", "TP", "FP"):
        sub = stage[stage.event_type == et]
        axes[1].hist(sub.windows_above_thr_frac, bins=20, alpha=0.5,
                     color=COLOR[et], label=et)
    axes[1].set_xlabel("fraction of covering windows whose event-region "
                       "max exceeds threshold")
    axes[1].set_ylabel("events")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Multi-window consistency", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stage_funnel_consistency.png")
    plt.close(fig)


def fig_persistence():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    for et in ("FN", "TP", "FP"):
        sub = stage[stage.event_type == et]
        axes[0].hist(sub.runlen_ratio.clip(0, 1), bins=20, alpha=0.5,
                     color=COLOR[et], label=et)
        axes[1].hist(sub.duration_points.clip(0, 600), bins=40, alpha=0.5,
                     color=COLOR[et], label=et)
    axes[0].set_xlabel("longest run above per-signal p90 / event duration")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Local persistence of the raw score", fontsize=8)
    axes[1].set_xlabel("event duration (points, clipped at 600)")
    axes[1].set_title("Duration distribution", fontsize=8)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "persistence_distributions.png")
    plt.close(fig)


def fig_reference():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    for et in ("FN", "TP", "FP"):
        sub = ref[ref.event_type == et]
        axes[0].hist(sub.mean_over_bg_p50.clip(0, 4), bins=30, alpha=0.5,
                     color=COLOR[et], label=et)
        axes[1].hist(sub.frac_above_bg_p95, bins=20, alpha=0.5,
                     color=COLOR[et], label=et)
    axes[0].axvline(1.0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("event patch dissimilarity / background median")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Reference-matching strength", fontsize=8)
    axes[1].set_xlabel("fraction of event patches above background p95")
    axes[1].set_title("Spatial extent of reference mismatch", fontsize=8)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "reference_matching_events.png")
    plt.close(fig)


def fig_q25_decomposition():
    """Where the q25 gain comes from: shared-alpha improvement vs alpha
    re-selection vs artifact robustness."""
    diag = json.loads((OUT / "artifact_diagnostic.json").read_text())
    smap_stab = diag["SMAP_dataset_f1max"]["stabilized"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(DATASETS))
    shared = [q25[d]["q25"]["f1_at_top_best_alpha"]
              - q25[d]["top"]["f1max_fixed_alpha"] for d in DATASETS]
    own = [q25[d]["q25"]["f1max_fixed_alpha"]
           - q25[d]["q25"]["f1_at_top_best_alpha"] for d in DATASETS]
    ax.bar(x, shared, 0.55, label="$\\Delta$ at shared fixed $\\alpha$",
           color="#1f77b4")
    ax.bar(x, own, 0.55, bottom=shared,
           label="$\\Delta$ from own-$\\alpha$ re-selection", color="#aec7e8")
    for i, d in enumerate(DATASETS):
        if d == "SMAP":
            ax.text(i, q25[d]["q25"]["f1max_fixed_alpha"] + 0.02,
                    "SMAP: 5 artifact sequences 0$\\to$1.0 after "
                    "stabilization alone ($F1_{max}$ 0.523$\\to$0.617)",
                    fontsize=6.5, ha="center", color="#d62728")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[d] for d in DATASETS])
    ax.set_ylabel("$\\Delta$ F1-max (row-q25 $-$ top25%)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("q25 gain decomposed: shared-$\\alpha$ improvement vs "
                 "$\\alpha$ re-selection", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "q25_gain_decomposition.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_q25()
    fig_q25_decomposition()
    fig_stage_funnel()
    fig_persistence()
    fig_reference()
    print("figures written to", FIG)
