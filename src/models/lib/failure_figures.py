"""Case figures for the ViT4TS real failure audit.

Generates success/failure comparison figures from the cached 224/56 baseline
scores, captured per-window vectors and real labels.  No new inference, no
synthetic anomalies.  Output: data/runs/vit4ts_failure_audit/case_figures/
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/models/lib"))
from models.model_utils import compute_detection_intervals
from failure_audit import DATASETS, STRIDE, WINDOW, load_index_and_labels, \
    load_scores, load_window_vectors

OUT = ROOT / "data/runs/vit4ts_failure_audit"
FIG = OUT / "case_figures"
FIG.mkdir(parents=True, exist_ok=True)

index, labels = load_index_and_labels()
best_alpha = json.loads((OUT / "audit_summary.json").read_text())["best_alpha"]
cases = pd.read_csv(OUT / "failure_cases.csv")
followup = pd.read_csv(OUT / "event_followup.csv")

plt.rcParams.update({"font.size": 8, "figure.dpi": 150})


def ewma_of(scores):
    span = max(1, int(len(scores) * 0.01))
    return pd.Series(scores).ewm(span=span).mean().values


def case_arrays(dataset, signal, alpha=None):
    alpha = alpha if alpha is not None else best_alpha[dataset]
    top, q25, ts, _ = load_scores(dataset, signal)
    truth = [(int(a), int(b)) for a, b in labels[signal]]
    ewma = ewma_of(top)
    iv, thr, _ = compute_detection_intervals(top, alpha=alpha)
    pred = [(int(ts[a]), int(ts[b])) for a, b in iv]
    vectors = load_window_vectors(dataset, signal)
    return dict(top=top, q25=q25, ts=ts, truth=truth, ewma=ewma, pred=pred,
                thr=thr, alpha=alpha, vectors=vectors)


def shade(axis, spans, color, alpha=0.25, label=None):
    first = True
    for a, b in spans:
        axis.axvspan(a, b, color=color, alpha=alpha, linewidth=0,
                     label=label if first else None)
        first = False


def draw_case(axes, dataset, signal, title, pred_color="#1f77b4",
              alpha=None, show_raw_score=True):
    data = case_arrays(dataset, signal, alpha)
    T = len(data["top"])
    x = np.arange(T)
    truth_idx = [(int(np.searchsorted(data["ts"], a, "left")),
                  int(np.searchsorted(data["ts"], b, "right")) - 1)
                 for a, b in data["truth"]]
    ax0, ax1 = axes
    raw = pd.read_csv(ROOT / "data" / dataset / (signal + ".csv"))
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    ax0.plot(x, raw.value.to_numpy(float), lw=0.6, color="black")
    shade(ax0, truth_idx, "#d62728", 0.18, "true interval")
    shade(ax0, [(a, b) for a, b in data["pred"]], pred_color, 0.28,
          "predicted (alpha=%.3g)" % data["alpha"])
    ax0.set_title(title, fontsize=8)
    ax0.set_ylabel("value")
    ax0.set_xlim(0, T - 1)
    if show_raw_score:
        ax1.plot(x, data["top"], lw=0.5, color="silver", label="raw score")
    ax1.plot(x, data["ewma"], lw=0.8, color="black", label="EWMA score")
    ax1.axhline(data["thr"], color="green", lw=0.8,
                label="threshold (official)")
    shade(ax1, truth_idx, "#d62728", 0.18)
    ax1.set_ylabel("score")
    ax1.set_xlabel("point index")
    ax1.set_xlim(0, T - 1)
    ax1.legend(fontsize=6, loc="upper right", framealpha=0.8)
    data["truth_idx"] = truth_idx
    return data


def fig_threshold_band():
    margins = cases.copy()
    margins["ratio"] = margins.score_max / margins.threshold
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    groups = [("TP", margins[margins.event_type == "TP"], "#2ca02c"),
              ("FN", margins[margins.event_type == "FN"], "#d62728"),
              ("FP", margins[margins.event_type == "FP"], "#1f77b4")]
    for name, group, color in groups:
        axes[0].hist(group["ratio"], bins=np.logspace(-1.3, 0.6, 60),
                     alpha=0.55, label=name, color=color)
    axes[0].axvline(1.0, color="black", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("event score_max / threshold")
    axes[0].set_ylabel("events")
    axes[0].legend(fontsize=7)
    axes[0].set_title("FN and FP crowd the same threshold band", fontsize=8)
    for name, group, color in groups:
        axes[1].hist(group["score_percentile"], bins=40, alpha=0.55,
                     label=name, color=color)
    axes[1].set_xlabel("event score percentile within sequence")
    axes[1].set_title("Both failure types live in the top ~10% scores", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fn_fp_threshold_band.png")
    plt.close(fig)


def fig_mode_decomposition():
    fn = followup[followup.event_type == "FN"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colors = dict(zip(DATASETS,
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2"]))
    for ds, group in fn.groupby("dataset"):
        ax.scatter(group.raw_over_thr, group.ewma_over_thr, s=18,
                   label=ds, color=colors[ds], alpha=0.8, linewidth=0)
    lim = [0, max(2.6, fn[["raw_over_thr", "ewma_over_thr"]].max().max() * 1.05)]
    ax.plot(lim, lim, color="silver", lw=0.7)
    ax.axvline(1, color="green", lw=0.8, ls="--")
    ax.axhline(1, color="black", lw=0.8)
    ax.set_xlim(0.3, lim[1]); ax.set_ylim(0.3, lim[1])
    ax.set_xlabel("raw stitched score_max / threshold (before EWMA)")
    ax.set_ylabel("EWMA score_max / threshold (official)")
    lost = fn[(fn.raw_over_thr > 1) & (fn.ewma_over_thr <= 1)]
    ax.annotate("lost at EWMA stage: %d events" % len(lost),
                xy=(1.6, 0.9), fontsize=7, color="#d62728")
    weak = fn[fn.raw_over_thr <= 1]
    ax.annotate("raw below threshold: %d events\n"
                "(20 of them also weak per-window)"
                % len(weak),
                xy=(0.42, 0.42), fontsize=7, color="black")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("62 real FN events: score at each recovery stage\n"
                 "vs. official threshold", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fn_mode_decomposition.png")
    plt.close(fig)


def fig_pair(dataset_a, signal_a, title_a, dataset_b, signal_b, title_b,
             filename, alpha_a=None, alpha_b=None):
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.4))
    draw_case(axes[:, 0], dataset_a, signal_a, title_a, alpha=alpha_a)
    draw_case(axes[:, 1], dataset_b, signal_b, title_b, alpha=alpha_b)
    fig.tight_layout()
    fig.savefig(FIG / filename)
    plt.close(fig)


def fig_case(dataset, signal, title, filename, alpha=None, zoom=None):
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.2))
    draw_case(axes, dataset, signal, title, alpha=alpha)
    if zoom:
        i0, i1 = zoom
        for ax in axes:
            ax.set_xlim(i0, i1)
    fig.tight_layout()
    fig.savefig(FIG / filename)
    plt.close(fig)


def fig_case_perwindow(dataset, signal, event_i0, event_i1, title, filename,
                       alpha=None):
    """Failure case with a per-window evidence panel (Mode S)."""
    data = case_arrays(dataset, signal, alpha)
    vectors = data["vectors"]
    T = len(data["top"])
    n = len(vectors)
    cover = np.zeros(T)
    w_min = max(0, -(-(event_i0 - WINDOW + 1) // STRIDE))
    w_max = min(n - 1, event_i1 // STRIDE)
    perwindow = np.full(T, np.nan)
    for w in range(w_min, w_max + 1):
        j0 = max(event_i0 - w * STRIDE, 0)
        j1 = min(event_i1 - w * STRIDE, WINDOW - 1)
        if j1 < j0:
            continue
        seg = vectors[w, j0:j1 + 1]
        pos = np.arange(w * STRIDE + j0, w * STRIDE + j0 + len(seg))
        cover[pos] = np.maximum(cover[pos], 0)
        perwindow[pos] = np.fmax(perwindow[pos], seg)
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.4))
    data = draw_case(axes[:2], dataset, signal, title, alpha=alpha)
    x = np.arange(T)
    axes[2].plot(x, data["top"], lw=0.7, color="black",
                 label="final score (overlap-mean)")
    m = ~np.isnan(perwindow)
    axes[2].plot(x[m], perwindow[m], lw=0.7, color="#d62728",
                 label="per-window max (captured vectors)")
    axes[2].axhline(data["thr"], color="green", lw=0.8, label="threshold")
    shade(axes[2], data["truth_idx"], "#d62728", 0.18)
    lo = max(0, event_i0 - 300); hi = min(T, event_i1 + 300)
    axes[2].set_xlim(lo, hi)
    axes[2].set_ylabel("score (event zoom)")
    axes[2].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / filename)
    plt.close(fig)


def main():
    fig_threshold_band()
    fig_mode_decomposition()

    # FN vs success contrast inside SMAP
    fig_pair("SMAP", "T-2", "SMAP T-2: FN - one long true event, margin -0.3%",
             "SMAP", "T-3", "SMAP T-3: success - two events, F1=1.0 at alpha=0.01",
             "case_SMAP_T2_failure_vs_T3_success.png")

    # SMAP E-13: three FN events (borderline)
    fig_case("SMAP", "E-13",
             "SMAP E-13: 3 FN events (all borderline, alpha=0.01 best)",
             "case_SMAP_E13_three_missed_events.png")

    # long anomaly, weak separability (SMAP A-8, 42% anomalous)
    fig_case("SMAP", "A-8",
             "SMAP A-8: 3806-pt event missed; score peak below even the "
             "normal-only threshold", "case_SMAP_A8_long_event.png")

    # AWS Mode-S FN with per-window evidence
    ev = cases[(cases.dataset == "realAWSCloudwatch")
               & (cases.signal == "ec2_cpu_utilization_5f5533")
               & (cases.event_type == "FN")]
    fig_case_perwindow("realAWSCloudwatch", "ec2_cpu_utilization_5f5533",
                       int(ev.iloc[0].i0), int(ev.iloc[0].i1),
                       "AWS ec2_cpu_utilization_5f5533: FN with 1.35x-threshold "
                       "per-window evidence (alpha=0.1)",
                       "case_AWS_5f5533_perwindow_evidence.png")

    # FP-only clean sequence
    fig_case("realAWSCloudwatch", "ec2_cpu_utilization_c6585a",
             "AWS ec2_cpu_utilization_c6585a: zero true events, 14 FPs at "
             "alpha=0.1", "case_FP_AWS_c6585a_clean_sequence.png")

    # FP burst sequence
    fig_case("MSL", "F-8",
             "MSL F-8: 1 true event detected but 12 FPs (alpha=0.1)",
             "case_FP_MSL_F8_bursts.png")

    # label/visual mismatch FPs
    fig_case("artificialWithAnomaly", "art_increase_spike_density",
             "artificialWithAnomaly art_increase_spike_density: 31 FPs - "
             "visually spiky, label covers only one region (alpha=0.1)",
             "case_FP_art_spike_density_label_mismatch.png")

    # realTweets FN recovered at loosest alpha
    fig_pair("realTweets", "Twitter_volume_IBM",
             "realTweets Twitter_volume_IBM: FN at best alpha=0.001",
             "realTweets", "Twitter_volume_IBM",
             "same sequence at alpha=0.1: both events detected, 30 FPs",
             "case_tweets_IBM_alpha_tradeoff.png", alpha_a=0.001, alpha_b=0.1)

    print("figures written to", FIG)


if __name__ == "__main__":
    main()
