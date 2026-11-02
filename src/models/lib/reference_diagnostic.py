"""Controlled first-stage ViT4TS median-reference contamination experiment.

Run: CLIP_PRETRAINED=openai conda run -n vit python src/models/lib/reference_diagnostic.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CLIP_PRETRAINED", "openai")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import detrend

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from models.vit4ts import ViT4TS
from models.model_utils import (
    align_anomaly_vector, compute_detection_intervals,
    compute_patch_dissimilarity, harmonic_aggregation, stitch_anomaly_maps,
)
from preprocessing.preprocess import draw_windowed_images, preprocess_time_series
from evaluation.evaluate import window_wise_metrics, compute_precision_recall_f1

OUT = ROOT / "data/runs/vit4ts_challenges/reference"
WINDOW, STEP, N_WINDOWS = 224, 56, 80
LENGTH = WINDOW + STEP * (N_WINDOWS - 1)
TARGET_K = 38
BACKGROUND_K = (6, 14, 22, 30, 45, 53, 61, 69)
LEVELS = (0, 5, 10, 20, 40)
ALPHAS = (0.1, 0.01, 0.001)


def make_series():
    t = np.arange(LENGTH)
    # Period exactly matches the stride; all motifs are aligned in image windows.
    clean = 0.5 + 0.11 * np.sin(2 * np.pi * t / STEP) + 0.018 * np.sin(4 * np.pi * t / STEP)
    motif = np.r_[np.linspace(0.0, 0.29, 6), np.linspace(0.29, 0.0, 6)]

    def add_motif(values, k):
        start = k * STEP + 16
        values[start:start + len(motif)] += motif
        return (start, start + len(motif) - 1)

    target_raw = clean.copy()
    target_interval = add_motif(target_raw, TARGET_K)
    master_raw = target_raw.copy()
    background_intervals = [add_motif(master_raw, k) for k in BACKGROUND_K]
    # The fixed master scale is a deliberate causal control. It uses the exact
    # source detrend + min/max implementation once, then holds its affine map.
    master_proc = preprocess_time_series(master_raw)
    detrended_master = detrend(master_raw)
    scale = np.ptp(detrended_master)
    clean_proc = master_proc - (master_raw - clean) / scale
    target_proc = master_proc - (master_raw - target_raw) / scale
    assert np.max(np.abs(target_proc[target_interval[0]:target_interval[1]+1] -
                         master_proc[target_interval[0]:target_interval[1]+1])) < 1e-12
    return clean, target_raw, master_raw, clean_proc, target_proc, master_proc, target_interval, background_intervals


def covered_windows(interval):
    s, e = interval
    return [i for i in range(N_WINDOWS) if i * STEP <= e and i * STEP + WINDOW - 1 >= s]


def images(values, temp_dir, name):
    ok = draw_windowed_images(
        base_series_id=name, save_path=temp_dir, time_series=values,
        time_points=np.arange(LENGTH), window_size=WINDOW, step_size=STEP,
        image_size=(224, 224), dpi=100, save_image=False,
        plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
    )
    if not ok:
        raise RuntimeError("Image rendering failed")
    return np.load(Path(temp_dir) / f"{name}_line_img.npy")


@torch.no_grad()
def encode(model, image_bank, batch_size=8):
    result = [[], [], []]
    for start in range(0, len(image_bank), batch_size):
        batch = torch.from_numpy(image_bank[start:start + batch_size]).float().to(model.device)
        large, mid, patch, _, _, _ = model.model.encode_image(batch, model.patch_size)
        for dest, block in zip(result, (large, mid, patch)):
            dest.append(block.detach().cpu())
    return tuple(torch.cat(parts, dim=0) for parts in result)


def compose(source_clean, source_target, source_master, injected_indices):
    """Use actual source-rendered sliding images; swap only affected positions."""
    combined = [x.clone() for x in source_clean]
    target_windows = covered_windows(target_interval_global)
    for scale in range(3):
        combined[scale][target_windows] = source_target[scale][target_windows]
        combined[scale][injected_indices] = source_master[scale][injected_indices]
    return tuple(combined)


def median_memory(features):
    # Source compute_patch_dissimilarity takes torch.median along window axis.
    # Singleton memory here preserves its exact result and avoids recomputing it.
    return tuple(torch.median(t, dim=0).values.unsqueeze(0) for t in features)


def cosine_to_reference(median, prototype):
    a = median.squeeze(0).reshape(-1, median.shape[-1])
    b = prototype.reshape(-1, prototype.shape[-1])
    return float(F.cosine_similarity(a, b, dim=-1).mean())


@torch.no_grad()
def score(model, test_features, memory, batch_size=8):
    maps = []
    large_mask, mid_mask = model.model.mask.make_mask(16, 48, 16), model.model.mask.make_mask(16, 32, 16)
    # mask.make_mask gives exactly the same masks as encode_image's aggregation.
    large_mask = large_mask.squeeze(0)
    mid_mask = mid_mask.squeeze(0)
    for start in range(0, N_WINDOWS, batch_size):
        end = min(N_WINDOWS, start + batch_size)
        b = end - start
        patch_maps = []
        for index in range(3):
            token = test_features[index][start:end].to(model.device)
            mem = {"series": memory[index].to(model.device)}
            dist = compute_patch_dissimilarity(mem, token, ["series"] * b)
            if index == 0:
                dist = harmonic_aggregation((b, 14, 14), dist, large_mask)
            elif index == 1:
                dist = harmonic_aggregation((b, 14, 14), dist, mid_mask)
            else:
                dist = dist.reshape(b, 14, 14)
            patch_maps.append(dist)
        combined = torch.nan_to_num(sum(patch_maps) / 3, nan=0, posinf=0, neginf=0)
        up = F.interpolate(combined.unsqueeze(1), size=(224, 224), mode="bilinear").squeeze(1)
        maps.append(up.cpu().numpy())
    image_maps = np.concatenate(maps)
    vector = stitch_anomaly_maps(image_maps, 4.0, 0.25)
    aligned = align_anomaly_vector(vector, LENGTH, WINDOW, STEP, N_WINDOWS)
    return aligned


def metric(model, scores, truths, alpha):
    detected, threshold, _ = compute_detection_intervals(scores, alpha=alpha)
    counts = window_wise_metrics(truths, detected)
    prf = compute_precision_recall_f1(counts)
    target_hit = any(s <= target_interval_global[1] and e >= target_interval_global[0]
                     for s, e in detected)
    return dict(alpha=alpha, threshold=float(threshold), detected_intervals=detected,
                target_hit=target_hit, **counts, precision=prf["precision"],
                recall=prf["recall"], f1=prf["F1"])


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def plot_summary(records, scores_by):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for mode in ("self_reference", "external_reference", "fixed"):
        vals = [r for r in records if r["mode"] == mode]
        ax[0].plot(LEVELS, [r["target_mean_score"] for r in vals], marker="o", label=mode)
        ax[1].plot(LEVELS, [r["target_percentile_rank"] for r in vals], marker="o", label=mode)
        ax[2].plot(LEVELS, [r["f1_max"] for r in vals], marker="o", label=mode)
    for a, title, ylabel in zip(ax, ("Target raw anomaly score", "Target score rank", "Contextual F1-max"),
                                ("mean score", "percentile (higher = stronger)", "F1-max")):
        a.set_title(title); a.set_xlabel("Background contaminated windows (%)"); a.set_ylabel(ylabel)
        a.grid(alpha=0.2); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "contamination_vs_detection.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    region = slice(target_interval_global[0]-180, target_interval_global[1]+180)
    for level in LEVELS:
        ax.plot(np.arange(LENGTH)[region], scores_by[(level, "external_reference")][region], label=f"external {level}%")
    ax.plot(np.arange(LENGTH)[region], scores_by[(0, "fixed")][region], color="black", ls="--", lw=2, label="fixed clean")
    ax.axvspan(*target_interval_global, color="tomato", alpha=0.2)
    ax.set(xlabel="time index", ylabel="raw ViT4TS score", title="Unchanged target anomaly")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "target_score_curves.png", dpi=160); plt.close(fig)


def plot_similarity(rows):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for a, scale in zip(ax, ("large", "mid", "patch")):
        for mode in ("self_reference", "external_reference", "fixed"):
            vals = [r for r in rows if r["mode"] == mode and r["scale"] == scale]
            a.plot(LEVELS, [r["reference_vs_target_cosine"] for r in vals],
                   marker="o", label=f"{mode}: target")
            a.plot(LEVELS, [r["reference_vs_normal_cosine"] for r in vals],
                   ls="--", alpha=0.8, label=f"{mode}: normal")
        a.set(title=f"{scale} scale", xlabel="Background contaminated windows (%)")
        a.grid(alpha=0.2)
    ax[0].set_ylabel("Cosine similarity to prototype")
    ax[2].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(); fig.savefig(OUT / "reference_target_similarity.png", dpi=160); plt.close(fig)


def run(args):
    global target_interval_global
    torch.manual_seed(17); np.random.seed(17)
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    clean, target, master, cp, tp, mp, target_interval_global, background_intervals = make_series()
    target_windows = covered_windows(target_interval_global)
    background_windows = [covered_windows(x) for x in background_intervals]
    assert len(target_windows) == 4 and all(len(x) == 4 for x in background_windows)
    assert len(set(sum(background_windows, []))) == 32
    assert not (set(target_windows) & set(sum(background_windows, [])))

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(clean, lw=0.8, label="normal")
    ax.plot(master, lw=0.8, label="normal + target + eight background anomalies")
    ax.axvspan(*target_interval_global, color="tomato", alpha=0.3)
    ax.set(xlabel="time index", ylabel="raw value", title="Controlled 224/56 sequence")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "controlled_sequence.png", dpi=150); plt.close(fig)

    model = ViT4TS(window_size=WINDOW, window_step_ratio=4, device="cpu",
                   batch_size=8, standardize=True, verbose=False)
    with tempfile.TemporaryDirectory() as work:
        clean_img = images(cp, work, "clean")
        target_img = images(tp, work, "target")
        master_img = images(mp, work, "master")
        clean_feat = encode(model, clean_img)
        target_feat = encode(model, target_img)
        master_feat = encode(model, master_img)
        source_vector = model._run_inference(work, "target")
        source_score = align_anomaly_vector(source_vector, LENGTH, WINDOW, STEP, N_WINDOWS)
        # A sequence-level check: render the 20% case as one contiguous series
        # under the same fixed affine transform, then compare every image.
        selected20 = set(sum(background_windows[:4], []))
        raw20 = target.copy()
        for s, e in background_intervals[:4]: raw20[s:e+1] = master[s:e+1]
        proc20 = mp - (master - raw20) / np.ptp(detrend(master))
        direct20 = images(proc20, work, "direct20")
        assembled20 = target_img.copy()
        assembled20[list(selected20)] = master_img[list(selected20)]
        max_image_diff = float(np.max(np.abs(direct20 - assembled20)))
        if max_image_diff > 1e-6:
            raise AssertionError(f"Direct sequence rendering differs: {max_image_diff}")

    pure_memory = median_memory(clean_feat)
    scores_by = {}; records = []; detail = []; similarities = []
    target_proto = tuple(t[target_windows].mean(dim=0) for t in target_feat)
    normal_proto = tuple(t[target_windows].mean(dim=0) for t in clean_feat)
    pure_score = score(model, target_feat, pure_memory)
    source_score_diff = None
    target_slice = slice(target_interval_global[0], target_interval_global[1]+1)
    for level in LEVELS:
        # 5/10/20/40% = 1/2/4/8 isolated motifs, each contaminates four windows.
        chosen = {0: 0, 5: 1, 10: 2, 20: 4, 40: 8}[level]
        selected = sorted(set(sum(background_windows[:chosen], [])))
        reference = compose(clean_feat, target_feat, master_feat, selected)
        self_memory = median_memory(reference)
        self_score = score(model, target_feat, self_memory)
        if level == 0:
            source_score_diff = float(np.max(np.abs(self_score - source_score)))
            if source_score_diff > 1e-6:
                raise AssertionError(f"Source score mismatch: {source_score_diff}")
        external_features = [x.clone() for x in clean_feat]
        for scale in range(3):
            external_features[scale][selected] = master_feat[scale][selected]
        external_memory = median_memory(external_features)
        external_score = score(model, target_feat, external_memory)
        for mode, vec, mem, target_in_reference in (
            ("self_reference", self_score, self_memory, True),
            ("external_reference", external_score, external_memory, False),
            ("fixed", pure_score, pure_memory, False),
        ):
            scores_by[(level, mode)] = vec
            truths = [target_interval_global]
            evaluations = [metric(model, vec, truths, a) for a in ALPHAS]
            best = max(evaluations, key=lambda x: x["f1"])
            normal_score = np.r_[vec[:target_interval_global[0]], vec[target_interval_global[1]+1:]]
            target_mean = float(vec[target_slice].mean())
            affected = len(selected) if mode != "fixed" else 0
            records.append(dict(level_percent=level, mode=mode,
                background_contaminated_windows=affected,
                actual_background_window_percent=100*affected/N_WINDOWS,
                target_contaminated_windows=len(target_windows) if target_in_reference else 0,
                total_contaminated_windows=affected+(len(target_windows) if target_in_reference else 0),
                actual_total_window_percent=100*(affected+(len(target_windows) if target_in_reference else 0))/N_WINDOWS,
                reference_window_count=N_WINDOWS,
                target_mean_score=target_mean, target_max_score=float(vec[target_slice].max()),
                target_percentile_rank=float(100*(normal_score < target_mean).mean()),
                f1_max=best["f1"], best_alpha=best["alpha"],
                best_target_hit=best["target_hit"], best_TP=best["TP"],
                best_FP=best["FP"], best_FN=best["FN"]))
            for e in evaluations:
                detail.append(dict(level_percent=level, mode=mode,
                                   **{**e, "detected_intervals": json.dumps(e["detected_intervals"])}))
            for i, scale in enumerate(("large", "mid", "patch")):
                similarities.append(dict(level_percent=level, mode=mode, scale=scale,
                    reference_vs_target_cosine=cosine_to_reference(mem[i], target_proto[i]),
                    reference_vs_normal_cosine=cosine_to_reference(mem[i], normal_proto[i]),
                    reference_vs_pure_cosine=cosine_to_reference(mem[i], pure_memory[i])))

    save_csv(OUT / "summary.csv", records)
    save_csv(OUT / "threshold_details.csv", detail)
    save_csv(OUT / "reference_similarity.csv", similarities)
    np.savez_compressed(OUT / "raw_scores.npz", **{f"{mode}_{level}": vec for (level, mode), vec in scores_by.items()})
    plot_summary(records, scores_by)
    plot_similarity(similarities)
    with (OUT / "metadata.json").open("w") as f:
        json.dump(dict(seed=17, checkpoint=str(model.model.checkpoint),
                       model="OpenAI CLIP ViT-B/16", window=WINDOW, stride=STEP,
                       image_size=[224,224], points=LENGTH, windows=N_WINDOWS,
                       target_interval=target_interval_global,
                       target_windows=target_windows,
                       background_intervals=background_intervals,
                       background_window_indices=background_windows,
                       fixed_normalization="one source detrend/minmax affine transform from master series",
                       sequence_level_image_max_abs_diff_20pct=max_image_diff,
                       source_score_max_abs_diff_0pct=source_score_diff), f, indent=2)
    print(pd.DataFrame(records).to_string(index=False), flush=True)
    print("sequence image verification max abs difference", max_image_diff, flush=True)
    print("source score verification max abs difference", source_score_diff, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
