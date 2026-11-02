"""Fixed real-signal external-validity anchor for reference contamination.

Run with the vit environment and CLIP_PRETRAINED=openai. The signal, crop,
motif, and injection sites below were fixed from label/length audit before
any model scores were inspected.
"""
from __future__ import annotations

import csv
import ast
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CLIP_PRETRAINED", "openai")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import detrend

import reference_diagnostic as r
from models.vit4ts import ViT4TS
from preprocessing.preprocess import preprocess_time_series

OUT = r.ROOT / "data/runs/vit4ts_challenges/reference_real"
SOURCE = r.ROOT / "data/SMAP/A-6.csv"
CROP_START = 200
ORIGINAL_TARGET = (2572, 2612)
TARGET = tuple(x-CROP_START for x in ORIGINAL_TARGET)
INJECTION_K = (6, 14, 22, 30, 52, 60, 68, 76)
LEVELS = (0, 5, 10, 20, 40)
ALPHAS = r.ALPHAS


def covered(interval):
    return r.covered_windows(interval)


def save_csv(path, rows):
    if not rows: return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot(sequence, injection_intervals, summary, similarities, scores):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(sequence, lw=0.8, label="SMAP A-6 observed")
    ax.axvspan(*TARGET, color="tomato", alpha=0.25, label="labeled target anomaly")
    for s, e in injection_intervals:
        ax.axvspan(s, e, color="royalblue", alpha=0.12)
    ax.set(xlabel="index within fixed 4,648-point crop", ylabel="observed value",
           title="Real A-6 signal and distant reference-copy positions")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(OUT / "real_sequence.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mode in ("self_reference", "external_reference", "fixed"):
        rows = [x for x in summary if x["mode"] == mode]
        ax[0].plot(LEVELS, [x["target_mean_score"] for x in rows], marker="o", label=mode)
        ax[1].plot(LEVELS, [x["f1_max"] for x in rows], marker="o", label=mode)
    ax[0].set(ylabel="mean target raw score", title="Target score")
    ax[1].set(ylabel="unweighted contextual F1-max", title="Detection")
    for a in ax:
        a.set_xlabel("Extra background-contaminated windows (%)"); a.grid(alpha=0.2); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "real_contamination_vs_detection.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for a, scale in zip(ax, ("large", "mid", "patch")):
        for mode in ("external_reference", "self_reference", "fixed"):
            rows = [x for x in similarities if x["scale"] == scale and x["mode"] == mode]
            a.plot(LEVELS, [x["reference_vs_target_cosine"] for x in rows], marker="o", label=f"{mode}: target")
            a.plot(LEVELS, [x["reference_vs_normal_cosine"] for x in rows], ls="--", label=f"{mode}: normal")
        a.set(title=scale, xlabel="Extra contaminated windows (%)")
        a.grid(alpha=0.2)
    ax[0].set_ylabel("Cosine similarity to prototype")
    ax[2].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(); fig.savefig(OUT / "real_reference_similarity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    lo, hi = max(0,TARGET[0]-250), min(r.LENGTH,TARGET[1]+251)
    xx = np.arange(lo,hi)
    for level in LEVELS:
        ax.plot(xx, scores[(level,"external_reference")][lo:hi], label=f"external {level}%")
    ax.plot(xx, scores[(0,"fixed")][lo:hi], color="black", ls="--", label="fixed")
    ax.axvspan(*TARGET, color="tomato", alpha=0.2)
    ax.set(xlabel="crop index", ylabel="raw score", title="Fixed real target score")
    ax.legend(ncol=3,fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "real_target_score_curves.png",dpi=150);plt.close(fig)


def run():
    r.target_interval_global = TARGET
    np.random.seed(17); torch.manual_seed(17); torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE).sort_values("timestamp").reset_index(drop=True)
    assert len(frame) == 5135 and len(frame.iloc[CROP_START:CROP_START+r.LENGTH]) == r.LENGTH
    full = frame.value.to_numpy(float)
    observed = full[CROP_START:CROP_START+r.LENGTH].copy()
    timestamps = frame.timestamp.to_numpy()
    assert np.isfinite(observed).all()
    with (r.ROOT / "data/anomalies.csv").open(newline="") as f:
        labels = {name: ast.literal_eval(events) for name, events in csv.reader(f) if name != "signal"}
    assert len(labels["A-6"]) == 1
    label_start, label_end = labels["A-6"][0]
    assert int(np.searchsorted(timestamps,label_start)) == ORIGINAL_TARGET[0]
    assert int(np.searchsorted(timestamps,label_end,side="right")-1) == ORIGINAL_TARGET[1]
    motif = observed[TARGET[0]:TARGET[1]+1].copy()
    assert len(motif) == 41
    master = observed.copy()
    intervals = []
    for k in INJECTION_K:
        start = k*r.STEP+8
        end = start+len(motif)-1
        assert end < r.LENGTH and (end < TARGET[0]-r.WINDOW or start > TARGET[1]+r.WINDOW)
        master[start:end+1] = motif
        intervals.append((start,end))
    target_windows = covered(TARGET)
    background_windows = [covered(x) for x in intervals]
    assert all(len(x)==4 for x in background_windows)
    assert len(set(sum(background_windows,[])))==32
    assert not (set(target_windows)&set(sum(background_windows,[])))

    master_proc = preprocess_time_series(master)
    scale = float(np.ptp(detrend(master)))
    target_proc = master_proc - (master-observed)/scale
    model = ViT4TS(window_size=r.WINDOW,window_step_ratio=4,device="cpu",
                   batch_size=8,standardize=True,verbose=False)
    with tempfile.TemporaryDirectory() as temp:
        target_img = r.images(target_proc,temp,"real_target")
        master_img = r.images(master_proc,temp,"real_master")
        target_feat = r.encode(model,target_img)
        master_feat = r.encode(model,master_img)
        direct20_raw = observed.copy()
        selected20 = sorted(set(sum(background_windows[:4],[])))
        for s,e in intervals[:4]: direct20_raw[s:e+1]=motif
        direct20_proc = master_proc-(master-direct20_raw)/scale
        direct20_img = r.images(direct20_proc,temp,"real_direct20")
        assembled20 = target_img.copy(); assembled20[selected20]=master_img[selected20]
        max_image_diff=float(np.max(np.abs(direct20_img-assembled20)))
        if max_image_diff>1e-6:raise AssertionError(max_image_diff)

    # Reference windows touching the real labeled target are replaced by
    # directly observed, label-normal donor windows eight strides earlier.
    # This builds a pure-normal external pool without inventing a time series.
    donors=[i-8 for i in target_windows]
    assert all(i>=0 and i not in target_windows and i not in sum(background_windows,[]) for i in donors)
    pure_features=[x.clone() for x in target_feat]
    for scale_index in range(3):
        pure_features[scale_index][target_windows]=target_feat[scale_index][donors]
    pure_features=tuple(pure_features)
    pure_mem=r.median_memory(pure_features)
    fixed_score=r.score(model,target_feat,pure_mem)
    target_proto=tuple(x[target_windows].mean(dim=0) for x in target_feat)
    normal_proto=tuple(x[donors].mean(dim=0) for x in target_feat)

    summary=[];details=[];similarities=[];scores={}
    for level in LEVELS:
        n={0:0,5:1,10:2,20:4,40:8}[level]
        selected=sorted(set(sum(background_windows[:n],[])))
        self_features=[x.clone() for x in target_feat]
        external_features=[x.clone() for x in pure_features]
        for i in range(3):
            self_features[i][selected]=master_feat[i][selected]
            external_features[i][selected]=master_feat[i][selected]
        self_mem=r.median_memory(tuple(self_features))
        external_mem=r.median_memory(tuple(external_features))
        for mode,mem,score_vec,base_count in (
            ("self_reference",self_mem,r.score(model,target_feat,self_mem),len(target_windows)),
            ("external_reference",external_mem,r.score(model,target_feat,external_mem),0),
            ("fixed",pure_mem,fixed_score,0),
        ):
            scores[(level,mode)]=score_vec
            affected=len(selected) if mode!="fixed" else 0
            evaluations=[r.metric(model,score_vec,[TARGET],a) for a in ALPHAS]
            best=max(evaluations,key=lambda x:x["f1"])
            target_mean=float(score_vec[TARGET[0]:TARGET[1]+1].mean())
            outside=np.r_[score_vec[:TARGET[0]],score_vec[TARGET[1]+1:]]
            summary.append(dict(level_percent=level,mode=mode,
                background_contaminated_windows=affected,
                target_contaminated_windows=base_count,
                total_contaminated_windows=affected+base_count,
                actual_total_window_percent=100*(affected+base_count)/r.N_WINDOWS,
                target_mean_score=target_mean,
                target_max_score=float(score_vec[TARGET[0]:TARGET[1]+1].max()),
                target_percentile_rank=float(100*(outside<target_mean).mean()),
                f1_max=best["f1"],best_alpha=best["alpha"],best_target_hit=best["target_hit"],
                best_TP=best["TP"],best_FP=best["FP"],best_FN=best["FN"]))
            for item in evaluations:
                details.append(dict(level_percent=level,mode=mode,
                    **{**item,"detected_intervals":json.dumps(item["detected_intervals"])}))
            for i,scale_name in enumerate(("large","mid","patch")):
                similarities.append(dict(level_percent=level,mode=mode,scale=scale_name,
                    reference_vs_target_cosine=r.cosine_to_reference(mem[i],target_proto[i]),
                    reference_vs_normal_cosine=r.cosine_to_reference(mem[i],normal_proto[i]),
                    reference_vs_pure_cosine=r.cosine_to_reference(mem[i],pure_mem[i])))

    save_csv(OUT/"summary.csv",summary)
    save_csv(OUT/"threshold_details.csv",details)
    save_csv(OUT/"reference_similarity.csv",similarities)
    np.savez_compressed(OUT/"raw_scores.npz",**{f"{mode}_{level}":x for (level,mode),x in scores.items()})
    plot(observed,intervals,summary,similarities,scores)
    with (OUT/"metadata.json").open("w") as f:
        json.dump(dict(source=str(SOURCE),crop_start=CROP_START,crop_length=r.LENGTH,
            original_target_indices=ORIGINAL_TARGET,target_interval=TARGET,
            target_timestamps=[int(timestamps[x]) for x in ORIGINAL_TARGET],
            target_windows=target_windows,donor_windows=donors,
            injection_intervals=intervals,injection_windows=background_windows,
            sequence_level_image_max_abs_diff_20pct=max_image_diff,
            normalization="source detrend/minmax once on master, then fixed affine map",
            checkpoint=str(model.model.checkpoint),seed=17),f,indent=2)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print("image verification max diff",max_image_diff,flush=True)


if __name__=="__main__":run()
