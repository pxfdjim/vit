"""Single-factor diagnostic for the harmonic-aggregation numerical instability.

The released multi-scale fusion computes, per output patch, the harmonic mean
k / sum_j (1/sim_j) over the scales covering it.  When a test patch is nearly
identical to its reference match, sim_j ~ 0 and float32 rounding can make
individual 1/sim_j terms carry either sign, so the sum can cross zero and the
quotient explodes to ~ +-2^31..2^32.  Five SMAP baselines are contaminated
this way (artifact_scan.csv); their Gaussian thresholds are destroyed.

The only changed factor here: clamp the per-scale dissimilarity to >= 1e-12
before the reciprocal, which makes every denominator term positive and bounds
the harmonic mean by 1.  Nothing else is touched (same weights, windows,
images, stitching, EWMA, alphas).

Runs all 53 SMAP sequences with the stabilized fusion, verifies that
artifact-free sequences reproduce the stored baseline, and reports the
per-sequence / dataset effect.

Outputs: data/runs/vit4ts_failure_audit/artifact_diagnostic.csv/.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/models/lib"))
import models.vit4ts as vit
from models.model_utils import compute_detection_intervals
from evaluation.evaluate import evaluate_intervals
from failure_audit import ALPHAS, load_index_and_labels, load_scores

OUT = ROOT / "data/runs/vit4ts_failure_audit"


def stabilized_harmonic(score_size, similarity, mask):
    b, h, w = score_size
    similarity = similarity.double().clamp(min=1e-12)
    mask = mask.T.long()
    score = torch.zeros((b, h * w), device=similarity.device).double()
    for idx in range(h * w):
        patch_idx = torch.tensor(
            [bool(torch.isin(idx + 1, m)) for m in mask],
            device=similarity.device)
        count = patch_idx.sum().item()
        denom = torch.sum(1.0 / similarity[:, patch_idx], dim=-1)
        score[:, idx] = count / denom
    return score.view(b, h, w)


def f1_table(score, ts, truth):
    out = {}
    for a in ALPHAS:
        intervals, _, _ = compute_detection_intervals(score, alpha=a)
        predicted = [[int(ts[i]), int(ts[j])] for i, j in intervals]
        out[a] = evaluate_intervals(truth, predicted)["F1"]
        out[f"pred_{a}"] = predicted
    return out


def main():
    if os.getenv("CLIP_PRETRAINED") != "openai":
        raise RuntimeError("Set CLIP_PRETRAINED=openai")
    index, labels = load_index_and_labels()
    model = vit.ViT4TS(window_size=224, window_step_ratio=4.0,
                       image_size=(224, 224), verbose=False)
    assert model.model.pretrained_tag == "openai"
    original = vit.harmonic_aggregation
    vit.harmonic_aggregation = stabilized_harmonic

    rows = []
    try:
        for signal in index["SMAP"]:
            top, q25, ts, _ = load_scores("SMAP", signal)
            truth = [(int(a), int(b)) for a, b in labels[signal]]
            frame = pd.read_csv(ROOT / "data" / "SMAP" / (signal + ".csv"))
            frame = frame.sort_values("timestamp")
            clean, observed = model.predict_scores(frame)
            if not np.array_equal(observed, ts):
                raise RuntimeError(f"timestamp mismatch {signal}")
            stored_artifact = bool(np.max(top) > 10)
            clean_artifact = bool(np.max(clean) > 10)
            diff = float(np.max(np.abs(clean - top)))
            if not stored_artifact and not clean_artifact and diff > 1e-5:
                raise RuntimeError(f"stabilized rerun changed a clean "
                                   f"sequence: {signal}, diff={diff}")
            clean_f1 = f1_table(clean, ts, truth)
            stored_f1 = f1_table(top, ts, truth)
            rows.append({
                "signal": signal,
                "stored_artifact": int(stored_artifact),
                "clean_artifact": int(clean_artifact),
                "max_abs_diff_vs_stored": diff,
                "stored_f1max": max(stored_f1[a] for a in ALPHAS),
                "clean_f1max": max(clean_f1[a] for a in ALPHAS),
                **{f"clean_f1_alpha_{a}": clean_f1[a] for a in ALPHAS},
                **{f"stored_f1_alpha_{a}": stored_f1[a] for a in ALPHAS}})
            print("DONE", signal, "stored_artifact", stored_artifact,
                  "f1max", round(rows[-1]["clean_f1max"], 4), flush=True)
    finally:
        vit.harmonic_aggregation = original

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "artifact_diagnostic.csv", index=False)
    art = df[df.stored_artifact == 1]
    clean_only = df[df.stored_artifact == 0]
    means_stored = {a: df[f"stored_f1_alpha_{a}"].mean() for a in ALPHAS}
    means_clean = {a: df[f"clean_f1_alpha_{a}"].mean() for a in ALPHAS}
    summary = {
        "changed_factor": "clamp per-scale dissimilarity >= 1e-12 before the "
                          "harmonic reciprocal (numerical stabilization only)",
        "sequences_with_stored_artifacts": int(df.stored_artifact.sum()),
        "clean_rerun_artifact_free": int(df.clean_artifact.sum()),
        "clean_sequences_unchanged_max_diff": float(
            clean_only.max_abs_diff_vs_stored.max()),
        "artifact_sequences": {
            r.signal: {"stored_f1max": round(r.stored_f1max, 4),
                       "clean_f1max": round(r.clean_f1max, 4)}
            for _, r in art.iterrows()},
        "SMAP_dataset_f1max": {
            "stored": round(max(means_stored.values()), 4),
            "stabilized": round(max(means_clean.values()), 4),
            "best_alpha_stored": max(means_stored, key=means_stored.get),
            "best_alpha_stabilized": max(means_clean, key=means_clean.get)},
    }
    (OUT / "artifact_diagnostic.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
