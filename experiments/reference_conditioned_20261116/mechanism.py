"""Unlabeled time-warp and local-injection mechanism diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reference_conditioned_20261116.model import load_model_checkpoint
from src.core.scoring import final_score
from src.models.model import reversible_warp

DATASETS = ("MSL", "SMAP", "realTraffic", "YAHOOA1")
MODES = ("posterior", "no_reference", "t3_direct")


@torch.inference_mode()
def encode(model, raw, visual, device, batch=128):
    numeric, projected = [], []
    for start in range(0, len(raw), batch):
        numeric.append(model.numeric(raw[start:start + batch].to(device)))
        projected.append(model.visual(visual[start:start + batch].float().to(device)))
    return torch.cat(numeric), torch.cat(projected)


@torch.inference_mode()
def run_windows(model, nq, vq, nr, vr, ids, anchor, batch=96):
    result = {name: [] for name in ("delta", "confidence", "joint_norm", "posterior",
                                   "prior", "numeric_h", "visual_h")}
    for start in range(0, len(nq), batch):
        stop = min(start + batch, len(nq))
        output = model.forward_tokens(nq[start:stop], vq[start:stop],
                                      nr[ids[start:stop]], vr[ids[start:stop]],
                                      anchor[start:stop])
        rep = output["representation"]
        result["delta"].append(output["delta"])
        result["confidence"].append(rep["match_confidence"])
        result["joint_norm"].append(rep["joint"].norm(dim=-1))
        result["numeric_h"].append(rep["numeric"])
        result["visual_h"].append(rep["visual"])
        result["posterior"].append(output["alignment"]["soft_match"])
        result["prior"].append(output["prior"])
    return {key: torch.cat(value) for key, value in result.items()}


def masked_mean(tensor, mask):
    return float(tensor[mask].mean().cpu()) if bool(mask.any()) else float("nan")


@torch.inference_mode()
def diagnostic_signal(model, cache, indices, frozen_path, device, figure_sample=False):
    raw = cache["raw"][indices]
    synthetic_raw = cache["synthetic_raw"][indices]
    visual = cache["visual"][indices]
    synthetic_visual = cache["synthetic_visual"][indices]
    numeric, visual_tokens = encode(model, raw, visual, device)
    synthetic_numeric, synthetic_visual_tokens = encode(model, synthetic_raw, synthetic_visual, device)
    with np.load(frozen_path, allow_pickle=False) as stored:
        if not np.array_equal(stored["source_indices"], indices):
            raise RuntimeError("training source indices changed")
        row = {key: torch.from_numpy(stored[key].copy()).to(device)
               for key in ("clean_score", "synthetic_score", "point_mask",
                           "clean_reference_ids", "clean_reference_columns",
                           "synthetic_reference_ids", "synthetic_reference_columns")}
    clean_ids = row["clean_reference_ids"].long()
    synthetic_ids = row["synthetic_reference_ids"].long()
    clean_anchor = row["clean_reference_columns"].float()
    synthetic_anchor = row["synthetic_reference_columns"].float()
    clean = run_windows(model, numeric, visual_tokens, numeric, visual_tokens,
                        clean_ids, clean_anchor)
    synthetic = run_windows(model, synthetic_numeric, synthetic_visual_tokens,
                            synthetic_numeric, synthetic_visual_tokens,
                            synthetic_ids, synthetic_anchor)
    clean_score, _ = final_score(clean["delta"], row["clean_score"])
    synthetic_score, _ = final_score(synthetic["delta"], row["synthetic_score"])
    hn, hv = synthetic["numeric_h"], synthetic["visual_h"]
    zero_n = torch.zeros_like(hn)
    zero_v = torch.zeros_like(hv)
    without_numeric = torch.tanh(model.fusion.out(model.fusion.joint(
        torch.cat((zero_n, hv, hv.abs()), dim=-1))).squeeze(-1))
    without_visual = torch.tanh(model.fusion.out(model.fusion.joint(
        torch.cat((hn, zero_v, hn.abs()), dim=-1))).squeeze(-1))
    point_mask = row["point_mask"].bool()
    patch_mask = cache["patch_mask"][indices].bool().to(device)
    clean_normal = ~patch_mask
    injection = patch_mask
    chosen = torch.nonzero(injection.any(-1), as_tuple=False).flatten()
    chosen = chosen[0].item() if chosen.numel() else 0
    sample = torch.tensor([chosen], device=device)
    warped_n, target = reversible_warp(numeric[sample],
                                       numeric.new_tensor([1.0]))
    warped_v = target @ visual_tokens[sample]
    warped_anchor = (target @ clean_anchor[sample, :, None]).squeeze(-1)
    warped = model.forward_tokens(warped_n, warped_v,
                                  numeric[clean_ids[sample]], visual_tokens[clean_ids[sample]],
                                  warped_anchor)
    match = warped["alignment"]["soft_match"]
    posterior_columns = (match * torch.arange(14, device=device)[None, None]).sum(-1) / \
        match.sum(-1).clamp_min(1e-8)
    expected_delta = (target @ clean["delta"][sample, :, None]).squeeze(-1)
    result = {
        "windows": len(indices), "injected_tokens": int(injection.sum()),
        "match_confidence_normal": masked_mean(clean["confidence"], clean_normal),
        "match_confidence_clean_at_injection": masked_mean(clean["confidence"], injection),
        "match_confidence_injected": masked_mean(synthetic["confidence"], injection),
        "confidence_change_injected": masked_mean(synthetic["confidence"] -
                                                  clean["confidence"], injection),
        "joint_norm_normal": masked_mean(clean["joint_norm"], clean_normal),
        "joint_norm_clean_at_injection": masked_mean(clean["joint_norm"], injection),
        "joint_norm_injected": masked_mean(synthetic["joint_norm"], injection),
        "residual_norm_change_injected": masked_mean(synthetic["joint_norm"] -
                                                     clean["joint_norm"], injection),
        "delta_normal_abs": masked_mean(clean["delta"].abs(), clean_normal),
        "delta_injected_abs": masked_mean(synthetic["delta"].abs(), injection),
        "raw_delta_change_injected": masked_mean(synthetic["delta"] -
                                                 clean["delta"], injection),
        "numeric_view_removal_effect": masked_mean((synthetic["delta"] - without_numeric).abs(), injection),
        "visual_view_removal_effect": masked_mean((synthetic["delta"] - without_visual).abs(), injection),
        "final_score_injection_gain": masked_mean(
            (synthetic_score - clean_score) - (row["synthetic_score"] - row["clean_score"]), point_mask),
        "final_score_unchanged_shift_abs": masked_mean(
            ((synthetic_score - clean_score) -
             (row["synthetic_score"] - row["clean_score"])).abs(), ~point_mask),
        "normal_warp_alignment_error": float((posterior_columns - warped_anchor).abs().mean()),
        "normal_warp_delta_error": float((warped["delta"] - expected_delta).abs().mean()),
        "normal_warp_match_confidence": float(warped["representation"]["match_confidence"].mean()),
    }
    perturbations = json.loads(cache["perturbations_json"])
    kinds = []
    for local_index, global_index in enumerate(indices):
        token_mask = injection[local_index]
        if not bool(token_mask.any()):
            continue
        kinds.append({"kind": perturbations[global_index]["kind"],
                      "confidence_change": float((synthetic["confidence"][local_index, token_mask] -
                                                   clean["confidence"][local_index, token_mask]).mean()),
                      "residual_norm_change": float((synthetic["joint_norm"][local_index, token_mask] -
                                                     clean["joint_norm"][local_index, token_mask]).mean()),
                      "delta_change": float((synthetic["delta"][local_index, token_mask] -
                                             clean["delta"][local_index, token_mask]).mean())})
    if not figure_sample:
        return result, None, kinds
    sample_data = {
        "patch_mask": patch_mask[chosen].cpu().numpy(),
        "prior_clean": clean["prior"][chosen].cpu().numpy(),
        "posterior_clean": clean["posterior"][chosen].cpu().numpy(),
        "posterior_synthetic": synthetic["posterior"][chosen].cpu().numpy(),
        "posterior_warped": warped["alignment"]["soft_match"][0].cpu().numpy(),
        "confidence_clean": clean["confidence"][chosen].cpu().numpy(),
        "confidence_synthetic": synthetic["confidence"][chosen].cpu().numpy(),
        "joint_norm_clean": clean["joint_norm"][chosen].cpu().numpy(),
        "joint_norm_synthetic": synthetic["joint_norm"][chosen].cpu().numpy(),
        "delta_clean": clean["delta"][chosen].cpu().numpy(),
        "delta_synthetic": synthetic["delta"][chosen].cpu().numpy(),
        "delta_warped": warped["delta"][0].cpu().numpy(),
        "anchor_clean": clean_anchor[chosen].cpu().numpy(),
        "anchor_warped": warped_anchor[0].cpu().numpy(),
    }
    return result, sample_data, kinds


def draw(sample, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    navy, teal, coral = "#18324F", "#187F83", "#C7674B"
    cmap = LinearSegmentedColormap.from_list("reference", ["#F6F8FA", "#9AC9C5", navy])
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), constrained_layout=True)
    for ax, key, title, anchor in (
        (axes[0, 0], "prior_clean", "T3 position prior", sample["anchor_clean"]),
        (axes[0, 1], "posterior_synthetic", "Posterior · local injection", sample["anchor_clean"]),
        (axes[0, 2], "posterior_warped", "Posterior · normal time warp", sample["anchor_warped"])):
        heat = ax.imshow(sample[key], origin="lower", aspect="equal", cmap=cmap, vmin=0, vmax=1)
        ax.plot(anchor, np.arange(14), color=coral, lw=1.3, alpha=.9)
        ax.set(title=title, xlabel="Reference position", ylabel="Query position",
               xticks=[0, 6, 13], yticks=[0, 6, 13])
    fig.colorbar(heat, ax=axes[0], location="right", shrink=.76, pad=.015,
                 label="Probability mass")
    x = np.arange(14)
    injected = sample["patch_mask"]
    for ax in axes[1]:
        ax.fill_between(x, 0, 1, where=injected, transform=ax.get_xaxis_transform(),
                        color="#F7DDD4", alpha=.65, step="mid", zorder=0)
        ax.set(xlabel="Query position", xlim=(0, 13), xticks=[0, 3, 6, 9, 13])
    axes[1, 0].plot(x, sample["confidence_clean"], color=teal, lw=2, label="Clean")
    axes[1, 0].plot(x, sample["confidence_synthetic"], color=coral, lw=2, label="Injected")
    axes[1, 0].set(title="Match confidence", ylabel="Probability", ylim=(0, 1.02))
    axes[1, 1].plot(x, sample["joint_norm_clean"], color=teal, lw=2)
    axes[1, 1].plot(x, sample["joint_norm_synthetic"], color=coral, lw=2)
    axes[1, 1].set(title="Learned two-view residual", ylabel="Feature norm")
    axes[1, 2].plot(x, sample["delta_clean"], color=teal, lw=2)
    axes[1, 2].plot(x, sample["delta_synthetic"], color=coral, lw=2)
    axes[1, 2].plot(x, sample["delta_warped"], color=navy, lw=1.5, ls="--")
    axes[1, 2].set(title="Bounded anomaly correction", ylabel="Raw delta")
    axes[1, 0].legend(loc="lower left", ncol=2, frameon=False, fontsize=8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((ROOT / "configs/model.json").read_text())
    device = torch.device(args.device)
    rows, kind_rows = [], []
    first_sample = None
    for dataset in args.datasets:
        cache_path = ROOT / config["synthetic_training_cache_root"] / f"{dataset}.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        records = json.loads(cache["records_json"])
        groups = {}
        for index, record in enumerate(records):
            groups.setdefault(record["signal"], []).append(index)
        for mode in args.modes:
            model, payload = load_model_checkpoint(run / "checkpoints" / mode /
                                                   f"{dataset}.pt", device)
            model.eval()
            for signal, indices in groups.items():
                result, sample, kinds = diagnostic_signal(
                    model, cache, indices,
                    ROOT / config["frozen_b3_root"] / dataset / f"{signal}.npz",
                    device, figure_sample=(mode == "posterior" and dataset == "MSL"
                                           and first_sample is None))
                rows.append({"dataset": dataset, "signal": signal,
                             "variant": mode, **result})
                kind_rows.extend({"dataset": dataset, "signal": signal,
                                  "variant": mode, **kind} for kind in kinds)
                if sample is not None and first_sample is None:
                    first_sample = sample
                    np.savez_compressed(run / "mechanism_example.npz", **sample)
            print(json.dumps({"dataset": dataset, "mode": mode,
                              "signals": len(groups)}), flush=True)
    pd.DataFrame(rows).to_csv(run / "mechanism_diagnostics.csv", index=False)
    pd.DataFrame(kind_rows).to_csv(run / "mechanism_by_perturbation.csv", index=False)
    if first_sample is not None:
        draw(first_sample, run / "figures" / "posterior_reference_mechanism")


if __name__ == "__main__":
    main()
