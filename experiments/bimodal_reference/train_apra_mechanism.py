#!/usr/bin/env python3
"""Train one detection-only APRA mechanism ablation without labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.models.apra_acef import (
    APRAACEF, LossWeights, detection_loss, gradient_norms,
    load_numeric_from_full, reversible_warp,
)


def load_frozen_visual(model: APRAACEF, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {name.removeprefix("visual."): value for name, value in payload["state_dict"].items()
             if name.startswith("visual.")}
    model.visual.load_state_dict(state, strict=True)
    for parameter in model.visual.parameters(): parameter.requires_grad_(False)


@torch.no_grad()
def projected_visual(model, visual, device, batch_size=256):
    return torch.cat([model.visual(visual[start:start+batch_size].float().to(device)).half()
                      for start in range(0, len(visual), batch_size)])


def base_proxy(nq, vq, nr, vr):
    numeric = 1-F.cosine_similarity(nq, nr, dim=-1)
    visual = 1-F.cosine_similarity(vq, vr, dim=-1)
    return .5*numeric.detach()+.5*visual.detach()


import torch.nn.functional as F


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config = json.loads(args.config.read_text()); spec = config["variants"][args.variant]
    seed = int(args.seed); random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cached["labels_read"]: raise RuntimeError("cache integrity")
    model = APRAACEF("apra", "apra_head", spec["reject"], skip=spec["skip"],
                     structural_residual=spec["structural_residual"]).to(device)
    full_checkpoint = Path(config["full_checkpoint_root"]) / f"{cached['dataset']}.pt"
    load_numeric_from_full(model, str(full_checkpoint), random_numeric=False)
    rcca_checkpoint = (Path(config["rcca_run"])/"checkpoints/M1_full"/
                       f"seed_{seed}"/f"{cached['dataset']}.pt")
    load_frozen_visual(model, rcca_checkpoint)
    initial_numeric = {name: value.detach().cpu().clone()
                       for name,value in model.numeric.state_dict().items()}
    raw = cached["raw"].to(device); synthetic_raw = cached["synthetic_raw"].to(device)
    altered = cached["patch_mask"].to(device); reference_ids = cached["reference_ids"].to(device)
    reference_columns = cached["reference_columns"].to(device)
    visual = projected_visual(model, cached["visual"], device)
    synthetic_visual = projected_visual(model, cached["synthetic_visual"], device)
    weights = LossWeights(**config["loss_weights"])
    perturbations = json.loads(cached["perturbations_json"])
    reject_kinds = set(config["selective_supervision"]["reject_perturbations"])
    structural_rows = torch.tensor([row["kind"] in reject_kinds for row in perturbations],
                                   device=device, dtype=torch.bool)
    structural_reject = altered & structural_rows[:, None]
    initial_hash=hashlib.sha256()
    for name,value in model.state_dict().items():
        initial_hash.update(name.encode()); initial_hash.update(value.detach().cpu().numpy().tobytes())
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(torch.arange(len(raw)), batch_size=config["training"]["batch_size"],
                        shuffle=True, generator=generator, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

    def batch_outputs(indices):
        ids = reference_ids[indices]; columns = reference_columns[indices][...,None]
        nq=model.numeric(raw[indices]); ns=model.numeric(synthetic_raw[indices])
        nr_full=model.numeric(raw[ids]); nr=nr_full.gather(1,columns.expand(-1,-1,nr_full.shape[-1]))
        vq=visual[indices].float(); vs=synthetic_visual[indices].float()
        vr_full=visual[ids].float(); vr=vr_full.gather(1,columns.expand(-1,-1,vr_full.shape[-1]))
        clean=model.forward_tokens(nq,vq,nr,vr); synthetic=model.forward_tokens(ns,vs,nr,vr)
        warped_n,target=reversible_warp(nq)
        warped_v=target@vq
        warped=model.forward_tokens(warped_n,warped_v,nr,vr)
        clean_base=base_proxy(nq,vq,nr,vr); synthetic_base=base_proxy(ns,vs,nr,vr)
        return clean,synthetic,warped,target,clean_base,synthetic_base

    def losses(indices):
        clean,synthetic,warped,target,clean_base,synthetic_base=batch_outputs(indices)
        detect = detection_loss(clean, synthetic, warped, clean_base, synthetic_base,
                                altered[indices], weights)
        return weights.detect * detect, {"detect": detect}, (clean, synthetic, warped)

    # The signed correction is exactly zero before optimization.
    correction_names={"fusion.correction.weight","fusion.correction.bias",
                      "fusion.out.weight","fusion.out.bias"}
    correction_parameters=[p for name,p in model.named_parameters() if name in correction_names]
    zero_initialized=max((float(p.detach().abs().max()) for p in correction_parameters),default=0.0)==0.0
    first=next(iter(DataLoader(torch.arange(len(raw)),batch_size=config["training"]["batch_size"],
                               shuffle=False,drop_last=True))).to(device)
    # Probe one update then audit L_detect so upstream gradients are measured after
    # the mathematically unavoidable zero-head first step. Restore all weights.
    saved=copy.deepcopy(model.state_dict()); model.train(); probe,_,_=losses(first)
    optimizer.zero_grad(set_to_none=True); probe.backward(); optimizer.step()
    _,probe_parts,_=losses(first); model.zero_grad(set_to_none=True)
    probe_parts["detect"].backward()
    detect_after_one_step=gradient_norms(model)
    model.load_state_dict(saved); optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"])
    individual={}
    for loss_name in ("detect",):
        model.zero_grad(set_to_none=True); _,parts,_=losses(first)
        parts[loss_name].backward(); individual[loss_name]=gradient_norms(model)
    model.zero_grad(set_to_none=True)
    if detect_after_one_step["apra"]<=0 and spec["reject"] and spec["skip"]:
        raise RuntimeError("L_detect does not update APRA after zero-head first step")
    if detect_after_one_step["acef"]<=0:
        raise RuntimeError("L_detect does not update fusion after zero-head first step")

    history=[]; updates=0; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(config["training"]["epochs"]):
        model.train(); sums={}; batches=0
        for indices in loader:
            indices=indices.to(device); total,parts,_=losses(indices)
            optimizer.zero_grad(set_to_none=True); total.backward(); norms=gradient_norms(model)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           config["training"]["gradient_clip"])
            optimizer.step(); updates+=1; batches+=1
            for name,value in {"total":total,**parts}.items(): sums[name]=sums.get(name,0)+float(value.detach())
            for name,value in norms.items(): sums[f"grad_{name}"]=sums.get(f"grad_{name}",0)+value
        row={"epoch":epoch+1,**{k:v/batches for k,v in sums.items()}}
        history.append(row); print(json.dumps(row),flush=True)
    elapsed=time.perf_counter()-started
    model.eval()
    with torch.inference_mode():
        clean,_,_,_,_,_=batch_outputs(first); base=clean["delta"]
        nq=clean["numeric_query"]; vq=clean["visual_query"]
        nr=clean["numeric_reference"].roll(1,0).flip(1)
        vr=clean["visual_reference"].roll(1,0).flip(1)
        replaced=model.forward_tokens(nq,vq,nr,vr)["delta"]
        reference_effect=float((base-replaced).abs().mean())

    # Full-cache, label-free mechanism audit at the immutable final epoch.
    audit_loader=DataLoader(torch.arange(len(raw)),batch_size=128,shuffle=False,num_workers=0)
    audit_sums={}; audit_counts={}
    def add(name,value,mask=None):
        values=value if mask is None else value[mask]
        if values.numel()==0: return
        audit_sums[name]=audit_sums.get(name,0.0)+float(values.double().sum())
        audit_counts[name]=audit_counts.get(name,0)+values.numel()
    model.eval()
    with torch.inference_mode():
        for audit_indices in audit_loader:
            audit_indices=audit_indices.to(device)
            clean_a,synth_a,warp_a,target_a,_,_=batch_outputs(audit_indices)
            for prefix,out_a in (("clean",clean_a),("synthetic",synth_a),("warp",warp_a)):
                alignment=out_a["alignment"]; soft=alignment["soft_match"]
                normalized=soft/soft.sum(-1,keepdim=True).clamp_min(1e-8)
                entropy=-(normalized.clamp_min(1e-8)*normalized.clamp_min(1e-8).log()).sum(-1)/np.log(14)
                index=torch.arange(14,device=device,dtype=soft.dtype)
                displacement=((normalized*index[None,None]).sum(-1)-index[None]).abs()
                add(f"{prefix}_path_entropy",entropy); add(f"{prefix}_match_probability",soft.sum(-1))
                add(f"{prefix}_rejection",alignment["rejection"])
                add(f"{prefix}_reference_skip",alignment["reference_skip"])
                add(f"{prefix}_displacement",displacement)
                residual=(alignment["structural_numeric"].norm(dim=-1)+
                          alignment["structural_visual"].norm(dim=-1))
                add(f"{prefix}_structural_residual",residual)
                row_error=(soft.sum(-1)+alignment["rejection"]-1).abs()
                column_error=(soft.sum(-2)+alignment["reference_skip"]-1).abs()
                add(f"{prefix}_row_conservation_error",row_error)
                add(f"{prefix}_column_conservation_error",column_error)
            structural=structural_reject[audit_indices]
            changed=altered[audit_indices]
            matchable=changed & ~structural
            unchanged=~changed
            add("synthetic_structural_rejection",synth_a["alignment"]["rejection"],structural)
            add("synthetic_matchable_rejection",synth_a["alignment"]["rejection"],matchable)
            add("synthetic_unchanged_rejection",synth_a["alignment"]["rejection"],unchanged)
            add("synthetic_structural_delta",synth_a["delta"],structural)
            add("synthetic_matchable_delta",synth_a["delta"],matchable)
            warp_mass=(target_a*warp_a["alignment"]["soft_match"]).sum(-1)
            add("warp_target_match_mass",warp_mass)
    mechanism_audit={name:audit_sums[name]/audit_counts[name] for name in audit_sums}
    numeric_change=max(float((value.detach().cpu()-initial_numeric[name]).abs().max())
                       for name,value in model.numeric.state_dict().items())
    payload={
        "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "model_config":{"aligner":"apra","fusion":"apra_head",
                        "reject":spec["reject"],"skip":spec["skip"],
                        "structural_residual":spec["structural_residual"],
                        "hidden":192,"heads":4,"layers":2},
        "dataset":cached["dataset"],"variant":args.variant,"seed":seed,"history":history,
        "gradient_audit_by_loss":individual,"detect_gradient_after_one_step":detect_after_one_step,
        "mechanism_audit":mechanism_audit,
        "zero_initialized_correction":zero_initialized,"reference_effect":reference_effect,
        "numeric_max_abs_change":numeric_change,
        "parameter_counts":{"all":sum(p.numel() for p in model.parameters()),
          "trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),
          "numeric":sum(p.numel() for p in model.numeric.parameters()),
          "apra":sum(p.numel() for p in model.aligner.parameters()),
          "acef":sum(p.numel() for p in model.fusion.parameters())},
        "elapsed_seconds":elapsed,"optimizer_updates":updates,
        "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(device),
        "labels_used":False,"validation_split_used":False,
        "checkpoint_selection":"fixed_final_epoch","source_cache":str(args.cache),
        "numeric_initialization":str(full_checkpoint),
        "visual_initialization":str(rcca_checkpoint),"loss_weights":config["loss_weights"],
        "active_losses":["detect"],
        "selective_supervision":config["selective_supervision"],
        "initialization_sha256":initial_hash.hexdigest(),"synthetic_cache_seed":cached["seed"],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); temp=args.output.with_suffix(".tmp")
    torch.save(payload,temp); temp.replace(args.output)
    print(json.dumps({"output":str(args.output),"elapsed_seconds":elapsed,
                      "updates":updates,"reference_effect":reference_effect}),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
