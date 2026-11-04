#!/usr/bin/env python3
"""Train one APRA-only loss-ablation checkpoint without evaluation labels."""
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
    APRAACEF, LossWeights, detection_loss, explanation_loss, gradient_norms,
    load_numeric_from_full, path_loss, reversible_warp,
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
    model = APRAACEF("apra", "apra_head", True).to(device)
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
        warped_n,target=reversible_warp(nq); warped_v=target@vq
        warped=model.forward_tokens(warped_n,warped_v,nr,vr)
        clean_base=base_proxy(nq,vq,nr,vr); synthetic_base=base_proxy(ns,vs,nr,vr)
        return clean,synthetic,warped,target,clean_base,synthetic_base

    def losses(indices):
        clean,synthetic,warped,target,clean_base,synthetic_base=batch_outputs(indices)
        zero=clean["delta"].sum()*0
        parts={
            "path": path_loss(warped,target) if spec["path"] else zero,
            "explain": zero,
            "detect": detection_loss(clean,synthetic,warped,clean_base,synthetic_base,
                                     altered[indices],weights) if spec["detect"] else zero,
        }
        total=weights.path*parts["path"]+weights.explain*parts["explain"]+weights.detect*parts["detect"]
        return total,parts,(clean,synthetic,warped)

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
    if spec["detect"]: probe_parts["detect"].backward()
    detect_after_one_step=gradient_norms(model)
    model.load_state_dict(saved); optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"])
    individual={}
    for loss_name in ("path","explain","detect"):
        model.zero_grad(set_to_none=True); _,parts,_=losses(first)
        if bool(spec.get(loss_name,False)):
            parts[loss_name].backward(); individual[loss_name]=gradient_norms(model)
        else: individual[loss_name]=gradient_norms(model)
    model.zero_grad(set_to_none=True)
    if spec["detect"] and detect_after_one_step["apra"]<=0:
        raise RuntimeError("L_detect does not update APRA after zero-head first step")
    if spec["detect"] and detect_after_one_step["acef"]<=0:
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
    numeric_change=max(float((value.detach().cpu()-initial_numeric[name]).abs().max())
                       for name,value in model.numeric.state_dict().items())
    payload={
        "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "model_config":{"aligner":"apra","fusion":"apra_head",
                        "reject":True,"hidden":192,"heads":4,"layers":2},
        "dataset":cached["dataset"],"variant":args.variant,"seed":seed,"history":history,
        "gradient_audit_by_loss":individual,"detect_gradient_after_one_step":detect_after_one_step,
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
        "active_losses":[name for name in ("path","detect") if spec[name]],
        "initialization_sha256":initial_hash.hexdigest(),"synthetic_cache_seed":cached["seed"],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); temp=args.output.with_suffix(".tmp")
    torch.save(payload,temp); temp.replace(args.output)
    print(json.dumps({"output":str(args.output),"elapsed_seconds":elapsed,
                      "updates":updates,"reference_effect":reference_effect}),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
