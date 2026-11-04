#!/usr/bin/env python3
"""Train only independent TREM while the validated APRA model stays frozen."""
from __future__ import annotations
import argparse, copy, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.models.apra_acef import LossWeights, detection_loss, reversible_warp
from src.models.apra_trem import b3_token_evidence
from src.models.apra_trem_independent import (FrozenAPRAIndependentTREM,
    load_apra_only, stage_gradient_norms)

@torch.no_grad()
def projected_visual(model, value, device, batch=256):
    return torch.cat([model.visual(value[i:i+batch].float().to(device)).half()
                      for i in range(0,len(value),batch)])

def reference_raw(raw, ids, columns):
    patches=raw[ids].reshape(len(ids),14,16)
    return patches.gather(1,columns[...,None].expand(-1,-1,16))

def snapshot(modules):
    return {f"{prefix}.{name}":value.detach().cpu().clone()
            for prefix,module in modules for name,value in module.state_dict().items()}

def max_change(modules, initial):
    values=[]
    for prefix,module in modules:
        values.extend(float((value.detach().cpu()-initial[f"{prefix}.{name}"]).abs().max())
                      for name,value in module.state_dict().items())
    return max(values,default=0.)

def norms(value):
    return {"mean":float(value.mean().detach()),"abs_mean":float(value.abs().mean().detach()),
            "std":float(value.std().detach()),"max_abs":float(value.abs().max().detach())}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True)
    p.add_argument("--variant",required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--cache",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0"); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    cfg=json.loads(a.config.read_text()); spec=cfg["variants"][a.variant]; seed=a.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device=torch.device(a.device); cached=torch.load(a.cache,map_location="cpu",weights_only=False)
    if cached["labels_read"]: raise RuntimeError("training cache opened labels")
    model=FrozenAPRAIndependentTREM(trem_width=cfg["trem"]["width"],**spec).to(device)
    apra_checkpoint=Path(cfg["apra_run"])/"checkpoints/APRA_only"/f"{cached['dataset']}.pt"
    load_apra_only(model,str(apra_checkpoint))
    base_modules=[("numeric",model.numeric),("visual",model.visual),("aligner",model.aligner),("apra_head",model.apra_head)]
    initial=snapshot(base_modules)
    raw=cached["raw"].to(device); synthetic_raw=cached["synthetic_raw"].to(device)
    altered=cached["patch_mask"].to(device); reference_ids=cached["reference_ids"].to(device)
    reference_columns=cached["reference_columns"].to(device)
    visual=projected_visual(model,cached["visual"],device)
    synthetic_visual=projected_visual(model,cached["synthetic_visual"],device)
    loss_cfg=cfg["loss"]
    weights=LossWeights(path=0.,explain=0.,detect=1.,detect_margin=loss_cfg["detect_margin"],gamma=loss_cfg["training_beta"])
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(torch.arange(len(raw)),batch_size=cfg["training"]["batch_size"],shuffle=True,
                      generator=generator,drop_last=True,num_workers=0)

    def batch_outputs(indices):
        ids=reference_ids[indices]; columns=reference_columns[indices]; expanded=columns[...,None]
        with torch.no_grad():
            nq=model.numeric(raw[indices]); ns=model.numeric(synthetic_raw[indices]); nr_full=model.numeric(raw[ids])
            nr=nr_full.gather(1,expanded.expand(-1,-1,nr_full.shape[-1]))
            vq=visual[indices].float(); vs=synthetic_visual[indices].float(); vr_full=visual[ids].float()
            vr=vr_full.gather(1,expanded.expand(-1,-1,vr_full.shape[-1]))
            rq=raw[indices].reshape(len(indices),14,16); rs=synthetic_raw[indices].reshape(len(indices),14,16)
            rr=reference_raw(raw,ids,columns)
            clean_base=b3_token_evidence(nq,vq,nr,vr,rq,rr); synth_base=b3_token_evidence(ns,vs,nr,vr,rs,rr)
            warped_n,target=reversible_warp(nq); warped_v=target@vq; warped_raw=target@rq
            warped_base=b3_token_evidence(warped_n,warped_v,nr,vr,warped_raw,rr)
        clean=model.forward_tokens(nq,vq,nr,vr,clean_base)
        synth=model.forward_tokens(ns,vs,nr,vr,synth_base)
        warped=model.forward_tokens(warped_n,warped_v,nr,vr,warped_base)
        return clean,synth,warped,clean_base,synth_base

    def loss(indices):
        clean,synth,warped,clean_base,synth_base=batch_outputs(indices)
        value=detection_loss(clean,synth,warped,clean_base,synth_base,altered[indices],weights)
        return value,(clean,synth,warped)

    trainable=list(model.trem.parameters())
    if any(p.requires_grad for name,p in model.named_parameters() if not name.startswith("trem.")):
        raise RuntimeError("base is not frozen")
    optimizer=torch.optim.AdamW(trainable,lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    first=next(iter(DataLoader(torch.arange(len(raw)),batch_size=cfg["training"]["batch_size"],drop_last=True))).to(device)
    model.eval()
    with torch.no_grad():
        initial_out=batch_outputs(first)[0]
        zero_error=float(initial_out["trem_raw"].abs().max())
    if zero_error!=0: raise RuntimeError("zero initialization did not preserve APRA")
    saved=copy.deepcopy(model.trem.state_dict()); model.train(); probe,_=loss(first)
    optimizer.zero_grad(set_to_none=True); probe.backward(); optimizer.step()
    second,_=loss(first); optimizer.zero_grad(set_to_none=True); second.backward()
    gradient_after_one_step=stage_gradient_norms(model)
    if gradient_after_one_step["output"]<=0 or gradient_after_one_step["projection"]<=0:
        raise RuntimeError(f"invalid TREM gradient path {gradient_after_one_step}")
    if spec["use_local"] and gradient_after_one_step["local"]<=0: raise RuntimeError("local gradient absent")
    if spec["use_multiscale"] and gradient_after_one_step["multiscale"]<=0: raise RuntimeError("context gradient absent")
    if spec["use_reliability"] and gradient_after_one_step["reliability_fusion"]<=0: raise RuntimeError("reliability gradient absent")
    model.trem.load_state_dict(saved); optimizer=torch.optim.AdamW(trainable,lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    history=[]; updates=0; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(cfg["training"]["epochs"]):
        model.train(); sums={}; batches=0
        for indices in loader:
            indices=indices.to(device); value,outputs=loss(indices)
            optimizer.zero_grad(set_to_none=True); value.backward(); gradient=stage_gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(trainable,cfg["training"]["gradient_clip"]); optimizer.step()
            updates+=1; batches+=1; sums["loss"]=sums.get("loss",0.)+float(value.detach())
            for name,item in gradient.items(): sums[f"grad_{name}"]=sums.get(f"grad_{name}",0.)+item
            for key in ("input_features","residual_projection","first_difference","second_difference","local_dynamics","multiscale_context","fused_features","pre_tanh","trem_raw"):
                sums[f"amp_{key}"]=sums.get(f"amp_{key}",0.)+float(outputs[0][key].abs().mean().detach())
        row={"epoch":epoch+1,**{k:v/batches for k,v in sums.items()}}; history.append(row); print(json.dumps(row),flush=True)
    elapsed=time.perf_counter()-started; model.eval()
    with torch.inference_mode():
        clean=batch_outputs(first)[0]; alignment={k:v for k,v in clean["alignment"].items()}
        if spec["input_mode"]=="apra":
            changed=dict(alignment); changed["structural_numeric"]=changed["structural_numeric"]+.1
            residual_changed=model.trem(changed,clean["apra_delta"],clean["b3_evidence"],clean["numeric_query"],clean["numeric_reference"])["trem_raw"]
        else:
            residual_changed=model.trem(alignment,clean["apra_delta"],clean["b3_evidence"],clean["numeric_query"]+.1,clean["numeric_reference"])["trem_raw"]
        input_effect=float((residual_changed-clean["trem_raw"]).abs().mean())
        changed=dict(alignment); changed["reliability"]=1-changed["reliability"]
        reliability_changed=model.trem(changed,clean["apra_delta"],clean["b3_evidence"],clean["numeric_query"],clean["numeric_reference"])["trem_raw"]
        reliability_effect=float((reliability_changed-clean["trem_raw"]).abs().mean())
        stage_amplitudes={key:norms(clean[key]) for key in ("input_features","residual_projection","first_difference","second_difference","local_dynamics","multiscale_context","reliability_modulation","fused_features","pre_tanh","trem_raw")}
    frozen_change=max_change(base_modules,initial)
    if frozen_change!=0: raise RuntimeError(f"frozen base changed: {frozen_change}")
    payload={"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
      "model_config":{"hidden":192,"heads":4,"layers":2,"trem_width":cfg["trem"]["width"],**spec},
      "dataset":cached["dataset"],"variant":a.variant,"seed":seed,"synthetic_cache_seed":cached["seed"],
      "history":history,"gradient_after_one_step":gradient_after_one_step,"stage_amplitudes":stage_amplitudes,
      "zero_initialization_error":zero_error,"frozen_base_max_abs_change":frozen_change,
      "input_effect":input_effect,"reliability_effect":reliability_effect,
      "parameter_counts":{"all":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in trainable),"trem":sum(p.numel() for p in model.trem.parameters())},
      "elapsed_seconds":elapsed,"optimizer_updates":updates,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(device),
      "labels_used":False,"validation_split_used":False,"checkpoint_selection":"fixed_final_epoch",
      "source_cache":str(a.cache),"apra_initialization":str(apra_checkpoint),"active_losses":["detect"],
      "final_score_training_formula":"B3 + 0.05*fixed_APRA + 0.05*TREM_raw"}
    a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_suffix(".tmp"); torch.save(payload,tmp); tmp.replace(a.output)
    print(json.dumps({"output":str(a.output),"elapsed_seconds":elapsed,"updates":updates}),flush=True); return 0
if __name__=="__main__": raise SystemExit(main())
