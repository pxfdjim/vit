#!/usr/bin/env python3
"""Fix lambda_select from initial unlabeled loss scales before training."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.models.apra_acef import (APRAACEF,LossWeights,detection_loss,
    load_numeric_from_full,selective_alignment_loss,selective_time_warp)
from experiments.bimodal_reference.train_apra_selective import load_frozen_visual,projected_visual

def base_proxy(nq,vq,nr,vr):
    return .5*(1-F.cosine_similarity(nq,nr,dim=-1)).detach()+.5*(1-F.cosine_similarity(vq,vr,dim=-1)).detach()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0")
    a=p.parse_args(); cfg=json.loads(a.config.read_text()); device=torch.device(a.device); rows=[]
    for dataset in cfg["datasets"]:
        cache=torch.load(Path(cfg["rcca_training_cache"])/f"{dataset}.pt",map_location="cpu",weights_only=False)
        torch.manual_seed(cfg["primary_seed"]); torch.cuda.manual_seed_all(cfg["primary_seed"])
        model=APRAACEF("apra","apra_head",True).to(device)
        load_numeric_from_full(model,str(Path(cfg["full_checkpoint_root"])/f"{dataset}.pt"))
        load_frozen_visual(model,Path(cfg["rcca_run"])/"checkpoints/M1_full"/f"seed_{cfg['primary_seed']}"/f"{dataset}.pt")
        n=min(cfg["training"]["batch_size"],len(cache["raw"])); idx=torch.arange(n,device=device)
        raw=cache["raw"][:n].to(device); synth_raw=cache["synthetic_raw"][:n].to(device)
        visual=projected_visual(model,cache["visual"][:n],device).float()
        synth_visual=projected_visual(model,cache["synthetic_visual"][:n],device).float()
        ids_all=cache["reference_ids"][:n]; unique=torch.unique(ids_all)
        # Reference indices can fall outside the audit batch; encode them directly.
        nr_full=model.numeric(cache["raw"][ids_all].to(device)); vr_full=projected_visual(model,cache["visual"][ids_all],device).float()
        cols=cache["reference_columns"][:n].to(device)[...,None]
        nq=model.numeric(raw); ns=model.numeric(synth_raw)
        nr=nr_full.gather(1,cols.expand(-1,-1,nr_full.shape[-1])); vr=vr_full.gather(1,cols.expand(-1,-1,vr_full.shape[-1]))
        clean=model.forward_tokens(nq,visual,nr,vr); synthetic=model.forward_tokens(ns,synth_visual,nr,vr)
        modes=torch.arange(n,device=device)%3
        amplitudes=torch.linspace(-1.,1.,n,device=device).sign()*torch.linspace(.35,1.,n,device=device)
        wn,target,_=selective_time_warp(nq,modes,amplitudes,torch.linspace(.3,.7,n,device=device))
        warped=model.forward_tokens(wn,target@visual,nr,vr)
        altered=cache["patch_mask"][:n].to(device)
        kinds=[x["kind"] for x in json.loads(cache["perturbations_json"])[:n]]
        structural_rows=torch.tensor([x in set(cfg["selective_supervision"]["reject_perturbations"]) for x in kinds],device=device)
        structural=altered&structural_rows[:,None]
        select,parts=selective_alignment_loss(clean,synthetic,warped,target,altered,structural)
        detect=detection_loss(clean,synthetic,warped,base_proxy(nq,visual,nr,vr),
                              base_proxy(ns,synth_visual,nr,vr),altered,LossWeights(**cfg["loss_weights"]))
        rows.append({"dataset":dataset,"select":float(select),"detect":float(detect),
                     **{f"select_{k}":float(v) for k,v in parts.items()}})
        del model,cache; torch.cuda.empty_cache()
    select_median=float(np.median([x["select"] for x in rows])); detect_median=float(np.median([x["detect"] for x in rows]))
    raw=detect_median/max(select_median,1e-8); fixed=round(float(np.clip(raw,.25,4.0))/0.05)*0.05
    result={"labels_used":False,"rule":cfg["lambda_selection_rule"],"rows":rows,
            "median_select":select_median,"median_detect":detect_median,
            "raw_ratio":raw,"lambda_select":fixed}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
