#!/usr/bin/env python3
"""Infer new APRA mechanism ablations and reuse immutable E0--E5 scores."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.bimodal_reference.orionbench_protocol import DATASETS,evaluation_frame,signal_names
from experiments.bimodal_reference.run_amr_feature_scoring import load_caches,read_score,restore,write_score
from experiments.temporal_matching.runner import publish_json_no_replace,publish_npz_no_replace
from src.models.apra_acef import APRAACEF,selective_time_warp
from src.models.bimodal_reference_vit4ts import _fitted_standardized_windows,_visual_costs
from src.models.fixed_reference_numeric_evidence import positive_median
from src.models.numeric_temporal_matching import match_paths
from src.models.vfg_paired_vit4ts import _patch_tokens,select_common_visual_windows

def model_ids(cfg): return [f"{v}_s{s}" for v,spec in cfg["variants"].items() if spec.get("train",True) for s in cfg["seeds"]]
def inference_ids(cfg): return [f"E6_detect_only_s{cfg['primary_seed']}",*model_ids(cfg)]
def existing_ids(cfg):
 return [f"E1_existing_s{s}" for s in cfg["seeds"]]+[f"{v}_s{s}" for v in ("E2_select","E3_no_reject","E4_no_warp","E5_fixed_dtw","E6_detect_only") for s in cfg["seeds"]]
def output_variants(cfg): return ["B3"]+existing_ids(cfg)+model_ids(cfg)
def checkpoint(cfg,root,variant,seed,dataset):
 seed=int(seed)
 if variant=="E6_detect_only":
  return Path(cfg["loss_weight_run"])/"checkpoints/E1_no_path"/f"seed_{seed}"/f"{dataset}.pt"
 return root/"checkpoints"/variant/f"seed_{seed}"/f"{dataset}.pt"
def load_model(path,device):
 p=torch.load(path,map_location="cpu",weights_only=True); m=APRAACEF(**p["model_config"]).to(device); m.load_state_dict(p["state_dict"],strict=True); m.eval()
 for q in m.parameters(): q.requires_grad_(False)
 return m,p
def signed_calibration(local,base,n,device):
 point=restore(local,n,device); mag=np.abs(point[np.isfinite(point)]); mag=mag[mag>1e-12]
 if not len(mag): return np.zeros_like(point),1.
 scale=float(np.clip(positive_median(base)/max(float(np.median(mag)),1e-12),1e-4,1e4)); return point*scale,scale
@torch.inference_mode()
def evidence(model,windows,patch,ids,cols,batch=256):
 numeric=torch.cat([model.numeric(windows[i:i+batch]) for i in range(0,len(windows),batch)])
 visual=torch.cat([model.visual(patch[i:i+batch]) for i in range(0,len(patch),batch)])
 deltas=[]; rejection=[]; match_probability=[]; reliability=[]; residual=[]; entropy_all=[]; displacement_all=[]; rows=[]
 for st in range(0,len(windows),batch):
  en=min(st+batch,len(windows)); ri=ids[st:en]; c=cols[st:en][...,None]; nq=numeric[st:en]; vq=visual[st:en]
  nr0=numeric[ri]; vr0=visual[ri]; nr=nr0.gather(1,c.expand(-1,-1,nr0.shape[-1])); vr=vr0.gather(1,c.expand(-1,-1,vr0.shape[-1]))
  out=model.forward_tokens(nq,vq,nr,vr); a=out["alignment"]; deltas.append(out["delta"]); match=a["soft_match"]; norm=match/match.sum(-1,keepdim=True).clamp_min(1e-8)
  ent=-(norm.clamp_min(1e-8)*norm.clamp_min(1e-8).log()).sum(-1)/np.log(14); index=torch.arange(14,device=match.device,dtype=match.dtype); disp=(norm*index[None,None]).sum(-1)-index[None]
  wn,target,_=selective_time_warp(nq,torch.arange(len(nq),device=nq.device)%3,torch.ones(len(nq),device=nq.device)*.75,torch.ones(len(nq),device=nq.device)*.5); warped=model.forward_tokens(wn,target@vq,nr,vr)
  structural=a["structural_numeric"].norm(dim=-1)+a["structural_visual"].norm(dim=-1)
  rejection.append(a["rejection"]); match_probability.append(match.sum(-1)); reliability.append(a["reliability"]); residual.append(structural); entropy_all.append(ent); displacement_all.append(disp.abs())
  rows.append({"path_entropy":float(ent.mean()),"match_probability":float(match.sum(-1).mean()),"rejection":float(a["rejection"].mean()),"reference_skip":float(a["reference_skip"].mean()),"reliability":float(a["reliability"].mean()),"alignment_displacement":float(disp.abs().mean()),"structural_residual":float(structural.mean()),"delta_abs":float(out["delta"].abs().mean()),"normal_warp_delta_change":float((warped["delta"]-out["delta"]).abs().mean()),"normal_warp_rejection":float(warped["alignment"]["rejection"].mean()),"normal_warp_target_mass":float((target*warped["alignment"]["soft_match"]).sum(-1).mean())})
 token={"delta":torch.cat(deltas),"rejection":torch.cat(rejection),"match_probability":torch.cat(match_probability),"reliability":torch.cat(reliability),"structural_residual":torch.cat(residual),"path_entropy":torch.cat(entropy_all),"alignment_displacement":torch.cat(displacement_all)}
 return token,{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--worker-index",type=int,default=0); p.add_argument("--num-workers",type=int,default=1); p.add_argument("--skip-complete",action="store_true"); a=p.parse_args()
 cfg=json.loads(a.config.read_text()); rcfg=json.loads(Path(cfg["rcca_config"]).read_text()); variants=output_variants(cfg); device=torch.device(a.device); root=Path(cfg["data_root"]); feature=Path(cfg["feature_scoring_run"]); stats=json.loads(Path(cfg["normalization_stats"]).read_text()); tasks=[(d,s) for d in DATASETS for s in signal_names(root,d)][a.worker_index::a.num_workers]; records=[]; diagnostics=[]; failures=0; current=None; models={}; commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip()
 for ordinal,(dataset,signal) in enumerate(tasks,1):
  tick=time.perf_counter(); expected=[a.output_root/"point_scores"/dataset/signal/f"{v}.npz" for v in variants]; diagpath=a.output_root/"feature_diagnostics"/dataset/signal/"evidence.npz"
  if a.skip_complete and diagpath.is_file() and all(x.is_file() for x in expected): records.append({"dataset":dataset,"signal":signal,"status":"skipped_complete"}); continue
  try:
   frame=evaluation_frame(root,dataset,signal); raw=frame.value.to_numpy(np.float64); ts=frame.timestamp.to_numpy(); final,block,_,_,digest=load_caches(raw,{"dataset":dataset,"signal":signal},{"cache":rcfg["cache"]}); patch=_patch_tokens(block["intermediate_patch"]).float().to(device); pool=select_common_visual_windows(_patch_tokens(final["patch"]).to(device)); t3=match_paths(_visual_costs(patch.reshape(len(patch),14,14,-1),pool),pool,band=2,penalty_scale=.1)
   with np.load(feature/"feature_diagnostics"/dataset/signal/"evidence.npz") as z:
    if not np.array_equal(t3.reference_ids.cpu(),z["t3_reference_window_ids"]) or not np.array_equal(t3.reference_columns.cpu(),z["t3_reference_columns"]): raise RuntimeError("T3 mismatch")
   ids=t3.reference_ids.long().to(device); cols=t3.reference_columns.round().long().to(device); fit=stats[dataset][signal]; start=int(fit["fit_points"]) if dataset in {"MSL","SMAP"} else 0; windows=_fitted_standardized_windows(raw,fit,start).to(device); b3,bts=read_score(feature/"point_scores"/dataset/signal/"B3_raw_numeric_l010.npz")
   if not np.array_equal(bts,ts): raise RuntimeError("timestamp mismatch")
   scores={"B3":b3}
   for mid in existing_ids(cfg):
    source_mid=mid.replace("E6_detect_only_","E1_no_path_")
    source_root=Path(cfg["loss_weight_run"]) if mid.startswith("E6_detect_only_") else Path(cfg["selective_run"])
    reused,reused_ts=read_score(source_root/"point_scores"/dataset/signal/f"{source_mid}.npz")
    if not np.array_equal(reused_ts,ts): raise RuntimeError(f"timestamp mismatch: {mid}")
    scores[mid]=reused
   if current!=dataset:
    models={mid:load_model(checkpoint(cfg,a.output_root,*mid.rsplit("_s",1),dataset),device) for mid in inference_ids(cfg)}; current=dataset
   stored={}
   for mid,(model,payload) in models.items():
    mt=time.perf_counter(); token,diag=evidence(model,windows,patch,ids,cols); point,scale=signed_calibration(token["delta"],b3,len(raw),device); variant,seed=mid.rsplit("_s",1)
    if variant!="E6_detect_only": scores[mid]=b3+float(cfg["loss_weights"]["gamma"])*point
    stored[f"point_delta__{mid}"]=point
    if int(seed)==cfg["primary_seed"]:
     for name,value in token.items(): stored[f"point_{name}__{variant}"]=restore(value,len(raw),device)
    diag.update({"dataset":dataset,"signal":signal,"model_id":mid,"variant":variant,"seed":int(seed),"calibration_scale":scale,"inference_seconds":time.perf_counter()-mt}); diagnostics.append(diag)
   for v in variants: write_score(a.output_root/"point_scores"/dataset/signal/f"{v}.npz",scores[v],ts,{"variant":v,"labels_opened":False,"fixed_reference":"T3","gamma":0.05 if v!="B3" else 0.0,"code_commit":commit},time.perf_counter()-tick)
   publish_npz_no_replace(diagpath,**stored); records.append({"dataset":dataset,"signal":signal,"status":"passed","elapsed_seconds":time.perf_counter()-tick,"labels_opened":False,"t3_error":0}); print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s",flush=True)
  except Exception as e: failures+=1; records.append({"dataset":dataset,"signal":signal,"status":"failed","error":f"{type(e).__name__}: {e}"}); print(f"FAILED {dataset}/{signal}: {e}",flush=True)
 publish_json_no_replace(a.output_root/"inference_manifests"/f"worker_{a.worker_index:02d}_of_{a.num_workers:02d}.json",{"status":"passed" if not failures else "failed","records":records,"diagnostics":diagnostics,"failures":failures,"labels_opened":False}); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())
