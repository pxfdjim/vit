#!/usr/bin/env python3
"""Infer fixed APRA plus independently calibrated TREM evidence."""
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
from src.models.apra_trem import b3_token_evidence
from src.models.apra_trem_independent import (FrozenAPRAIndependentTREM,load_apra_only,
    load_old_trem,robust_signed_calibration)
from src.models.bimodal_reference_vit4ts import _fitted_standardized_windows,_visual_costs
from src.models.numeric_temporal_matching import match_paths
from src.models.vfg_paired_vit4ts import _patch_tokens,select_common_visual_windows

BASELINES=("E0_B3","E1_APRA_only")
def beta_code(value): return f"b{int(round(1000*value)):03d}"
def model_specs(cfg):
 specs=[{"id":"X_existing","source":"existing","seed":cfg["primary_seed"],"variant":cfg["existing_checkpoint_variant"]}]
 primary=cfg["primary_seed"]
 for seed in cfg["seeds"]:
  variants=cfg["variants"] if seed==primary else cfg["multiseed_variants"]
  specs += [{"id":f"{variant}_s{seed}","source":"trained","seed":seed,"variant":variant} for variant in variants]
 return specs
def output_variants(cfg):
 out=list(BASELINES)
 for spec in model_specs(cfg):
  betas=cfg["beta_values"] if spec["id"] in {"X_existing",f"F_full_s{cfg['primary_seed']}"} else [cfg["primary_beta"]]
  out += [f"{spec['id']}_{beta_code(beta)}" for beta in betas]
 return out
def load_model(cfg,spec,dataset,output_root,device):
 if spec["source"]=="existing":
  model=FrozenAPRAIndependentTREM(trem_width=cfg["trem"]["width"],**cfg["variants"]["F_full"]).to(device)
  load_apra_only(model,str(Path(cfg["apra_run"])/"checkpoints/APRA_only"/f"{dataset}.pt"))
  load_old_trem(model,str(Path(cfg["old_trem_run"])/"checkpoints"/cfg["existing_checkpoint_variant"]/f"{dataset}.pt"))
  payload={"parameter_counts":{"all":sum(p.numel() for p in model.parameters()),"trem":sum(p.numel() for p in model.trem.parameters())}}
 else:
  path=output_root/"checkpoints"/spec["variant"]/f"seed_{spec['seed']}"/f"{dataset}.pt"
  payload=torch.load(path,map_location="cpu",weights_only=True); model=FrozenAPRAIndependentTREM(**payload["model_config"]).to(device); model.load_state_dict(payload["state_dict"],strict=True)
 model.eval(); model.freeze_base()
 for p in model.trem.parameters(): p.requires_grad_(False)
 return model,payload
def reference_raw(raw,ids,columns):
 patches=raw[ids].reshape(len(ids),14,16); return patches.gather(1,columns[...,None].expand(-1,-1,16))
@torch.inference_mode()
def model_evidence(model,windows,patch,ids,columns,batch=256):
 numeric=torch.cat([model.numeric(windows[i:i+batch]) for i in range(0,len(windows),batch)])
 visual=torch.cat([model.visual(patch[i:i+batch]) for i in range(0,len(patch),batch)])
 raws=[]; rows=[]
 for start in range(0,len(windows),batch):
  stop=min(start+batch,len(windows)); ri=ids[start:stop]; rc=columns[start:stop]; ex=rc[...,None]
  nq=numeric[start:stop]; vq=visual[start:stop]; nr0=numeric[ri]; vr0=visual[ri]
  nr=nr0.gather(1,ex.expand(-1,-1,nr0.shape[-1])); vr=vr0.gather(1,ex.expand(-1,-1,vr0.shape[-1]))
  rq=windows[start:stop].reshape(stop-start,14,16); rr=reference_raw(windows,ri,rc)
  base=b3_token_evidence(nq,vq,nr,vr,rq,rr); tick=time.perf_counter(); out=model.forward_tokens(nq,vq,nr,vr,base); elapsed=time.perf_counter()-tick
  raws.append(out["trem_raw"]); w=out["fusion_weights"]
  row={"inference_seconds":elapsed,"raw_abs":float(out["trem_raw"].abs().mean()),"raw_std":float(out["trem_raw"].std()),"pre_tanh_abs":float(out["pre_tanh"].abs().mean()),
       "input_abs":float(out["input_features"].abs().mean()),"projection_abs":float(out["residual_projection"].abs().mean()),"first_diff_abs":float(out["first_difference"].abs().mean()),"second_diff_abs":float(out["second_difference"].abs().mean()),
       "local_abs":float(out["local_dynamics"].abs().mean()),"multiscale_abs":float(out["multiscale_context"].abs().mean()),"fused_abs":float(out["fused_features"].abs().mean()),
       "reliability":float(out["alignment"]["reliability"].mean()),"rejection":float(out["alignment"]["rejection"].mean()),
       "local_weight":float(w[...,0].mean()) if model.trem.use_local else 0.,"multiscale_weight":float(w[...,-1].mean()) if model.trem.use_multiscale else 0.}
  rows.append(row)
 return torch.cat(raws),{key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--worker-index",type=int,default=0); p.add_argument("--num-workers",type=int,default=1); p.add_argument("--skip-complete",action="store_true"); a=p.parse_args()
 cfg=json.loads(a.config.read_text()); rcca=json.loads(Path(cfg["rcca_config"]).read_text()); variants=output_variants(cfg); specs=model_specs(cfg); device=torch.device(a.device); data_root=Path(cfg["data_root"]); feature=Path(cfg["feature_scoring_run"]); apra=Path(cfg["apra_run"]); stats=json.loads(Path(cfg["normalization_stats"]).read_text())
 tasks=[(d,s) for d in DATASETS for s in signal_names(data_root,d)][a.worker_index::a.num_workers]; commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip(); records=[]; diagnostics=[]; failures=0; current=None; models={}
 cal=cfg["calibration"]
 for ordinal,(dataset,signal) in enumerate(tasks,1):
  tick=time.perf_counter(); expected=[a.output_root/"point_scores"/dataset/signal/f"{v}.npz" for v in variants]; diagnostic_path=a.output_root/"feature_diagnostics"/dataset/signal/"evidence.npz"
  if a.skip_complete and diagnostic_path.is_file() and all(x.is_file() for x in expected): records.append({"dataset":dataset,"signal":signal,"status":"skipped_complete"}); continue
  try:
   frame=evaluation_frame(data_root,dataset,signal); raw=frame.value.to_numpy(np.float64); timestamps=frame.timestamp.to_numpy(); final,block,_,_,digest=load_caches(raw,{"dataset":dataset,"signal":signal},{"cache":rcca["cache"]}); patch=_patch_tokens(block["intermediate_patch"]).float().to(device); pool=select_common_visual_windows(_patch_tokens(final["patch"]).to(device)); t3=match_paths(_visual_costs(patch.reshape(len(patch),14,14,-1),pool),pool,band=2,penalty_scale=.1)
   with np.load(feature/"feature_diagnostics"/dataset/signal/"evidence.npz") as z:
    if not np.array_equal(t3.reference_ids.cpu(),z["t3_reference_window_ids"]) or not np.array_equal(t3.reference_columns.cpu(),z["t3_reference_columns"]): raise RuntimeError("T3 reference/path mismatch")
   ids=t3.reference_ids.long().to(device); columns=t3.reference_columns.round().long().to(device); fitted=stats[dataset][signal]; start=int(fitted["fit_points"]) if dataset in {"MSL","SMAP"} else 0; windows=_fitted_standardized_windows(raw,fitted,start).to(device)
   scores={}; source={"E0_B3":feature/"point_scores"/dataset/signal/"B3_raw_numeric_l010.npz","E1_APRA_only":apra/"point_scores"/dataset/signal/"APRA_only.npz"}
   for name,path in source.items(): scores[name],ts=read_score(path); assert np.array_equal(ts,timestamps)
   if current!=dataset:
    models.clear()
    for spec in specs: models[spec["id"]]=(*load_model(cfg,spec,dataset,a.output_root,device),spec)
    current=dataset
   evidence={}
   for model_id,(model,payload,spec) in models.items():
    mt=time.perf_counter(); local,diag=model_evidence(model,windows,patch,ids,columns); point_raw=restore(local,len(raw),device); point,cs=robust_signed_calibration(point_raw,mad_multiplier=cal["mad_multiplier"],divisor=cal["scale_divisor"],floor=cal["scale_floor"],clip=cal["clip"])
    evidence[f"raw__{model_id}"]=point_raw; evidence[f"calibrated__{model_id}"]=point
    betas=cfg["beta_values"] if model_id in {"X_existing",f"F_full_s{cfg['primary_seed']}"} else [cfg["primary_beta"]]
    for beta in betas: scores[f"{model_id}_{beta_code(beta)}"]=scores["E1_APRA_only"]+float(beta)*point
    diag.update({"model_id":model_id,"dataset":dataset,"signal":signal,"seed":spec["seed"],"variant":spec["variant"],"source":spec["source"],"calibration_center":cs["center"],"calibration_mad":cs["mad"],"calibration_scale":cs["scale"],"calibrated_abs":float(np.abs(point).mean()),"calibrated_std":float(np.std(point)),"total_inference_seconds":time.perf_counter()-mt,"parameters_all":payload["parameter_counts"]["all"],"parameters_trem":payload["parameter_counts"]["trem"]}); diagnostics.append(diag)
   for variant in variants: write_score(a.output_root/"point_scores"/dataset/signal/f"{variant}.npz",scores[variant],timestamps,{"variant":variant,"labels_opened":False,"fixed_reference":"T3","fixed_apra":"APRA_only","code_commit":commit},time.perf_counter()-tick)
   publish_npz_no_replace(diagnostic_path,**evidence); records.append({"dataset":dataset,"signal":signal,"status":"passed","elapsed_seconds":time.perf_counter()-tick,"labels_opened":False,"t3_reference_error":0,"t3_path_error":0}); print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s",flush=True)
  except Exception as error:
   failures+=1; records.append({"dataset":dataset,"signal":signal,"status":"failed","error":f"{type(error).__name__}: {error}"}); print(f"FAILED {dataset}/{signal}: {error}",flush=True)
 publish_json_no_replace(a.output_root/"inference_manifests"/f"worker_{a.worker_index:02d}_of_{a.num_workers:02d}.json",{"status":"passed" if not failures else "failed","records":records,"diagnostics":diagnostics,"failures":failures,"labels_opened":False,"code_commit":commit}); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())
