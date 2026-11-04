#!/usr/bin/env python3
"""Run frozen-B3 plus bounded APRA/ACEF correction on fixed T3 references."""
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
from src.models.apra_acef import APRAACEF,reversible_warp
from src.models.bimodal_reference_vit4ts import _fitted_standardized_windows,_visual_costs
from src.models.fixed_reference_numeric_evidence import positive_median
from src.models.numeric_temporal_matching import match_paths
from src.models.vfg_paired_vit4ts import _patch_tokens,select_common_visual_windows

BASELINES=("T3","E2","B3","RCCA_CADM_M1","C2_simplified")
def output_variants(config): return list(BASELINES)+list(config["variants"])

def load_model(path,device):
    payload=torch.load(path,map_location="cpu",weights_only=True); model=APRAACEF(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"],strict=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,payload

def signed_calibration(local,base,full_length,device):
    point=restore(local,full_length,device); magnitude=np.abs(point[np.isfinite(point)])
    magnitude=magnitude[magnitude>1e-12]
    if not len(magnitude): return np.zeros_like(point),1.0
    scale=float(np.clip(positive_median(base)/max(float(np.median(magnitude)),1e-12),1e-4,1e4))
    return point*scale,scale

@torch.inference_mode()
def model_evidence(model,windows,patch,reference_ids,reference_columns,batch_size=256):
    numeric=[]; visual=[]
    for start in range(0,len(windows),batch_size):
        numeric.append(model.numeric(windows[start:start+batch_size])); visual.append(model.visual(patch[start:start+batch_size]))
    numeric=torch.cat(numeric); visual=torch.cat(visual); deltas=[]; rows=[]
    for start in range(0,len(windows),batch_size):
        stop=min(start+batch_size,len(windows)); ids=reference_ids[start:stop]
        columns=reference_columns[start:stop][...,None]; nr_full=numeric[ids]; vr_full=visual[ids]
        nr=nr_full.gather(1,columns.expand(-1,-1,nr_full.shape[-1])); vr=vr_full.gather(1,columns.expand(-1,-1,vr_full.shape[-1]))
        nq=numeric[start:stop]; vq=visual[start:stop]; out=model.forward_tokens(nq,vq,nr,vr); a=out["alignment"]
        deltas.append(out["delta"])
        match=a["soft_match"]; normalized=match/match.sum(-1,keepdim=True).clamp_min(1e-8)
        entropy=-(normalized.clamp_min(1e-8)*normalized.clamp_min(1e-8).log()).sum(-1)/np.log(14)
        index=torch.arange(14,device=match.device,dtype=match.dtype); displacement=(normalized*index[None,None]).sum(-1)-index[None]
        warped_n,target=reversible_warp(nq,torch.ones(len(nq),device=nq.device)); warped=model.forward_tokens(warped_n,target@vq,nr,vr)
        row={"path_entropy":entropy.mean(),"rejection":a["rejection"].mean(),"reliability":a["reliability"].mean(),
          "alignment_displacement":displacement.abs().mean(),"unmatched_numeric":a["unmatched_numeric"].norm(dim=-1).mean(),
          "unmatched_visual":a["unmatched_visual"].norm(dim=-1).mean(),"structural_numeric":a["structural_numeric"].norm(dim=-1).mean(),
          "structural_visual":a["structural_visual"].norm(dim=-1).mean(),"delta_mean":out["delta"].mean(),
          "delta_std":out["delta"].std(unbiased=False),"normal_warp_delta_change":(warped["delta"]-out["delta"]).abs().mean()}
        if "shared_evidence" in out:
            row["shared_evidence_norm"]=out["shared_evidence"].norm(dim=-1).mean()
            row["complementary_evidence_norm"]=out["complementary_evidence"].norm(dim=-1).mean()
        rows.append({k:float(v.cpu()) for k,v in row.items()})
    return torch.cat(deltas),{k:float(np.mean([r.get(k,np.nan) for r in rows])) for k in rows[0]}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--device",default="cuda:0"); p.add_argument("--worker-index",type=int,default=0); p.add_argument("--num-workers",type=int,default=1)
    p.add_argument("--datasets",nargs="+"); p.add_argument("--signals",nargs="+"); p.add_argument("--skip-complete",action="store_true"); a=p.parse_args()
    cfg=json.loads(a.config.read_text()); rcca_cfg=json.loads(Path(cfg["rcca_config"]).read_text()); variants=output_variants(cfg)
    device=torch.device(a.device); data_root=Path(cfg["data_root"]); feature=Path(cfg["feature_scoring_run"])
    ablation=Path(cfg["training_ablation_run"]); rcca=Path(cfg["rcca_run"]); stats=json.loads(Path(cfg["normalization_stats"]).read_text())
    datasets=a.datasets or list(DATASETS); tasks=[(d,s) for d in datasets for s in signal_names(data_root,d)]
    if a.signals: tasks=[x for x in tasks if x[1] in set(a.signals)]
    tasks=tasks[a.worker_index::a.num_workers]; records=[]; diagnostic_rows=[]; failures=0; current=None; models={}
    commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip(); gamma=float(cfg["loss_weights"]["gamma"])
    for ordinal,(dataset,signal) in enumerate(tasks,1):
        tick=time.perf_counter(); expected=[a.output_root/"point_scores"/dataset/signal/f"{v}.npz" for v in variants]
        diag_path=a.output_root/"feature_diagnostics"/dataset/signal/"evidence.npz"
        if a.skip_complete and diag_path.is_file() and all(x.is_file() for x in expected):
            records.append({"dataset":dataset,"signal":signal,"status":"skipped_complete"}); continue
        try:
            frame=evaluation_frame(data_root,dataset,signal); raw=frame.value.to_numpy(np.float64); timestamps=frame.timestamp.to_numpy()
            final,block,_,_,digest=load_caches(raw,{"dataset":dataset,"signal":signal},{"cache":rcca_cfg["cache"]})
            patch=_patch_tokens(block["intermediate_patch"]).float().to(device); pool=select_common_visual_windows(_patch_tokens(final["patch"]).to(device))
            visual_costs=_visual_costs(patch.reshape(len(patch),14,14,-1),pool); t3=match_paths(visual_costs,pool,band=2,penalty_scale=.1)
            with np.load(feature/"feature_diagnostics"/dataset/signal/"evidence.npz") as stored:
                if not np.array_equal(t3.reference_ids.cpu().numpy(),stored["t3_reference_window_ids"]): raise RuntimeError("T3 reference mismatch")
                if not np.array_equal(t3.reference_columns.cpu().numpy(),stored["t3_reference_columns"]): raise RuntimeError("T3 path mismatch")
            ids=t3.reference_ids.long().to(device); columns=t3.reference_columns.round().long().to(device)
            fitted=stats[dataset][signal]; start=int(fitted["fit_points"]) if dataset in {"MSL","SMAP"} else 0
            windows=_fitted_standardized_windows(raw,fitted,start).to(device)
            source={"T3":feature/"point_scores"/dataset/signal/"T3.npz","E2":feature/"point_scores"/dataset/signal/"E2_T3_raw_l010.npz",
              "B3":feature/"point_scores"/dataset/signal/"B3_raw_numeric_l010.npz","RCCA_CADM_M1":rcca/"point_scores"/dataset/signal/"M1_full.npz",
              "C2_simplified":ablation/"point_scores"/dataset/signal/"C2_no_cross.npz"}
            scores={}
            for name,path in source.items():
                scores[name],source_time=read_score(path)
                if not np.array_equal(source_time,timestamps): raise RuntimeError(f"timestamp mismatch {name}")
            if current!=dataset:
                models.clear()
                for variant in cfg["variants"]: models[variant],_=load_model(a.output_root/"checkpoints"/variant/f"{dataset}.pt",device)
                current=dataset
            evidence={}; full_reference_effect=None
            for variant,model in models.items():
                local,diag=model_evidence(model,windows,patch,ids,columns); point,scale=signed_calibration(local,scores["B3"],len(raw),device)
                scores[variant]=scores["B3"]+gamma*point; evidence[f"point_delta__{variant}"]=point
                diag.update({"delta_calibration_scale":scale,"inference_seconds":time.perf_counter()-tick})
                diagnostic_rows.append({"dataset":dataset,"signal":signal,"variant":variant,**diag})
                if variant=="APRA_ACEF": full_reference_effect=diag["unmatched_numeric"]+diag["unmatched_visual"]
            if np.max(np.abs(scores["no_Ldetect"]-scores["B3"]))!=0: raise RuntimeError("no_Ldetect must preserve B3 exactly")
            for variant in variants:
                metadata={"variant":variant,"labels_opened":False,"fixed_reference":"T3","fixed_dtw_anchor":"T3",
                  "code_commit":commit,"input_digest":digest[:20],"base":"B3" if variant not in BASELINES else None,
                  "gamma":gamma if variant not in BASELINES else None}
                write_score(a.output_root/"point_scores"/dataset/signal/f"{variant}.npz",scores[variant],timestamps,metadata,time.perf_counter()-tick)
            publish_npz_no_replace(diag_path,**evidence)
            records.append({"dataset":dataset,"signal":signal,"status":"passed","elapsed_seconds":time.perf_counter()-tick,
                            "labels_opened":False,"t3_reference_error":0,"t3_path_error":0,"no_detect_error":0})
            print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal} {records[-1]['elapsed_seconds']:.2f}s",flush=True)
        except Exception as error:
            failures+=1; records.append({"dataset":dataset,"signal":signal,"status":"failed","error":f"{type(error).__name__}: {error}"}); print(f"FAILED {dataset}/{signal}: {error}",flush=True)
    publish_json_no_replace(a.output_root/"inference_manifests"/f"worker_{a.worker_index:02d}_of_{a.num_workers:02d}.json",
      {"status":"passed" if not failures else "failed","records":records,"diagnostics":diagnostic_rows,"failures":failures,"labels_opened":False,"code_commit":commit})
    return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())
