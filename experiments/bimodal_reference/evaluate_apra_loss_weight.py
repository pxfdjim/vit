#!/usr/bin/env python3
"""Evaluate APRA loss and gamma variants after scores are immutable."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.bimodal_reference.mindts_metrics import _events,affiliation_scores,fast_vus
from experiments.bimodal_reference.orionbench_protocol import DATASETS,evaluation_frame,labels_for,load_truth,signal_names
from experiments.bimodal_reference.run_apra_loss_weight_inference import output_variants
from src.models.model_utils import compute_detection_intervals
M=("aff_f1","vus_pr","vus_roc")
def hits(events,intervals): return np.asarray([any(l<=e and r>=s for l,r in intervals) for s,e in events],bool)
def fps(events,intervals): return sum(not any(s<=ee and e>=es for es,ee in events) for s,e in intervals)
def aggregate(metrics,cfg,variants):
 rows=[]
 for (d,v),g in metrics.groupby(["dataset","variant"],sort=False):
  row={"dataset":d,"variant":v,"signals":len(g)}
  for m in M: x=g[m].to_numpy(float); row[m]=float(np.nanmean(x)) if np.isfinite(x).any() else np.nan; row[f"{m}_valid_signals"]=int(np.isfinite(x).sum())
  rows.append(row)
 ds=pd.DataFrame(rows); scopes=[]
 for scope,datasets in cfg["macro_scopes"].items():
  for v in variants:
   g=ds[(ds.dataset.isin(datasets))&(ds.variant==v)]; row={"scope":scope,"variant":v,"datasets":len(g)}
   for m in M: row[m]=float(np.nanmean(g[m])); row[f"{m}_valid_signals"]=int(g[f"{m}_valid_signals"].sum())
   scopes.append(row)
 return ds,pd.DataFrame(scopes)
def shard(a,cfg,variants):
 truth=load_truth(Path(cfg["data_root"])); mr=[]; er=[]; ar=[]; task=0
 for d in DATASETS:
  for s in signal_names(Path(cfg["data_root"]),d):
   cur=task; task+=1
   if cur%a.num_workers!=a.worker_index: continue
   frame=evaluation_frame(Path(cfg["data_root"]),d,s); labels=labels_for(frame,truth[s]); events=_events(labels)
   for v in variants:
    with np.load(a.output_root/"point_scores"/d/s/f"{v}.npz") as z: score=z["scores"].astype(float); ts=z["timestamps"]
    valid=len(score)==len(frame) and np.array_equal(ts,frame.timestamp.to_numpy()) and np.isfinite(score).all(); ar.append({"dataset":d,"signal":s,"variant":v,"valid":valid,"points":len(score)})
    if not valid: raise RuntimeError(f"alignment {d}/{s}/{v}")
    base={"dataset":d,"signal":s,"variant":v,"points":len(score),"anomaly_points":int(labels.sum())}
    if labels.min()==labels.max(): mr.append({**base,"aff_f1":np.nan,"vus_pr":np.nan,"vus_roc":np.nan,"excluded_reason":"single_class_ground_truth"}); continue
    intervals,threshold,_=compute_detection_intervals(score,alpha=float(cfg["evaluation"]["affiliation_alpha"]),method=cfg["evaluation"]["threshold_method"],smoothing=True,sliding=False,anomaly_padding=0); pred=np.zeros(len(score),np.uint8)
    for l,r in intervals: pred[l:r+1]=1
    precision,recall,f1=affiliation_scores(labels,pred,Path(cfg["mindts_root"]),empty_prediction_f1_zero=True); vus=fast_vus(labels,score); h=hits(events,intervals)
    mr.append({**base,"aff_precision":precision,"aff_recall":recall,"aff_f1":f1,"vus_pr":vus.vus_pr,"vus_roc":vus.vus_roc,"aff_threshold":float(threshold),"predicted_intervals":len(intervals),"excluded_reason":""}); er.append({"dataset":d,"signal":s,"variant":v,"true_events":len(events),"tp_events":int(h.sum()),"fn_events":int((~h).sum()),"fp_intervals":fps(events,intervals),"predicted_intervals":len(intervals),"event_hits_json":json.dumps(h.astype(int).tolist())})
 out=a.output_root/"evaluation_shards"; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(mr).to_csv(out/f"metrics_{a.worker_index:02d}.csv",index=False); pd.DataFrame(er).to_csv(out/f"events_{a.worker_index:02d}.csv",index=False); pd.DataFrame(ar).to_csv(out/f"alignment_{a.worker_index:02d}.csv",index=False); print(json.dumps({"status":"passed","worker":a.worker_index,"metric_rows":len(mr)}))
def combine(a,cfg,variants):
 root=a.output_root/"evaluation_shards"
 def read(name):
  paths=sorted(root.glob(f"{name}_*.csv"));
  if len(paths)!=a.num_workers: raise RuntimeError(f"expected {a.num_workers} {name} shards, got {len(paths)}")
  return pd.concat([pd.read_csv(x) for x in paths],ignore_index=True)
 metrics=read("metrics").sort_values(["dataset","signal","variant"]); events=read("events"); alignment=read("alignment"); ds,scope=aggregate(metrics,cfg,variants)
 event_all=events.groupby("variant",as_index=False).agg(signals=("signal","count"),true_events=("true_events","sum"),tp_events=("tp_events","sum"),fn_events=("fn_events","sum"),fp_intervals=("fp_intervals","sum"),predicted_intervals=("predicted_intervals","sum")); event_ds=events.groupby(["dataset","variant"],as_index=False).agg(signals=("signal","count"),true_events=("true_events","sum"),tp_events=("tp_events","sum"),fn_events=("fn_events","sum"),fp_intervals=("fp_intervals","sum"),predicted_intervals=("predicted_intervals","sum"))
 differences=[]
 for (d,s),g in events.groupby(["dataset","signal"],sort=False):
  lookup=g.set_index("variant"); base=np.asarray(json.loads(lookup.loc["E0_full_s20261104","event_hits_json"]),bool)
  for v in variants:
   h=np.asarray(json.loads(lookup.loc[v,"event_hits_json"]),bool); differences.append({"dataset":d,"signal":s,"variant":v,"recovered_E0_misses":int(((~base)&h).sum()),"lost_E0_hits":int((base&(~h)).sum()),"delta_fp_intervals":int(lookup.loc[v,"fp_intervals"]-lookup.loc["E0_full_s20261104","fp_intervals"])})
 valid=metrics[metrics.excluded_reason.fillna("")==""]; paired=[]
 for v in variants:
  if v=="E0_full_s20261104": continue
  l=valid[valid.variant==v].set_index(["dataset","signal"]); r=valid[valid.variant=="E0_full_s20261104"].set_index(["dataset","signal"]); common=l.index.intersection(r.index)
  for m in M:
   d=l.loc[common,m]-r.loc[common,m]; nz=d[d!=0]; _,p=wilcoxon(nz) if len(nz) else (0,1); paired.append({"variant":v,"baseline":"E0_full_s20261104","metric":m,"signals":len(d),"mean_difference":float(d.mean()),"median_difference":float(d.median()),"positive":int((d>0).sum()),"negative":int((d<0).sum()),"zero":int((d==0).sum()),"wilcoxon_pvalue_descriptive":float(p)})
 seedrows=[]
 for sc in cfg["macro_scopes"]:
  lookup=scope[scope.scope==sc].set_index("variant")
  for v in cfg["variants"]:
   names=[f"{v}_s{s}" for s in cfg["seeds"]]; vals=lookup.loc[names]; row={"scope":sc,"variant":v,"seeds":3}
   for m in M: row[f"{m}_mean"]=float(vals[m].mean()); row[f"{m}_std"]=float(vals[m].std(ddof=0))
   seedrows.append(row)
 seed_ds_rows=[]
 for dataset in cfg["datasets"]:
  lookup=ds[ds.dataset==dataset].set_index("variant")
  for v in cfg["variants"]:
   names=[f"{v}_s{s}" for s in cfg["seeds"]]; vals=lookup.loc[names]; row={"dataset":dataset,"variant":v,"seeds":3}
   for m in M: row[f"{m}_mean"]=float(vals[m].mean()); row[f"{m}_std"]=float(vals[m].std(ddof=0))
   seed_ds_rows.append(row)
 checkpoints=[]
 for path in sorted((a.output_root/"checkpoints").glob("*/seed_*/*.pt")):
  p=torch.load(path,map_location="cpu",weights_only=True); row={"variant":p["variant"],"seed":p["seed"],"dataset":p["dataset"],"elapsed_seconds":p["elapsed_seconds"],"optimizer_updates":p["optimizer_updates"],"initialization_sha256":p["initialization_sha256"],"active_losses":"|".join(p["active_losses"]),"reference_effect":p["reference_effect"],"parameters_all":p["parameter_counts"]["all"],"parameters_trainable":p["parameter_counts"]["trainable"]}
  for loss,norms in p["gradient_audit_by_loss"].items():
   for group,value in norms.items(): row[f"gradient_{loss}_{group}"]=value
  for group,value in p["detect_gradient_after_one_step"].items(): row[f"gradient_detect_after_step_{group}"]=value
  checkpoints.append(row)
 manifests=[json.loads(x.read_text()) for x in sorted((a.output_root/"inference_manifests").glob("*.json"))]; diag=pd.DataFrame([x for m in manifests for x in m["diagnostics"]]); diag_summary=diag.groupby(["model_id","variant","seed"],as_index=False).mean(numeric_only=True)
 outputs={"metrics_per_signal.csv":metrics,"metrics_by_dataset.csv":ds,"metrics_by_scope.csv":scope,"event_counts_per_signal.csv":events,"event_counts_by_dataset.csv":event_ds,"event_counts_all.csv":event_all,"event_differences_per_signal.csv":pd.DataFrame(differences),"alignment_audit.csv":alignment,"paired_metric_tests.csv":pd.DataFrame(paired),"multiseed_summary.csv":pd.DataFrame(seedrows),"multiseed_by_dataset.csv":pd.DataFrame(seed_ds_rows),"training_checkpoint_summary.csv":pd.DataFrame(checkpoints),"path_diagnostics_per_signal.csv":diag,"path_diagnostics_summary.csv":diag_summary}
 for name,frame in outputs.items(): frame.to_csv(a.output_root/name,index=False)
 print(scope.to_string(index=False))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--worker-index",type=int,default=0); p.add_argument("--num-workers",type=int,default=1); p.add_argument("--combine-shards",action="store_true"); a=p.parse_args(); cfg=json.loads(a.config.read_text()); variants=output_variants(cfg); combine(a,cfg,variants) if a.combine_shards else shard(a,cfg,variants); return 0
if __name__=="__main__": raise SystemExit(main())
