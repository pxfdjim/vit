#!/usr/bin/env python3
"""Evaluate frozen APRA with independent TREM corrections."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon,spearmanr
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.bimodal_reference.mindts_metrics import _events,affiliation_scores,fast_vus
from experiments.bimodal_reference.orionbench_protocol import DATASETS,evaluation_frame,labels_for,load_truth,signal_names
from experiments.bimodal_reference.run_apra_trem_independent_inference import output_variants,model_specs,beta_code
from src.models.model_utils import compute_detection_intervals

METRICS=("aff_f1","vus_pr","vus_roc")
def hits(events,intervals): return np.asarray([any(l<=e and r>=s for l,r in intervals) for s,e in events],bool)
def fps(events,intervals): return sum(not any(s<=ee and e>=es for es,ee in events) for s,e in intervals)
def aggregate(metrics,cfg,variants):
 rows=[]
 for (dataset,variant),g in metrics.groupby(["dataset","variant"],sort=False):
  row={"dataset":dataset,"variant":variant,"signals":len(g)}
  for m in METRICS:
   v=g[m].to_numpy(float); row[m]=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan; row[f"{m}_valid_signals"]=int(np.isfinite(v).sum())
  rows.append(row)
 by_dataset=pd.DataFrame(rows); scopes=[]
 for scope,datasets in cfg["macro_scopes"].items():
  for variant in variants:
   g=by_dataset[(by_dataset.dataset.isin(datasets))&(by_dataset.variant==variant)]; row={"scope":scope,"variant":variant,"datasets":len(g)}
   for m in METRICS: row[m]=float(np.nanmean(g[m])); row[f"{m}_valid_signals"]=int(g[f"{m}_valid_signals"].sum())
   scopes.append(row)
 return by_dataset,pd.DataFrame(scopes)
def shard(a,cfg,variants):
 truth=load_truth(Path(cfg["data_root"])); metric=[]; event=[]; feature=[]; align=[]; task=0
 for dataset in DATASETS:
  for signal in signal_names(Path(cfg["data_root"]),dataset):
   index=task; task+=1
   if index%a.num_workers!=a.worker_index: continue
   frame=evaluation_frame(Path(cfg["data_root"]),dataset,signal); labels=labels_for(frame,truth[signal]); events=_events(labels)
   values=frame.value.to_numpy(float); vol=pd.Series(values).rolling(16,center=True,min_periods=4).std().bfill().ffill().to_numpy(); normal=labels==0; high=normal&(vol>=np.quantile(vol[normal],.9))
   for variant in variants:
    with np.load(a.output_root/"point_scores"/dataset/signal/f"{variant}.npz") as z: scores=z["scores"].astype(float); timestamps=z["timestamps"]
    valid=len(scores)==len(frame) and np.array_equal(timestamps,frame.timestamp.to_numpy()) and np.isfinite(scores).all(); align.append({"dataset":dataset,"signal":signal,"variant":variant,"points":len(scores),"valid":valid})
    if not valid: raise RuntimeError(f"alignment {dataset}/{signal}/{variant}")
    base={"dataset":dataset,"signal":signal,"variant":variant,"points":len(scores),"anomaly_points":int(labels.sum())}
    if labels.min()==labels.max(): metric.append({**base,"aff_f1":np.nan,"vus_pr":np.nan,"vus_roc":np.nan,"excluded_reason":"single_class_ground_truth"}); continue
    intervals,threshold,_=compute_detection_intervals(scores,alpha=cfg["evaluation"]["affiliation_alpha"],method=cfg["evaluation"]["threshold_method"],smoothing=True,sliding=False,anomaly_padding=0); pred=np.zeros(len(scores),np.uint8)
    for s,e in intervals: pred[s:e+1]=1
    precision,recall,f1=affiliation_scores(labels,pred,Path(cfg["mindts_root"]),empty_prediction_f1_zero=True); vus=fast_vus(labels,scores); h=hits(events,intervals)
    metric.append({**base,"aff_precision":precision,"aff_recall":recall,"aff_f1":f1,"vus_pr":vus.vus_pr,"vus_roc":vus.vus_roc,"aff_threshold":float(threshold),"predicted_intervals":len(intervals),"excluded_reason":""})
    event.append({"dataset":dataset,"signal":signal,"variant":variant,"true_events":len(events),"tp_events":int(h.sum()),"fn_events":int((~h).sum()),"fp_intervals":fps(events,intervals),"predicted_intervals":len(intervals),"event_hits_json":json.dumps(h.astype(int).tolist())})
   with np.load(a.output_root/"feature_diagnostics"/dataset/signal/"evidence.npz") as z:
    for spec in model_specs(cfg):
     model_id=spec["id"]; raw=z[f"raw__{model_id}"].astype(float); score=z[f"calibrated__{model_id}"].astype(float); iqr=max(float(np.quantile(score[normal],.75)-np.quantile(score[normal],.25)),1e-12)
     feature.append({"dataset":dataset,"signal":signal,"model_id":model_id,"variant":spec["variant"],"seed":spec["seed"],"source":spec["source"],"raw_abs_mean":float(np.abs(raw).mean()),"raw_std":float(raw.std()),"calibrated_abs_mean":float(np.abs(score).mean()),"calibrated_std":float(score.std()),"saturation_fraction":float((np.abs(score)>=.999999).mean()),"anomaly_minus_high_volatility_normal_in_normal_iqr":float((np.median(score[labels>0])-np.median(score[high]))/iqr) if (labels>0).any() and high.any() else np.nan,"normal_volatility_spearman":float(spearmanr(score[normal],vol[normal]).statistic) if np.std(score[normal]) and np.std(vol[normal]) else np.nan})
 out=a.output_root/"evaluation_shards"; out.mkdir(parents=True,exist_ok=True)
 for name,rows in {"metrics":metric,"events":event,"features":feature,"alignment":align}.items(): pd.DataFrame(rows).to_csv(out/f"independent_{name}_worker_{a.worker_index:02d}.csv",index=False)
 print(json.dumps({"status":"passed","worker":a.worker_index,"metric_rows":len(metric)}))
def combine(a,cfg,variants):
 root=a.output_root/"evaluation_shards"
 def read(name):
  paths=sorted(root.glob(f"independent_{name}_worker_*.csv"));
  if len(paths)!=a.num_workers: raise RuntimeError((name,len(paths)))
  return pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
 metrics=read("metrics").sort_values(["dataset","signal","variant"]); events=read("events"); features=read("features"); alignment=read("alignment"); by_dataset,by_scope=aggregate(metrics,cfg,variants)
 deltas=[]
 for scope,g in by_scope.groupby("scope",sort=False):
  base=g.set_index("variant").loc["E1_APRA_only"]
  for _,row in g.iterrows(): deltas.append({"scope":scope,"variant":row.variant,**{f"delta_{m}_vs_E1":float(row[m]-base[m]) for m in METRICS}})
 event_all=events.groupby("variant",as_index=False).agg(signals=("signal","count"),true_events=("true_events","sum"),tp_events=("tp_events","sum"),fn_events=("fn_events","sum"),fp_intervals=("fp_intervals","sum"),predicted_intervals=("predicted_intervals","sum")); event_dataset=events.groupby(["dataset","variant"],as_index=False).agg(signals=("signal","count"),true_events=("true_events","sum"),tp_events=("tp_events","sum"),fn_events=("fn_events","sum"),fp_intervals=("fp_intervals","sum"),predicted_intervals=("predicted_intervals","sum"))
 differences=[]
 for (dataset,signal),g in events.groupby(["dataset","signal"],sort=False):
  lookup=g.set_index("variant"); base=np.asarray(json.loads(lookup.loc["E1_APRA_only","event_hits_json"]),bool)
  for variant in variants:
   hit=np.asarray(json.loads(lookup.loc[variant,"event_hits_json"]),bool); differences.append({"dataset":dataset,"signal":signal,"variant":variant,"recovered_E1_misses":int(((~base)&hit).sum()),"lost_E1_hits":int((base&(~hit)).sum()),"delta_fp_intervals":int(lookup.loc[variant,"fp_intervals"]-lookup.loc["E1_APRA_only","fp_intervals"])})
 valid=metrics[metrics.excluded_reason.fillna("")==""]; paired=[]
 for variant in variants:
  if variant=="E1_APRA_only": continue
  left=valid[valid.variant==variant].set_index(["dataset","signal"]); right=valid[valid.variant=="E1_APRA_only"].set_index(["dataset","signal"]); common=left.index.intersection(right.index)
  for m in METRICS:
   d=left.loc[common,m]-right.loc[common,m]; nz=d[d!=0]; _,p=wilcoxon(nz) if len(nz) else (0,1); paired.append({"variant":variant,"baseline":"E1_APRA_only","metric":m,"signals":len(d),"mean_difference":float(d.mean()),"median_difference":float(d.median()),"positive":int((d>0).sum()),"negative":int((d<0).sum()),"zero":int((d==0).sum()),"wilcoxon_pvalue_descriptive":float(p)})
 checkpoints=[]
 for path in sorted((a.output_root/"checkpoints").glob("*/seed_*/*.pt")):
  x=torch.load(path,map_location="cpu",weights_only=True); row={"variant":x["variant"],"seed":x["seed"],"dataset":x["dataset"],"elapsed_seconds":x["elapsed_seconds"],"optimizer_updates":x["optimizer_updates"],"peak_cuda_memory_bytes":x["peak_cuda_memory_bytes"],"frozen_base_max_abs_change":x["frozen_base_max_abs_change"],"zero_initialization_error":x["zero_initialization_error"],"input_effect":x["input_effect"],"reliability_effect":x["reliability_effect"],**{f"parameters_{k}":v for k,v in x["parameter_counts"].items()}}
  row.update({f"gradient_{k}":v for k,v in x["gradient_after_one_step"].items()})
  for stage,values in x["stage_amplitudes"].items(): row.update({f"stage_{stage}_{k}":v for k,v in values.items()})
  checkpoints.append(row)
 manifests=[json.loads(x.read_text()) for x in sorted((a.output_root/"inference_manifests").glob("*.json"))]; diagnostics=pd.DataFrame([r for m in manifests for r in m.get("diagnostics",[])])
 seed_rows=[]; primary=cfg["primary_beta"]
 for scope in cfg["macro_scopes"]:
  lookup=by_scope[by_scope.scope==scope].set_index("variant")
  for variant in cfg["multiseed_variants"]:
   names=[f"{variant}_s{seed}_{beta_code(primary)}" for seed in cfg["seeds"]]; values=lookup.loc[names]
   row={"scope":scope,"variant":variant,"beta":primary,"seeds":len(names)}
   for m in METRICS: row[f"{m}_mean"]=float(values[m].mean()); row[f"{m}_std"]=float(values[m].std(ddof=0))
   seed_rows.append(row)
 outputs={"metrics_per_signal.csv":metrics,"metrics_by_dataset.csv":by_dataset,"metrics_by_scope.csv":by_scope,"metric_deltas_by_scope.csv":pd.DataFrame(deltas),"event_counts_per_signal.csv":events,"event_counts_by_dataset.csv":event_dataset,"event_counts_all.csv":event_all,"event_differences_per_signal.csv":pd.DataFrame(differences),"trem_features_per_signal.csv":features,"alignment_audit.csv":alignment,"paired_metric_tests.csv":pd.DataFrame(paired),"training_checkpoint_summary.csv":pd.DataFrame(checkpoints),"inference_diagnostics_per_signal.csv":diagnostics,"inference_diagnostics_summary.csv":diagnostics.groupby(["model_id","variant","seed","source"],as_index=False).mean(numeric_only=True),"multiseed_summary.csv":pd.DataFrame(seed_rows)}
 for name,frame in outputs.items(): frame.to_csv(a.output_root/name,index=False)
 print(by_scope.to_string(index=False))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--worker-index",type=int,default=0); p.add_argument("--num-workers",type=int,default=1); p.add_argument("--combine-shards",action="store_true"); a=p.parse_args(); cfg=json.loads(a.config.read_text()); variants=output_variants(cfg); combine(a,cfg,variants) if a.combine_shards else shard(a,cfg,variants); return 0
if __name__=="__main__": raise SystemExit(main())
