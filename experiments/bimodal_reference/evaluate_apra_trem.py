#!/usr/bin/env python3
"""Evaluate APRA/TREM scores, events, signed evidence, and paired changes."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr,spearmanr,wilcoxon
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.bimodal_reference.mindts_metrics import _events,affiliation_scores,fast_vus
from experiments.bimodal_reference.orionbench_protocol import DATASETS,evaluation_frame,labels_for,load_truth,signal_names
from experiments.bimodal_reference.run_apra_trem_inference import output_variants
from src.models.model_utils import compute_detection_intervals

def hits(events,intervals): return np.asarray([any(l<=e and r>=s for l,r in intervals) for s,e in events],bool)
def false_positives(events,intervals): return sum(not any(s<=ee and e>=es for es,ee in events) for s,e in intervals)
def corr(fn,x,y): return np.nan if np.std(x)==0 or np.std(y)==0 else float(fn(x,y).statistic)
def aggregate(metrics,cfg,variants):
 rows=[]
 for (dataset,variant),g in metrics.groupby(["dataset","variant"],sort=False):
  row={"dataset":dataset,"variant":variant,"signals":len(g)}
  for m in ("aff_f1","vus_pr","vus_roc"):
   v=g[m].to_numpy(float); row[m]=float(np.nanmean(v)) if np.isfinite(v).any() else np.nan; row[f"{m}_valid_signals"]=int(np.isfinite(v).sum())
  rows.append(row)
 by_dataset=pd.DataFrame(rows); scopes=[]
 for scope,datasets in cfg["macro_scopes"].items():
  for variant in variants:
   g=by_dataset[(by_dataset.dataset.isin(datasets))&(by_dataset.variant==variant)]; row={"scope":scope,"variant":variant,"datasets":len(g)}
   for m in ("aff_f1","vus_pr","vus_roc"): row[m]=float(np.nanmean(g[m])); row[f"{m}_valid_signals"]=int(g[f"{m}_valid_signals"].sum())
   scopes.append(row)
 return by_dataset,pd.DataFrame(scopes)
def deltas(frame,key,baselines):
 rows=[]
 for name,g in frame.groupby(key,sort=False):
  lookup=g.set_index("variant")
  for _,item in g.iterrows():
   row={key:name,"variant":item.variant}
   for b in baselines:
    for m in ("aff_f1","vus_pr","vus_roc"): row[f"delta_{m}_vs_{b}"]=float(item[m]-lookup.loc[b,m])
   rows.append(row)
 return pd.DataFrame(rows)

def shard(args,cfg,variants):
 truth=load_truth(Path(cfg["data_root"])); metric_rows=[]; event_rows=[]; feature_rows=[]; align_rows=[]; task=0
 for dataset in DATASETS:
  for signal in signal_names(Path(cfg["data_root"]),dataset):
   current=task; task+=1
   if current%args.num_workers!=args.worker_index: continue
   frame=evaluation_frame(Path(cfg["data_root"]),dataset,signal); labels=labels_for(frame,truth[signal]); events=_events(labels)
   values=frame.value.to_numpy(float); vol=pd.Series(values).rolling(16,center=True,min_periods=4).std().bfill().ffill().to_numpy(); normal=labels==0
   high=normal&(vol>=np.quantile(vol[normal],.9)); score_cache={}
   for variant in variants:
    with np.load(args.output_root/"point_scores"/dataset/signal/f"{variant}.npz") as z: scores=z["scores"].astype(float); timestamps=z["timestamps"]
    valid=len(scores)==len(frame) and np.array_equal(timestamps,frame.timestamp.to_numpy()) and np.isfinite(scores).all(); align_rows.append({"dataset":dataset,"signal":signal,"variant":variant,"points":len(scores),"valid":valid})
    if not valid: raise RuntimeError(f"alignment {dataset}/{signal}/{variant}")
    score_cache[variant]=scores; base={"dataset":dataset,"signal":signal,"variant":variant,"points":len(scores),"anomaly_points":int(labels.sum())}
    if labels.min()==labels.max(): metric_rows.append({**base,"aff_f1":np.nan,"vus_pr":np.nan,"vus_roc":np.nan,"excluded_reason":"single_class_ground_truth"}); continue
    intervals,threshold,_=compute_detection_intervals(scores,alpha=float(cfg["evaluation"]["affiliation_alpha"]),method=cfg["evaluation"]["threshold_method"],smoothing=cfg["evaluation"]["smoothing"],sliding=False,anomaly_padding=0)
    pred=np.zeros(len(scores),np.uint8)
    for s,e in intervals: pred[s:e+1]=1
    precision,recall,f1=affiliation_scores(labels,pred,Path(cfg["mindts_root"]),empty_prediction_f1_zero=True); vus=fast_vus(labels,scores); event_hit=hits(events,intervals)
    metric_rows.append({**base,"aff_precision":precision,"aff_recall":recall,"aff_f1":f1,"vus_pr":vus.vus_pr,"vus_roc":vus.vus_roc,"aff_threshold":float(threshold),"predicted_intervals":len(intervals),"excluded_reason":""})
    event_rows.append({"dataset":dataset,"signal":signal,"variant":variant,"true_events":len(events),"tp_events":int(event_hit.sum()),"fn_events":int((~event_hit).sum()),"fp_intervals":false_positives(events,intervals),"predicted_intervals":len(intervals),"event_hits_json":json.dumps(event_hit.astype(int).tolist())})
   with np.load(args.output_root/"feature_diagnostics"/dataset/signal/"evidence.npz") as z: evidence={k:z[k].astype(float) for k in z.files}
   for variant in cfg["variants"]:
    score=evidence[f"point_trem_change_vs_E1__{variant}"]; iqr=max(float(np.quantile(score[normal],.75)-np.quantile(score[normal],.25)),1e-12)
    feature_rows.append({"dataset":dataset,"signal":signal,"variant":variant,"delta_mean":float(score.mean()),"delta_std":float(score.std()),"positive_fraction":float((score>0).mean()),"negative_fraction":float((score<0).mean()),
      "anomaly_minus_high_volatility_normal_in_normal_iqr":float((np.median(score[labels>0])-np.median(score[high]))/iqr) if (labels>0).any() and high.any() else np.nan,
      "normal_volatility_spearman":corr(spearmanr,score[normal],vol[normal]),"pearson_with_E1":corr(pearsonr,score,score_cache["E1_APRA_only"])})
 out=args.output_root/"evaluation_shards"; out.mkdir(parents=True,exist_ok=True)
 for name,rows in {"metrics":metric_rows,"events":event_rows,"features":feature_rows,"alignment":align_rows}.items(): pd.DataFrame(rows).to_csv(out/f"trem_{name}_worker_{args.worker_index:02d}.csv",index=False)
 print(json.dumps({"status":"passed","worker":args.worker_index,"metric_rows":len(metric_rows)}))

def combine(args,cfg,variants):
 root=args.output_root/"evaluation_shards"
 def read(name):
  paths=sorted(root.glob(f"trem_{name}_worker_*.csv"))
  if len(paths)!=args.num_workers: raise RuntimeError(f"expected {args.num_workers} {name} shards, got {len(paths)}")
  return pd.concat([pd.read_csv(x) for x in paths],ignore_index=True)
 metrics=read("metrics").sort_values(["dataset","signal","variant"]); events=read("events"); features=read("features"); alignment=read("alignment")
 by_dataset,by_scope=aggregate(metrics,cfg,variants); delta_dataset=deltas(by_dataset,"dataset",["E0_B3","E1_APRA_only"]); delta_scope=deltas(by_scope,"scope",["E0_B3","E1_APRA_only"])
 event_all=events.groupby("variant",as_index=False).agg(signals=("signal","count"),true_events=("true_events","sum"),tp_events=("tp_events","sum"),fn_events=("fn_events","sum"),fp_intervals=("fp_intervals","sum"),predicted_intervals=("predicted_intervals","sum"))
 event_dataset=events.groupby(["dataset","variant"],as_index=False).agg(signals=("signal","count"),true_events=("true_events","sum"),tp_events=("tp_events","sum"),fn_events=("fn_events","sum"),fp_intervals=("fp_intervals","sum"),predicted_intervals=("predicted_intervals","sum"))
 diff=[]
 for (dataset,signal),g in events.groupby(["dataset","signal"],sort=False):
  lookup=g.set_index("variant"); base=np.asarray(json.loads(lookup.loc["E1_APRA_only","event_hits_json"]),bool)
  for variant in variants:
   hit=np.asarray(json.loads(lookup.loc[variant,"event_hits_json"]),bool); diff.append({"dataset":dataset,"signal":signal,"variant":variant,"recovered_E1_misses":int(((~base)&hit).sum()),"lost_E1_hits":int((base&(~hit)).sum()),"delta_fp_intervals":int(lookup.loc[variant,"fp_intervals"]-lookup.loc["E1_APRA_only","fp_intervals"])})
 paired=[]; valid=metrics[metrics.excluded_reason.fillna("")==""]
 for variant in variants:
  if variant=="E1_APRA_only": continue
  left=valid[valid.variant==variant].set_index(["dataset","signal"]); right=valid[valid.variant=="E1_APRA_only"].set_index(["dataset","signal"]); common=left.index.intersection(right.index)
  for m in ("aff_f1","vus_pr","vus_roc"):
   d=left.loc[common,m]-right.loc[common,m]; nz=d[d!=0]; stat,p=wilcoxon(nz) if len(nz) else (0.,1.)
   paired.append({"variant":variant,"baseline":"E1_APRA_only","metric":m,"signals":len(d),"mean_difference":float(d.mean()),"median_difference":float(d.median()),"positive":int((d>0).sum()),"negative":int((d<0).sum()),"zero":int((d==0).sum()),"wilcoxon_pvalue_descriptive":float(p)})
 checkpoints=[]
 for path in sorted((args.output_root/"checkpoints").glob("*/*.pt")):
  p=torch.load(path,map_location="cpu",weights_only=True); row={"variant":p["variant"],"dataset":p["dataset"],"elapsed_seconds":p["elapsed_seconds"],"optimizer_updates":p["optimizer_updates"],"peak_cuda_memory_bytes":p["peak_cuda_memory_bytes"],"numeric_max_abs_change":p["numeric_max_abs_change"],"apra_max_abs_change":p["apra_max_abs_change"],"residual_input_effect":p["residual_input_effect"],"reliability_input_effect":p["reliability_input_effect"],"zero_initialized_trem":p["zero_initialized_trem"],"initial_equivalence_max_error":p["initial_equivalence_max_error"],"active_losses":"|".join(p["active_losses"]),**{f"parameters_{k}":v for k,v in p["parameter_counts"].items()}}
  for loss,norms in p["gradient_audit_by_loss"].items():
   for group,value in norms.items(): row[f"gradient_{loss}_{group}"]=value
  for group,value in p["detect_gradient_after_one_step"].items(): row[f"gradient_detect_step1_{group}"]=value
  checkpoints.append(row)
 manifests=[json.loads(x.read_text()) for x in sorted((args.output_root/"inference_manifests").glob("*.json"))]; diagnostics=pd.DataFrame([r for m in manifests for r in m.get("diagnostics",[])])
 diag_summary=diagnostics.groupby("variant",as_index=False).mean(numeric_only=True)
 outputs={"metrics_per_signal.csv":metrics,"metrics_by_dataset.csv":by_dataset,"metrics_by_scope.csv":by_scope,"metric_deltas_by_dataset.csv":delta_dataset,"metric_deltas_by_scope.csv":delta_scope,"event_counts_per_signal.csv":events,"event_counts_by_dataset.csv":event_dataset,"event_counts_all.csv":event_all,"event_differences_per_signal.csv":pd.DataFrame(diff),"correction_features_per_signal.csv":features,"alignment_audit.csv":alignment,"paired_metric_tests.csv":pd.DataFrame(paired),"training_checkpoint_summary.csv":pd.DataFrame(checkpoints),"inference_diagnostics_per_signal.csv":diagnostics,"inference_diagnostics_summary.csv":diag_summary}
 for name,frame in outputs.items(): frame.to_csv(args.output_root/name,index=False)
 print(by_scope.to_string(index=False))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--worker-index",type=int,default=0); p.add_argument("--num-workers",type=int,default=1); p.add_argument("--combine-shards",action="store_true"); a=p.parse_args(); cfg=json.loads(a.config.read_text()); variants=output_variants(cfg); combine(a,cfg,variants) if a.combine_shards else shard(a,cfg,variants); return 0
if __name__=="__main__": raise SystemExit(main())
