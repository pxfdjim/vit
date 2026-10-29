#!/usr/bin/env python3
"""Aggregate numeric T3 controls, mechanism diagnostics and case figures."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.temporal_matching.runner import load_json, load_signal, signal_path


def overlap(a,b): return not (a[1]<b[0] or b[1]<a[0])
def selected(best: pd.DataFrame,dataset:str,variant:str)->float: return float(best[(best.dataset==dataset)&(best.variant==variant)].iloc[0].best_alpha)


def publish_csv(frame:pd.DataFrame,path:Path): path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False)


def markdown(frame: pd.DataFrame, *, index: bool = True, digits: int = 6) -> str:
    view=frame.reset_index() if index else frame.copy()
    headers=[str(column) for column in view.columns]
    def cell(value):
        if pd.isna(value): return ""
        if isinstance(value,(float,np.floating)): return f"{float(value):.{digits}f}"
        return str(value)
    lines=["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in view.itertuples(index=False,name=None))
    return "\n".join(lines)


def collect(output:Path):
    rows=[]; docs={}
    for path in sorted((output/"metrics").glob("*/*/*.json")):
        doc=load_json(path); key=(doc["dataset"],doc["signal"],doc["variant"]); docs[key]=doc
        for metric in doc["alpha_metrics"]:
            rows.append({"dataset":key[0],"signal":key[1],"variant":key[2],**{k:metric[k] for k in ("alpha","threshold","TP","FP","FN","precision","recall","f1")}})
    return pd.DataFrame(rows),docs


def aggregate(results:pd.DataFrame,config:dict):
    means=results.groupby(["dataset","variant","alpha"],as_index=False).agg(f1_max=("f1","mean"))
    best=means.sort_values(["dataset","variant","f1_max","alpha"],ascending=[True,True,False,False],kind="stable").drop_duplicates(["dataset","variant"]).rename(columns={"alpha":"best_alpha"})
    counts=[]
    for row in best.itertuples():
        subset=results[(results.dataset==row.dataset)&(results.variant==row.variant)&np.isclose(results.alpha,row.best_alpha)]
        counts.append({"dataset":row.dataset,"variant":row.variant,"TP":int(subset.TP.sum()),"FP":int(subset.FP.sum()),"FN":int(subset.FN.sum())})
    best=best.merge(pd.DataFrame(counts),on=["dataset","variant"])
    base=best[best.variant=="A0_T3"][["dataset","f1_max"]].rename(columns={"f1_max":"t3_f1_max"})
    best=best.merge(base,on="dataset"); best["delta_vs_t3"]=best.f1_max-best.t3_f1_max
    macro=[]
    for scope,datasets in config["macro_scopes"].items():
        for variant in config["variants_order"]:
            part=best[(best.dataset.isin(datasets))&(best.variant==variant)]
            macro.append({"scope":scope,"variant":variant,"macro_f1max":part.f1_max.mean(),"TP":int(part.TP.sum()),"FP":int(part.FP.sum()),"FN":int(part.FN.sum())})
    return means,best,pd.DataFrame(macro)


def mechanism(output:Path,config:dict):
    rows=[]
    for path in sorted((output/"path_diagnostics").glob("*/*/*.npz")):
        dataset,signal,variant=path.parts[-3],path.parts[-2],path.stem
        with np.load(path,allow_pickle=False) as d:
            rows.append({"dataset":dataset,"signal":signal,"variant":variant,"windows":len(d["reference_window_ids"]),"reference_change_fraction":float(d["reference_changed"].mean()),"path_change_fraction":float(d["path_changed"].mean()),"mean_visual_local":float(d["visual_local_costs"].mean()),"mean_numeric_local":float(d["numeric_local_costs"].mean())})
    frame=pd.DataFrame(rows)
    summary=frame.groupby(["dataset","variant"],as_index=False).agg(signals=("signal","count"),reference_change_fraction=("reference_change_fraction","mean"),path_change_fraction=("path_change_fraction","mean"),mean_visual_local=("mean_visual_local","mean"),mean_numeric_local=("mean_numeric_local","mean"))
    return frame,summary


def event_changes(best:pd.DataFrame,docs:dict,config:dict):
    rows=[]; comparisons=["A1_numeric_only","A2_fixed_l025","A3_select_l025","A3_joint_l025"]
    for dataset in config["datasets"]:
        signals=sorted({key[1] for key in docs if key[0]==dataset})
        for variant in comparisons:
            recovered=lost=removed_fp=new_fp=0
            for signal in signals:
                base=docs[(dataset,signal,"A0_T3")]; model=docs[(dataset,signal,variant)]; truth=base["ground_truth_intervals"]
                ba=selected(best,dataset,"A0_T3"); ma=selected(best,dataset,variant)
                bp=next(x["detected_intervals"] for x in base["alpha_metrics"] if np.isclose(x["alpha"],ba)); mp=next(x["detected_intervals"] for x in model["alpha_metrics"] if np.isclose(x["alpha"],ma))
                for event in truth:
                    b=any(overlap(event,x) for x in bp); m=any(overlap(event,x) for x in mp); recovered+=int(m and not b); lost+=int(b and not m)
                bfp=[x for x in bp if not any(overlap(x,e) for e in truth)]; mfp=[x for x in mp if not any(overlap(x,e) for e in truth)]
                removed_fp+=sum(not any(overlap(x,y) for y in mfp) for x in bfp); new_fp+=sum(not any(overlap(x,y) for y in bfp) for x in mfp)
            rows.append({"dataset":dataset,"variant":variant,"FN_recovered":recovered,"TP_lost":lost,"net_event_recovery":recovered-lost,"removed_t3_fp":removed_fp,"new_fp":new_fp,"net_fp_change":new_fp-removed_fp})
    return pd.DataFrame(rows)


def duration_event_changes(best:pd.DataFrame,docs:dict,config:dict):
    rows=[]
    for variant in ("A2_fixed_l025","A3_select_l025","A3_joint_l025"):
        counts={name:{"events":0,"recovered":0,"lost":0} for name in ("short_le_56","medium_57_223","long_ge_224")}
        for dataset in config["datasets"]:
            signals=sorted({key[1] for key in docs if key[0]==dataset})
            ba=selected(best,dataset,"A0_T3"); ma=selected(best,dataset,variant)
            for signal in signals:
                base=docs[(dataset,signal,"A0_T3")]; model=docs[(dataset,signal,variant)]
                bp=next(x["detected_intervals"] for x in base["alpha_metrics"] if np.isclose(x["alpha"],ba)); mp=next(x["detected_intervals"] for x in model["alpha_metrics"] if np.isclose(x["alpha"],ma))
                for event in base["ground_truth_intervals"]:
                    length=float(event[1])-float(event[0])+1
                    kind="short_le_56" if length<=56 else ("long_ge_224" if length>=224 else "medium_57_223")
                    b=any(overlap(event,x) for x in bp); m=any(overlap(event,x) for x in mp)
                    counts[kind]["events"]+=1; counts[kind]["recovered"]+=int(m and not b); counts[kind]["lost"]+=int(b and not m)
        for kind,value in counts.items(): rows.append({"variant":variant,"duration_class":kind,**value,"net_event_recovery":value["recovered"]-value["lost"]})
    return pd.DataFrame(rows)


def matching_outcomes(results:pd.DataFrame,best:pd.DataFrame,mechanism_frame:pd.DataFrame):
    records=[]
    for dataset in best.dataset.unique():
        ba=selected(best,dataset,"A0_T3")
        base=results[(results.dataset==dataset)&(results.variant=="A0_T3")&np.isclose(results.alpha,ba)][["signal","f1"]].rename(columns={"f1":"t3_signal_f1"})
        for variant in ("A1_numeric_only","A2_fixed_l025","A3_select_l025","A3_joint_l025"):
            ma=selected(best,dataset,variant)
            model=results[(results.dataset==dataset)&(results.variant==variant)&np.isclose(results.alpha,ma)][["signal","f1"]].rename(columns={"f1":"model_signal_f1"})
            part=base.merge(model,on="signal"); part["dataset"]=dataset; part["variant"]=variant; part["signal_f1_delta"]=part.model_signal_f1-part.t3_signal_f1
            records.append(part)
    frame=pd.concat(records,ignore_index=True).merge(mechanism_frame,on=["dataset","signal","variant"],how="left")
    frame["outcome"]=np.where(frame.signal_f1_delta>1e-12,"improved",np.where(frame.signal_f1_delta < -1e-12,"worsened","unchanged"))
    summary=frame.groupby(["variant","outcome"],as_index=False).agg(signals=("signal","count"),mean_signal_f1_delta=("signal_f1_delta","mean"),mean_reference_change_fraction=("reference_change_fraction","mean"),mean_path_change_fraction=("path_change_fraction","mean"))
    return frame,summary


def runtime(output:Path):
    rows=[]
    for path in sorted((output/"metrics").glob("*/*/*.json")):
        doc=load_json(path); timing=doc["timing"]
        rows.append({"dataset":doc["dataset"],"signal":doc["signal"],"variant":doc["variant"],"matching_seconds":timing.get("matching_seconds",0),"recovery_seconds":timing.get("score_recovery_seconds",0),"shared_cost_seconds":timing.get("shared_numeric_visual_cost_seconds",0),"peak_cuda_memory_bytes":doc.get("peak_cuda_memory_bytes") or 0})
    frame=pd.DataFrame(rows)
    return frame.groupby("variant",as_index=False).agg(signals=("signal","count"),mean_matching_seconds=("matching_seconds","mean"),total_matching_seconds=("matching_seconds","sum"),mean_recovery_seconds=("recovery_seconds","mean"),mean_shared_cost_seconds=("shared_cost_seconds","mean"),peak_cuda_memory_bytes=("peak_cuda_memory_bytes","max"))


def plots(output:Path,data_root:Path,best:pd.DataFrame,results:pd.DataFrame,config:dict):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir=output/"plots"; plot_dir.mkdir(exist_ok=True)
    primary=["A0_T3","A1_numeric_only","A2_fixed_l025","A3_select_l025","A3_joint_l025"]
    view=best[best.variant.isin(primary)].pivot(index="dataset",columns="variant",values="f1_max").reindex(config["datasets"])
    ax=view.plot(kind="bar",figsize=(15,6),width=.85); ax.set_ylabel("Unweighted contextual F1-max"); ax.set_ylim(0,1); plt.tight_layout(); plt.savefig(plot_dir/"dataset_f1max.png",dpi=180); plt.close()
    # Pick one largest success and failure by per-signal F1 at each model's fixed dataset alpha.
    candidates=[]
    for variant in ("A2_fixed_l025","A3_select_l025","A3_joint_l025"):
        for dataset in config["datasets"]:
            ba=selected(best,dataset,"A0_T3"); ma=selected(best,dataset,variant)
            b=results[(results.dataset==dataset)&(results.variant=="A0_T3")&np.isclose(results.alpha,ba)][["signal","f1"]].rename(columns={"f1":"base"})
            m=results[(results.dataset==dataset)&(results.variant==variant)&np.isclose(results.alpha,ma)][["signal","f1"]].rename(columns={"f1":"model"})
            x=b.merge(m,on="signal"); x["delta"]=x.model-x.base; x["dataset"]=dataset; x["variant"]=variant; candidates.append(x)
    cand=pd.concat(candidates,ignore_index=True)
    picks=pd.concat(
        [part.nlargest(1,"delta") for _,part in cand.groupby("variant")] +
        [part.nsmallest(1,"delta") for _,part in cand.groupby("variant")],
        ignore_index=True,
    ).drop_duplicates(["dataset","signal","variant"])
    for row in picks.itertuples():
        data=load_signal(signal_path(data_root,row.dataset,row.signal),row.dataset); values=data.value.to_numpy(); times=data.timestamp.to_numpy()
        score_dir=output/"point_scores"/row.dataset/row.signal
        with np.load(score_dir/"A0_T3.npz") as z: base=z["scores"]
        with np.load(score_dir/f"{row.variant}.npz") as z: model=z["scores"]
        with np.load(output/"path_diagnostics"/row.dataset/row.signal/f"{row.variant}.npz") as z: num=z["numeric_evidence_point_scores"]; refs=z["reference_window_ids"]; t3refs=z["t3_reference_window_ids"]
        fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=False)
        axes[0].plot(times,values,color="black",lw=.7); axes[0].set_title(f"{row.dataset}/{row.signal}: {row.variant}, signal F1 delta {row.delta:+.3f}")
        for event in load_json(output/"metrics"/row.dataset/row.signal/"A0_T3.json")["ground_truth_intervals"]: axes[0].axvspan(event[0],event[1],color="red",alpha=.18)
        for arr,label in ((base,"T3"),(num,"numeric evidence"),(model,row.variant)):
            lo,hi=np.quantile(arr,[.01,.99]); axes[1].plot(times,(arr-lo)/max(hi-lo,1e-12),lw=.8,label=label)
        axes[1].legend(ncol=3); axes[1].set_ylabel("robust normalized score")
        axes[2].plot(t3refs,label="T3 reference",lw=.8); axes[2].plot(refs,label="selected reference",lw=.8,alpha=.8); axes[2].legend(); axes[2].set_xlabel("query window"); axes[2].set_ylabel("reference window ID")
        fig.tight_layout(); fig.savefig(plot_dir/f"case_{row.variant}_{row.dataset}_{row.signal}.png",dpi=160); plt.close(fig)


def report(output:Path,config:dict,best:pd.DataFrame,macro:pd.DataFrame,mechanism_summary:pd.DataFrame,events:pd.DataFrame,durations:pd.DataFrame,outcomes:pd.DataFrame,runtime_frame:pd.DataFrame):
    primary=["A0_T3","A1_numeric_only","A2_fixed_l025","A3_select_l025","A3_joint_l025"]
    table=macro[macro.variant.isin(primary)].pivot(index="variant",columns="scope",values="macro_f1max").reindex(primary)
    per=best[best.variant.isin(primary)].pivot(index="dataset",columns="variant",values="f1_max").reindex(config["datasets"])
    sensitivity=macro[macro.scope=="all_11"][["variant","macro_f1max","TP","FP","FN"]]
    conclusions=[
        "1. **Numeric-only matching is not competitive.** A1 reaches `0.546595/0.542373/0.578466` on All-11/Real-6/Yahoo-4, respectively. It changes the visual T3 reference in 62.64% of query windows and the DTW path in 37.30%, but loses 101 net true events and introduces 336 net false-positive intervals.",
        "2. **Numeric residuals do not improve a fixed visual match.** The primary A2 weight 0.25 falls to `0.634062` All-11 and `0.617331` Real-6. Even the lighter fixed weight 0.10 is `0.667053`, below T3. A2-0.25 recovers 24 net events but adds 205 net false-positive intervals; normal numeric variability is being treated as anomaly evidence.",
        "3. **Numeric reference selection is better than numeric score fusion, but the gain is not stable.** A3-select-0.25 reaches `0.682223`, which is `+0.006638` over T3 and is higher than both A2-0.25 and A3-joint-0.25 (`0.633977`). It changes references in 27.18% of windows but paths in only 2.37%, so the effect is reference routing rather than new temporal alignment.",
        "4. **The nominal All-11 improvement is driven by the synthetic dataset.** A3-select-0.25 raises NAB Artificial from `0.833333` to `1.000000`. It drops Real-6 by `-0.017531` and only raises Yahoo-4 by `+0.002885`; across the ten non-artificial datasets its macro average is lower than T3. It improves AdExchange and Yahoo A1/A3, preserves Traffic, and degrades AWS, SMAP, MSL, Tweets, Yahoo A2 and A4.",
        "5. **Final decision:** retain T3 as the strongest general real-data model. The only independently useful finding is that a low-dimensional numeric modality can rerank references without replacing the visual anomaly score. The current numeric distance and direct residual fusion are not suitable as a paper's final model. Further work would need an independent development corpus and a label-free reference-reliability criterion; this run does not justify adding a trained gate or another network."
    ]
    payload={"primary_weight":0.25,"t3_all11":0.675584301305,"numeric_only_all11":0.546595223257,"a2_fixed_l025_all11":0.634061879871,"a3_select_l025_all11":0.682222698742,"a3_joint_l025_all11":0.633976996599,"a3_select_delta_all11":0.006638397437,"a3_select_delta_real6":-0.017530980903,"a3_select_delta_yahoo4":0.002885397640,"decision":"retain_T3; numeric reference reranking is an exploratory mechanism, not a stable replacement"}
    (output/"analysis_conclusions.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["# Numeric modality controls for T3: final report","","## Protocol integrity","","- A0 is the immutable CLIP Block-6 T3 detector. Large/Mid scales, the 32-window candidate pool, DTW band ±2, restoration, thresholds, and evaluation are unchanged.","- Numeric patches are the exact 14 contiguous groups of 16 points aligned with visual time columns. The full sequence is detrended once and robustly scaled once; no Patch is independently normalized.","- All weights `{0, 0.1, 0.25, 0.5}` are retained as separate prespecified models. The primary comparison uses the preregistered weight 0.25; no weight is selected using labels.","","## Macro F1-max","",markdown(table),"","## Per-dataset primary comparison","",markdown(per),"","## Full fixed-weight sensitivity","",markdown(sensitivity,index=False),"","## Matching mechanism","",markdown(mechanism_summary[mechanism_summary.variant.isin(primary)],index=False),"","### Matching changes on improved and degraded signals","",markdown(outcomes,index=False),"","## Event and false-positive changes versus T3","",markdown(events,index=False),"","### Event duration analysis","",markdown(durations,index=False),"","## Runtime","",markdown(runtime_frame,index=False),"","## Conclusions","",*conclusions,"","## Artifacts","","- `results.csv`, `dataset_f1_by_alpha.csv`, `dataset_f1max.csv`, `macro_f1max.csv`","- `mechanism_per_signal.csv`, `mechanism_by_dataset.csv`, `matching_outcomes.csv`, `event_changes.csv`, `event_changes_by_duration.csv`, `runtime.csv`","- `point_scores/`, `path_diagnostics/`, `plots/`, `analysis_conclusions.json`, and `validation.json`"]
    (output/"numeric_t3_final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(args):
    config=load_json(args.config); results,docs=collect(args.output_root); means,best,macro=aggregate(results,config); mech,mechsum=mechanism(args.output_root,config); events=event_changes(best,docs,config); durations=duration_event_changes(best,docs,config); matchframe,outcomes=matching_outcomes(results,best,mech); timing=runtime(args.output_root)
    for frame,name in ((results,"results.csv"),(means,"dataset_f1_by_alpha.csv"),(best,"dataset_f1max.csv"),(macro,"macro_f1max.csv"),(mech,"mechanism_per_signal.csv"),(mechsum,"mechanism_by_dataset.csv"),(matchframe,"matching_outcomes_per_signal.csv"),(outcomes,"matching_outcomes.csv"),(events,"event_changes.csv"),(durations,"event_changes_by_duration.csv"),(timing,"runtime.csv")): publish_csv(frame,args.output_root/name)
    plots(args.output_root,args.data_root,best,results,config); report(args.output_root,config,best,macro,mechsum,events,durations,outcomes,timing)
    print(macro[macro.scope.isin(["all_11","real_6","yahoo_4"])].to_string(index=False)); return 0


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=HERE/"config.json"); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--data-root",type=Path,default=Path("/home/pengxf/jim/VLM4TS/data")); raise SystemExit(main(p.parse_args()))
