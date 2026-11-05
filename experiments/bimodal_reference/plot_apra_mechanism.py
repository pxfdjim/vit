#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS=("aff_f1","vus_pr","vus_roc")
ORDER=("E1_existing","E2_select","E3_no_reject","E4_no_warp","E5_fixed_dtw",
       "E6_detect_only","E7_no_query_reject","E8_no_reference_skip",
       "E9_no_structural_residual","E10_match_only")
SHORT={"E1_existing":"E1","E2_select":"E2","E3_no_reject":"E3","E4_no_warp":"E4",
       "E5_fixed_dtw":"E5","E6_detect_only":"E6","E7_no_query_reject":"E7",
       "E8_no_reference_skip":"E8","E9_no_structural_residual":"E9","E10_match_only":"E10"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
    out=a.output_root/"figures"; out.mkdir(exist_ok=True)
    multi=pd.read_csv(a.output_root/"multiseed_summary.csv"); scope=pd.read_csv(a.output_root/"metrics_by_scope.csv")
    b3=scope[(scope.scope=="all_11")&(scope.variant=="B3")].iloc[0]
    data=multi[multi.scope.eq("all_11")].set_index("variant").loc[list(ORDER)]
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    colors=["#667085"]*5+["#175CD3","#7A5AF8","#9E77ED","#039855","#D92D20"]
    for ax,metric in zip(axes,METRICS):
        values=data[f"{metric}_mean"]-b3[metric]
        ax.bar(np.arange(len(ORDER)),values,yerr=data[f"{metric}_std"],color=colors,capsize=2)
        ax.axhline(0,color="black",linewidth=.8); ax.set_xticks(np.arange(len(ORDER)),[SHORT[x] for x in ORDER])
        ax.set_title(f"All-11 Δ{metric.upper()} vs B3"); ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(out/"mechanism_metric_deltas.png",dpi=180); plt.close(fig)
    diag=pd.read_csv(a.output_root/"path_diagnostics_summary.csv")
    primary=diag[diag.seed.eq(20261104)].set_index("variant").loc[list(ORDER)]
    columns=("match_probability","rejection","reference_skip","path_entropy","alignment_displacement")
    fig,axes=plt.subplots(1,len(columns),figsize=(18,4.4))
    for ax,column in zip(axes,columns):
        ax.bar(np.arange(len(ORDER)),primary[column],color=colors)
        ax.set_xticks(np.arange(len(ORDER)),[SHORT[x] for x in ORDER],rotation=45)
        ax.set_title(column.replace("_"," ")); ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(out/"mechanism_path_states.png",dpi=180); plt.close(fig)
    print(out)

if __name__=="__main__": main()
