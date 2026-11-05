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
LABELS={"E1_existing":"E1 Existing APRA","E2_select":"E2 L_select","E3_no_reject":"E3 No reject sup.","E4_no_warp":"E4 No warp sup.","E5_fixed_dtw":"E5 Fixed DTW"}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(); root=a.output_root; out=root/"figures"; out.mkdir(exist_ok=True)
 seed=pd.read_csv(root/"multiseed_summary.csv"); order=list(LABELS); scopes=["all_11","real_6","yahoo_4"]
 fig,axes=plt.subplots(1,3,figsize=(15,4.4))
 for ax,metric in zip(axes,METRICS):
  x=np.arange(len(scopes)); width=.15
  for i,v in enumerate(order):
   g=seed[seed.variant.eq(v)].set_index("scope").loc[scopes]
   ax.bar(x+(i-2)*width,g[f"{metric}_mean"],width,yerr=g[f"{metric}_std"],capsize=2,label=LABELS[v])
  ax.set_xticks(x,scopes); ax.set_title(metric.replace("_"," ").upper()); ax.grid(axis="y",alpha=.2)
 axes[0].legend(fontsize=8,loc="best"); fig.tight_layout(); fig.savefig(out/"selective_metrics.png",dpi=180); plt.close(fig)
 train=pd.read_csv(root/"training_checkpoint_summary.csv")
 mech=train.groupby("variant",as_index=False).mean(numeric_only=True).set_index("variant")
 variants=[v for v in order[1:] if v in mech.index]
 columns=["mechanism_synthetic_structural_rejection","mechanism_synthetic_matchable_rejection","mechanism_synthetic_unchanged_rejection","mechanism_warp_target_match_mass"]
 fig,ax=plt.subplots(figsize=(10,4.8)); x=np.arange(len(variants)); width=.2
 for i,c in enumerate(columns): ax.bar(x+(i-1.5)*width,mech.loc[variants,c],width,label=c.removeprefix("mechanism_").replace("_"," "))
 ax.set_xticks(x,[LABELS[v] for v in variants],rotation=12); ax.set_ylabel("Mean probability / mass"); ax.grid(axis="y",alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"selective_path_behavior.png",dpi=180); plt.close(fig)
 print(out)
if __name__=="__main__": main()
