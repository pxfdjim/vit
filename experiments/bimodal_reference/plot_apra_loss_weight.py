#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
M=("aff_f1","vus_pr","vus_roc"); TITLES=("Aff-F1","VUS-PR","VUS-ROC")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(); cfg=json.loads(a.config.read_text()); scope=pd.read_csv(a.output_root/"metrics_by_scope.csv"); seeds=pd.read_csv(a.output_root/"multiseed_summary.csv"); out=a.output_root/"figures"; out.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,3,figsize=(13.5,3.8))
 for ax,m,title in zip(axes,M,TITLES):
  for sc,label in (("all_11","All-11"),("real_6","Real-6"),("yahoo_4","Yahoo-4")):
   g=scope[(scope.scope==sc)&scope.variant.str.startswith("G_APRA_")].copy(); g["gamma"]=g.variant.str.extract(r"g(\d+)")[0].astype(int)/1000; g=g.sort_values("gamma"); ax.plot(g.gamma,g[m],marker="o",label=label)
  ax.axvline(.05,color="black",ls="--",lw=1); ax.set(title=title,xlabel="APRA coefficient γ",ylabel=title); ax.grid(alpha=.25)
 axes[0].legend(frameon=False); fig.tight_layout(); fig.savefig(out/"gamma_sensitivity.png",dpi=180); plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13.5,3.8)); labels={"E0_full":"E0 Full","E1_no_path":"E1 No path","E2_path_only":"E2 Path only"}; x=np.arange(3)
 for ax,m,title in zip(axes,M,TITLES):
  g=seeds[seeds.scope=="all_11"].set_index("variant").loc[list(labels)]; mean=g[f"{m}_mean"].to_numpy(); std=g[f"{m}_std"].to_numpy()
  ax.errorbar(x,mean,yerr=std,marker="o",lw=1.8,capsize=5,color="#2f6b9a"); span=max(float((mean+std).max()-(mean-std).min()),1e-4); ax.set_ylim(float((mean-std).min()-0.35*span),float((mean+std).max()+0.35*span)); ax.set_xticks(x,list(labels.values()),rotation=15); ax.set(title=f"All-11 {title}",ylabel=title); ax.grid(axis="y",alpha=.25)
 fig.tight_layout(); fig.savefig(out/"loss_ablation.png",dpi=180); plt.close(fig)
 print(out)
if __name__=="__main__": main()
