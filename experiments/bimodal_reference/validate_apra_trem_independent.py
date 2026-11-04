#!/usr/bin/env python3
"""Independent integrity validation for frozen APRA/TREM experiment."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.bimodal_reference.orionbench_protocol import signal_names
from experiments.bimodal_reference.run_apra_trem_independent_inference import output_variants,beta_code
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(); cfg=json.loads(a.config.read_text()); root=a.output_root; variants=output_variants(cfg); errors=[]; checks={}
 signals=sum(len(signal_names(Path(cfg["data_root"]),d)) for d in cfg["datasets"]); expected_checkpoints=len(cfg["datasets"])*(len(cfg["variants"])+(len(cfg["seeds"])-1)*len(cfg["multiseed_variants"])); paths=list((root/"checkpoints").glob("*/seed_*/*.pt"))
 if len(paths)!=expected_checkpoints: errors.append(f"checkpoints {len(paths)} != {expected_checkpoints}")
 frozen=[]; gradient=[]; zero=[]; effects=[]
 for path in paths:
  x=torch.load(path,map_location="cpu",weights_only=True); spec=cfg["variants"][x["variant"]]
  if x["frozen_base_max_abs_change"]!=0: frozen.append(str(path))
  if x["zero_initialization_error"]!=0: zero.append(str(path))
  g=x["gradient_after_one_step"]
  if g["output"]<=0 or g["projection"]<=0 or (spec["use_local"] and g["local"]<=0) or (spec["use_multiscale"] and g["multiscale"]<=0) or (spec["use_reliability"] and g["reliability_fusion"]<=0): gradient.append(str(path))
  if x["input_effect"]<=0 or (spec["use_reliability"] and x["reliability_effect"]<=0): effects.append(str(path))
 if frozen: errors.append(f"frozen base changed {frozen[:3]}")
 if zero: errors.append(f"zero equivalence {zero[:3]}")
 if gradient: errors.append(f"gradient paths {gradient[:3]}")
 if effects: errors.append(f"input effects {effects[:3]}")
 score_paths=list((root/"point_scores").glob("*/*/*.npz")); expected_scores=signals*len(variants)
 if len(score_paths)!=expected_scores: errors.append(f"scores {len(score_paths)} != {expected_scores}")
 metrics=pd.read_csv(root/"metrics_per_signal.csv")
 if len(metrics)!=expected_scores: errors.append(f"metrics {len(metrics)} != {expected_scores}")
 if metrics.groupby(["dataset","signal"]).variant.nunique().min()!=len(variants): errors.append("incomplete variant coverage")
 valid=metrics.groupby("variant").aff_f1.apply(lambda x:int(np.isfinite(x).sum()))
 if not valid.eq(488).all(): errors.append(f"valid signals {valid.to_dict()}")
 alignment=pd.read_csv(root/"alignment_audit.csv")
 if len(alignment)!=expected_scores or not alignment.valid.all(): errors.append("alignment audit")
 maximum={}
 sources={"E0_B3":(Path(cfg["feature_scoring_run"]),"B3_raw_numeric_l010"),"E1_APRA_only":(Path(cfg["apra_run"]),"APRA_only")}
 for name,(source,source_name) in sources.items():
  value=0.
  for dataset in cfg["datasets"]:
   for signal in signal_names(Path(cfg["data_root"]),dataset):
    with np.load(root/"point_scores"/dataset/signal/f"{name}.npz") as got,np.load(source/"point_scores"/dataset/signal/f"{source_name}.npz") as expected: value=max(value,float(np.max(np.abs(got["scores"]-expected["scores"]))))
  maximum[name]=value
  if value!=0: errors.append(f"baseline reproduction {name} {value}")
 beta0=[v for v in variants if v.endswith("_b000")]; beta0_error=0.
 for variant in beta0:
  for dataset in cfg["datasets"]:
   for signal in signal_names(Path(cfg["data_root"]),dataset):
    with np.load(root/"point_scores"/dataset/signal/f"{variant}.npz") as got,np.load(root/"point_scores"/dataset/signal/"E1_APRA_only.npz") as expected: beta0_error=max(beta0_error,float(np.max(np.abs(got["scores"]-expected["scores"]))))
 if beta0_error!=0: errors.append(f"beta zero mismatch {beta0_error}")
 manifests=[json.loads(x.read_text()) for x in sorted((root/"inference_manifests").glob("*.json"))]
 records=[r for m in manifests for r in m["records"]]
 if any(m["status"]!="passed" for m in manifests) or len(records)!=signals: errors.append("inference manifests")
 if any(r.get("t3_reference_error",0)!=0 or r.get("t3_path_error",0)!=0 for r in records): errors.append("T3 audit")
 diagnostics=pd.read_csv(root/"inference_diagnostics_per_signal.csv")
 if not (diagnostics.calibration_scale>=cfg["calibration"]["scale_floor"]).all(): errors.append("calibration floor")
 checks={"signals":signals,"variants":len(variants),"checkpoints":len(paths),"point_scores":len(score_paths),"metric_rows":len(metrics),"valid_signals_per_variant":valid.to_dict(),"frozen_base_valid":len(paths)-len(frozen),"gradient_valid":len(paths)-len(gradient),"input_effect_valid":len(paths)-len(effects),"baseline_reproduction_max_abs_errors":maximum,"beta_zero_reproduction_max_abs_error":beta0_error,"inference_records":len(records)}
 result={"status":"passed" if not errors else "failed","errors":errors,"checks":checks,"labels_used_for_training_calibration_weighting_or_selection":False}; (root/"validation.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); return 0 if not errors else 1
if __name__=="__main__": raise SystemExit(main())
