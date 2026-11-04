#!/usr/bin/env python3
"""Independent completeness, leakage, gradient, and score validation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.bimodal_reference.orionbench_protocol import signal_names
from experiments.bimodal_reference.run_apra_acef_inference import output_variants

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
 cfg=json.loads(a.config.read_text()); root=a.output_root; variants=output_variants(cfg); errors=[]; checks={}
 signals=sum(len(signal_names(Path(cfg["data_root"]),d)) for d in cfg["datasets"])
 checkpoints=list((root/"checkpoints").glob("*/*.pt")); expected_checkpoints=len(cfg["variants"])*len(cfg["datasets"])
 if len(checkpoints)!=expected_checkpoints: errors.append(f"checkpoint count {len(checkpoints)} != {expected_checkpoints}")
 gradient_errors=[]; zero_errors=[]; active_loss_errors=[]; reference_effects=[]
 for path in checkpoints:
  payload=torch.load(path,map_location="cpu",weights_only=True); spec=cfg["variants"][payload["variant"]]
  if payload["labels_used"] or payload["validation_split_used"] or payload["checkpoint_selection"]!="fixed_final_epoch": errors.append(f"selection/leakage {path}")
  if not payload["zero_initialized_correction"]: zero_errors.append(str(path))
  expected=[name for name in ("path","explain","detect") if spec[name]]
  if payload["active_losses"]!=expected: active_loss_errors.append(str(path))
  audit=payload["gradient_audit_by_loss"]
  # With rejection disabled and equal query/reference lengths, the monotone
  # lattice has a single feasible all-match path.  L_path is then a constant
  # diagnostic by construction and has no trainable APRA path to audit.
  if spec["path"] and spec["reject"] and audit["path"]["apra"]<=0: gradient_errors.append(f"{path}:L_path")
  if spec["explain"] and audit["explain"]["acef"]<=0: gradient_errors.append(f"{path}:L_explain")
  if spec["detect"]:
   step=payload["detect_gradient_after_one_step"]
   if step["acef"]<=0: gradient_errors.append(f"{path}:L_detect/fusion")
   if spec["aligner"]!="fixed_dtw" and step["apra"]<=0: gradient_errors.append(f"{path}:L_detect/aligner")
  if spec["random_numeric"] and payload["numeric_max_abs_change"]!=0: gradient_errors.append(f"{path}:random numeric changed")
  reference_effects.append(payload["reference_effect"])
 if zero_errors: errors.append(f"zero initialization failures {len(zero_errors)}")
 if active_loss_errors: errors.append(f"active loss failures {len(active_loss_errors)}")
 if gradient_errors: errors.append(f"gradient failures {gradient_errors[:8]}")
 checks.update({"checkpoints":len(checkpoints),"zero_initialized_checkpoints":len(checkpoints)-len(zero_errors),"minimum_final_reference_effect":float(min(reference_effects)) if reference_effects else None})
 score_paths=list((root/"point_scores").glob("*/*/*.npz")); expected_scores=signals*len(variants)
 if len(score_paths)!=expected_scores: errors.append(f"score files {len(score_paths)} != {expected_scores}")
 metrics=pd.read_csv(root/"metrics_per_signal.csv")
 if len(metrics)!=expected_scores: errors.append(f"metric rows {len(metrics)} != {expected_scores}")
 if metrics.groupby(["dataset","signal"]).variant.nunique().min()!=len(variants): errors.append("variant coverage incomplete")
 valid=metrics.groupby("variant").aff_f1.apply(lambda x:int(np.isfinite(x).sum()))
 if not valid.eq(488).all(): errors.append(f"common valid mismatch {valid.to_dict()}")
 alignment=pd.read_csv(root/"alignment_audit.csv")
 if len(alignment)!=expected_scores or not alignment.valid.all(): errors.append("timestamp/score alignment")
 checks.update({"point_scores":len(score_paths),"metric_rows":len(metrics),"valid_signals_per_variant":valid.to_dict()})
 manifests=[json.loads(x.read_text()) for x in sorted((root/"inference_manifests").glob("worker_*_of_08.json"))]
 if len(manifests)!=8 or any(x["status"]!="passed" for x in manifests): errors.append("inference manifests")
 records=[r for m in manifests for r in m["records"]]
 if len(records)!=signals or any(r.get("t3_reference_error",0)!=0 or r.get("t3_path_error",0)!=0 for r in records): errors.append("fixed T3 audit")
 source={"T3":(Path(cfg["feature_scoring_run"]),"T3"),"E2":(Path(cfg["feature_scoring_run"]),"E2_T3_raw_l010"),"B3":(Path(cfg["feature_scoring_run"]),"B3_raw_numeric_l010"),"RCCA_CADM_M1":(Path(cfg["rcca_run"]),"M1_full"),"C2_simplified":(Path(cfg["training_ablation_run"]),"C2_no_cross")}
 reproduction={}; no_detect=0.0
 for name,(src,src_name) in source.items():
  maximum=0.0
  for dataset in cfg["datasets"]:
   for signal in signal_names(Path(cfg["data_root"]),dataset):
    with np.load(root/"point_scores"/dataset/signal/f"{name}.npz") as got,np.load(src/"point_scores"/dataset/signal/f"{src_name}.npz") as expected:
     maximum=max(maximum,float(np.max(np.abs(got["scores"]-expected["scores"]))))
  reproduction[name]=maximum
  if maximum!=0: errors.append(f"baseline reproduction {name}: {maximum}")
 for dataset in cfg["datasets"]:
  for signal in signal_names(Path(cfg["data_root"]),dataset):
   with np.load(root/"point_scores"/dataset/signal/"no_Ldetect.npz") as got,np.load(root/"point_scores"/dataset/signal/"B3.npz") as base: no_detect=max(no_detect,float(np.max(np.abs(got["scores"]-base["scores"]))))
 if no_detect!=0: errors.append(f"no L_detect does not equal B3: {no_detect}")
 checks.update({"baseline_reproduction_max_abs_errors":reproduction,"no_Ldetect_max_abs_error_vs_B3":no_detect,"inference_records":len(records)})
 result={"status":"passed" if not errors else "failed","errors":errors,"checks":checks,"labels_used_for_training_calibration_or_selection":False}
 (root/"validation.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); return 0 if not errors else 1
if __name__=="__main__": raise SystemExit(main())
