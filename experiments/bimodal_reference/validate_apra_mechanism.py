#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.bimodal_reference.orionbench_protocol import DATASETS,signal_names
from experiments.bimodal_reference.run_apra_mechanism_inference import output_variants

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args()
 cfg=json.loads(a.config.read_text()); variants=output_variants(cfg); errors=[]; checks={}
 paths=sorted((a.output_root/"checkpoints").glob("*/seed_*/*.pt")); expected=4*3*11; checks["checkpoints"]=len(paths)
 if len(paths)!=expected: errors.append(f"checkpoints {len(paths)} != {expected}")
 hashes={}; gradients=0; mechanism=0
 for path in paths:
  x=torch.load(path,map_location="cpu",weights_only=True); v=x["variant"]; spec=cfg["variants"][v]
  hashes.setdefault((x["dataset"],x["seed"]),set()).add(x["initialization_sha256"])
  audit=x["gradient_audit_by_loss"]; after=x["detect_gradient_after_one_step"]
  good=after["acef"]>0 and after["numeric"]>0
  if spec["reject"] and spec["skip"]: good=good and after["apra"]>0
  gradients+=int(good)
  ma=x.get("mechanism_audit",{}); conserved=(ma.get("clean_row_conservation_error",1)<2e-5 and ma.get("clean_column_conservation_error",1)<2e-5)
  disabled=((spec["reject"] or ma.get("clean_rejection",1)<1e-6) and (spec["skip"] or ma.get("clean_reference_skip",1)<1e-6))
  mechanism+=int(bool(ma) and np.isfinite(list(ma.values())).all() and conserved and disabled)
 for key,values in hashes.items():
  if len(values)!=1: errors.append(f"initialization mismatch {key}")
 checks["gradient_valid"]=gradients; checks["mechanism_audit_valid"]=mechanism
 if gradients!=len(paths): errors.append("gradient audit")
 if mechanism!=len(paths): errors.append("mechanism audit")
 metrics=pd.read_csv(a.output_root/"metrics_per_signal.csv"); expected_rows=492*len(variants)
 scores=sum(1 for _ in (a.output_root/"point_scores").glob("*/*/*.npz")); checks.update(metric_rows=len(metrics),point_scores=scores)
 if len(metrics)!=expected_rows or scores!=expected_rows: errors.append("score/metric count")
 valid=metrics[metrics.excluded_reason.fillna("")==""].groupby("variant").size().to_dict(); checks["valid_signals_per_variant"]=valid
 if any(valid.get(v)!=488 for v in variants): errors.append("valid signal set")
 maxerr={"B3":0.,"E1_primary":0.,"reused_E2_E5":0.,"reused_E6":0.}; feature=Path(cfg["feature_scoring_run"]); apra=Path(cfg["apra_run"])
 for d in DATASETS:
  for s in signal_names(Path(cfg["data_root"]),d):
   with np.load(feature/"point_scores"/d/s/"B3_raw_numeric_l010.npz") as z: b=z["scores"]
   with np.load(apra/"point_scores"/d/s/"APRA_only.npz") as z: e=z["scores"]
   with np.load(a.output_root/"point_scores"/d/s/"B3.npz") as z: maxerr["B3"]=max(maxerr["B3"],float(np.max(np.abs(z["scores"]-b))))
   with np.load(a.output_root/"point_scores"/d/s/f"E1_existing_s{cfg['primary_seed']}.npz") as z: maxerr["E1_primary"]=max(maxerr["E1_primary"],float(np.max(np.abs(z["scores"]-e))))
   for variant in ("E2_select","E3_no_reject","E4_no_warp","E5_fixed_dtw"):
    for seed in cfg["seeds"]:
     with np.load(Path(cfg["selective_run"])/"point_scores"/d/s/f"{variant}_s{seed}.npz") as old, np.load(a.output_root/"point_scores"/d/s/f"{variant}_s{seed}.npz") as new:
      maxerr["reused_E2_E5"]=max(maxerr["reused_E2_E5"],float(np.max(np.abs(old["scores"]-new["scores"]))))
   for seed in cfg["seeds"]:
    with np.load(Path(cfg["loss_weight_run"])/"point_scores"/d/s/f"E1_no_path_s{seed}.npz") as old, np.load(a.output_root/"point_scores"/d/s/f"E6_detect_only_s{seed}.npz") as new:
     maxerr["reused_E6"]=max(maxerr["reused_E6"],float(np.max(np.abs(old["scores"]-new["scores"]))))
 checks["reproduction_max_abs_errors"]=maxerr
 if any(x>1e-12 for x in maxerr.values()): errors.append(f"reproduction {maxerr}")
 state=pd.read_csv(a.output_root/"real_state_separation_per_signal.csv"); checks["state_rows"]=len(state)
 if len(state)!=492*10: errors.append("state separation rows")
 manifests=[json.loads(x.read_text()) for x in sorted((a.output_root/"inference_manifests").glob("*.json"))]; records=[r for m in manifests for r in m["records"]]
 checks["inference_records"]=len(records)
 if len(records)!=492 or any(x["status"] not in {"passed","skipped_complete"} for x in records): errors.append("inference records")
 result={"status":"passed" if not errors else "failed","errors":errors,"checks":checks,
         "labels_used_for_training_calibration_weighting_or_selection":False}
 (a.output_root/"validation.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); return 0 if not errors else 1
if __name__=="__main__": raise SystemExit(main())
