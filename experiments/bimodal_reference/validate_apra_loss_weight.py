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
from experiments.bimodal_reference.run_apra_loss_weight_inference import output_variants
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); a=p.parse_args(); cfg=json.loads(a.config.read_text()); variants=output_variants(cfg); errors=[]; checks={}; expected_ckpt=88; paths=sorted((a.output_root/"checkpoints").glob("*/seed_*/*.pt")); checks["checkpoints"]=len(paths)
 if len(paths)!=expected_ckpt: errors.append(f"checkpoints {len(paths)} != {expected_ckpt}")
 hashes={}; gradient_valid=0
 for path in paths:
  x=torch.load(path,map_location="cpu",weights_only=True); key=(x["dataset"],x["seed"]); hashes.setdefault(key,set()).add(x["initialization_sha256"]); v=x["variant"]; audits=x["gradient_audit_by_loss"]; after=x["detect_gradient_after_one_step"]
  good=(audits["path"]["apra"]>0)==bool(cfg["variants"][v]["path"])
  if cfg["variants"][v]["detect"]: good=good and after["apra"]>0 and after["acef"]>0
  else: good=good and after["apra"]==0 and after["acef"]==0
  gradient_valid+=int(good)
 for key,values in hashes.items():
  if len(values)!=1: errors.append(f"initialization mismatch {key}")
 checks["gradient_valid"]=gradient_valid
 if gradient_valid!=len(paths): errors.append("gradient audit")
 metrics=pd.read_csv(a.output_root/"metrics_per_signal.csv"); checks["metric_rows"]=len(metrics); expected_scores=492*len(variants); checks["point_scores"]=sum(1 for _ in (a.output_root/"point_scores").glob("*/*/*.npz"))
 if len(metrics)!=expected_scores or checks["point_scores"]!=expected_scores: errors.append("score/metric count")
 valid=metrics[metrics.excluded_reason.fillna("")==""].groupby("variant").size().to_dict(); checks["valid_signals_per_variant"]=valid
 if any(valid.get(v)!=488 for v in variants): errors.append("valid signal set")
 maxerr={"B3":0.,"gamma0":0.,"gamma05":0.,"E0":0.,"path_only":0.}; apra=Path(cfg["apra_run"]); feature=Path(cfg["feature_scoring_run"])
 for d in DATASETS:
  for s in signal_names(Path(cfg["data_root"]),d):
   with np.load(feature/"point_scores"/d/s/"B3_raw_numeric_l010.npz") as z: b=z["scores"]
   with np.load(apra/"point_scores"/d/s/"APRA_only.npz") as z: e=z["scores"]
   for name,expect,key in (("B3",b,"B3"),("G_APRA_g000",b,"gamma0"),("G_APRA_g050",e,"gamma05"),("E0_full_s20261104",e,"E0")):
    with np.load(a.output_root/"point_scores"/d/s/f"{name}.npz") as z: maxerr[key]=max(maxerr[key],float(np.max(np.abs(z["scores"]-expect))))
   for seed in cfg["seeds"]:
    with np.load(a.output_root/"point_scores"/d/s/f"E2_path_only_s{seed}.npz") as z: maxerr["path_only"]=max(maxerr["path_only"],float(np.max(np.abs(z["scores"]-b))))
 checks["reproduction_max_abs_errors"]=maxerr
 reproduction_tolerance=1e-12; checks["reproduction_tolerance"]=reproduction_tolerance
 if any(x>reproduction_tolerance for x in maxerr.values()): errors.append(f"reproduction {maxerr}")
 manifests=[json.loads(x.read_text()) for x in sorted((a.output_root/"inference_manifests").glob("*.json"))]; records=[r for m in manifests for r in m["records"]]; checks["inference_records"]=len(records)
 if len(records)!=492 or any(r["status"] not in {"passed","skipped_complete"} for r in records): errors.append("inference records")
 result={"status":"passed" if not errors else "failed","errors":errors,"checks":checks,"labels_used_for_training_calibration_weighting_or_selection":False}; (a.output_root/"validation.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2)); return 0 if not errors else 1
if __name__=="__main__": raise SystemExit(main())
