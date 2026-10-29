#!/usr/bin/env python3
"""Strict integrity and exact-recovery checks for numeric T3 controls."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments.temporal_matching.runner import load_json, publish_json_no_replace


def score(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as item:
        value = item["scores"].astype(np.float64)
    if not np.isfinite(value).all(): raise RuntimeError(f"non-finite score {path}")
    return value


def run(args: argparse.Namespace) -> int:
    config=load_json(args.config); output=args.output_root
    expected_signals=sum(int(config["expected_signal_counts"][d]) for d in config["datasets"])
    expected=expected_signals*len(config["variants_order"])
    metrics=sorted((output/"metrics").glob("*/*/*.json")); scores=sorted((output/"point_scores").glob("*/*/*.npz")); diagnostics=sorted((output/"path_diagnostics").glob("*/*/*.npz"))
    if (len(metrics),len(scores),len(diagnostics)) != (expected,expected,expected):
        raise RuntimeError(f"expected {expected} files, got metrics={len(metrics)}, scores={len(scores)}, diagnostics={len(diagnostics)}")
    max_old=max_zero=max_ref=max_path=0.0; pools={}; checked=0
    for a0 in sorted((output/"point_scores").glob("*/*/A0_T3.npz")):
        dataset,signal=a0.parts[-3],a0.parts[-2]; base=score(a0)
        old=args.previous_root/"point_scores"/dataset/signal/"T3_constrained_dtw.npz"
        if not old.is_file(): raise FileNotFoundError(old)
        prior=score(old)
        if base.shape != prior.shape: raise RuntimeError(f"shape mismatch {a0}")
        max_old=max(max_old,float(np.max(np.abs(base-prior))))
        base_diag=output/"path_diagnostics"/dataset/signal/"A0_T3.npz"
        with np.load(base_diag,allow_pickle=False) as d0:
            ref0=d0["reference_window_ids"]; path0=d0["path_masks"]
        for name in ("A2_fixed_l000","A3_select_l000","A3_joint_l000"):
            max_zero=max(max_zero,float(np.max(np.abs(base-score(a0.with_name(name+".npz"))))))
            with np.load(base_diag.with_name(name+".npz"),allow_pickle=False) as dz:
                max_ref=max(max_ref,float(np.max(np.abs(ref0-dz["reference_window_ids"]))))
                max_path=max(max_path,float(np.max(np.abs(path0.astype(float)-dz["path_masks"].astype(float)))))
        for metric in (output/"metrics"/dataset/signal).glob("*.json"):
            doc=load_json(metric); meta=doc["backend_metadata"]
            pools.setdefault((dataset,signal),set()).add(meta["candidate_pool_digest"])
            if not meta.get("visual_audit_reference_ids_equal") or float(meta.get("visual_audit_max_abs_score_error",1))>1e-6:
                raise RuntimeError(f"visual audit failed: {metric}")
            scales=np.asarray(list(meta["numeric_component_scales"].values()),dtype=float)
            if not np.isfinite(scales).all() or np.any(scales<=0): raise RuntimeError(f"bad numeric scale {metric}")
        checked+=1
    if checked!=expected_signals or any(len(value)!=1 for value in pools.values()): raise RuntimeError("signal/pool coverage failed")
    baseline_tolerance=1e-10
    zero_weight_tolerance=1e-6
    if max_old>baseline_tolerance or max_zero>zero_weight_tolerance or max_ref or max_path:
        raise RuntimeError(f"exact recovery failed old={max_old} zero={max_zero} ref={max_ref} path={max_path}")
    payload={"status":"passed","signals":checked,"variants":len(config["variants_order"]),"metric_files":len(metrics),"point_score_files":len(scores),"diagnostic_files":len(diagnostics),"a0_vs_immutable_t3_max_abs_error":max_old,"a0_tolerance":baseline_tolerance,"zero_weight_max_abs_error":max_zero,"zero_weight_tolerance":zero_weight_tolerance,"zero_weight_reference_difference":max_ref,"zero_weight_path_difference":max_path,"all_candidate_pools_shared":True,"all_scores_and_scales_finite":True,"no_trainable_model":True}
    target=output/"validation.json"
    if not publish_json_no_replace(target,payload) and load_json(target)!=payload: raise FileExistsError(target)
    print(json.dumps(payload,indent=2,sort_keys=True)); return 0


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=HERE/"config.json"); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--previous-root",type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
