#!/usr/bin/env python3
"""Strict baseline-first and final validation for temporal matching runs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from experiments.temporal_matching.runner import load_json, publish_json_no_replace

BMID = "B_paired_2d_mid"
TARGET_MACRO = 0.657370
TOLERANCE = 1e-6
DIAGNOSTIC_FAMILIES = {"T2", "T3", "T4", "T5"}


def previous_score(roots: list[Path], dataset: str, signal: str) -> Path:
    candidates = [root / "point_scores" / dataset / signal / f"{name}.npz" for root in roots
                  for name in (BMID, "B_paired_visual_mid")]
    found = next((path for path in candidates if path.is_file()), None)
    if found is None: raise FileNotFoundError(f"missing previous B-mid score for {dataset}/{signal}")
    return found


def validate_bmid(config: dict, output: Path, previous_roots: list[Path], target: float) -> dict:
    if not previous_roots: raise ValueError("B-mid reproduction requires --previous-root")
    expected = sum(int(config["expected_signal_counts"][d]) for d in config["datasets"])
    paths = sorted((output / "metrics").glob(f"*/*/{BMID}.json"))
    if len(paths) != expected: raise RuntimeError(f"expected {expected} B-mid metrics, found {len(paths)}")
    maximum = 0.0; rows = []
    for path in paths:
        doc = load_json(path); dataset, signal = doc["dataset"], doc["signal"]
        score_path = output / "point_scores" / dataset / signal / f"{BMID}.npz"
        with np.load(score_path, allow_pickle=False) as now, np.load(previous_score(previous_roots, dataset, signal), allow_pickle=False) as old:
            a, b = now["scores"].astype(float), old["scores"].astype(float)
        if len(a) != len(b) or not np.isfinite(a).all(): raise RuntimeError(f"invalid B-mid score {score_path}")
        maximum = max(maximum, float(np.max(np.abs(a-b))))
        for metric in doc["alpha_metrics"]:
            rows.append({"dataset": dataset, "signal": signal, "alpha": float(metric["alpha"]), "f1": float(metric["f1"])})
    frame = pd.DataFrame(rows)
    means = frame.groupby(["dataset", "alpha"], as_index=False).f1.mean()
    chosen = means.sort_values(["dataset", "f1", "alpha"], ascending=[True, False, False], kind="stable").drop_duplicates("dataset")
    macro = float(chosen.f1.mean())
    if maximum > TOLERANCE: raise RuntimeError(f"B-mid score reproduction failed: max difference {maximum}")
    if abs(macro-target) > TOLERANCE: raise RuntimeError(f"B-mid macro {macro:.12f} differs from {target:.12f}")
    return {"signals": expected, "maximum_absolute_score_difference": maximum, "macro_f1max_all_11": macro,
            "expected_macro_f1max": target, "tolerance": TOLERANCE, "passed": True}


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config); roots = list(args.previous_root)
    baseline = validate_bmid(config, args.output_root, roots, args.expected_bmid_macro)
    if args.baseline_only:
        payload = {"status": "passed", "mode": "baseline_only", "B_mid_reproduction": baseline}
        target = args.output_root / "validation_bmid.json"
        if not publish_json_no_replace(target, payload) and load_json(target) != payload: raise FileExistsError(target)
        print(json.dumps(payload, indent=2, sort_keys=True)); return 0

    expected_signals = sum(int(config["expected_signal_counts"][d]) for d in config["datasets"])
    expected = expected_signals * len(config["variants_order"])
    metric_paths = sorted((args.output_root / "metrics").glob("*/*/*.json")); score_paths = sorted((args.output_root / "point_scores").glob("*/*/*.npz"))
    if len(metric_paths) != expected or len(score_paths) != expected:
        raise RuntimeError(f"expected {expected} metrics/scores, found {len(metric_paths)}/{len(score_paths)}")
    alpha_set = {float(v) for v in config["evaluation"]["alphas"]}; keys=set(); pool={}; diag_expected=0
    for path in metric_paths:
        doc=load_json(path); key=(doc["dataset"],doc["signal"],doc["variant"])
        if key in keys or doc.get("status") != "success": raise RuntimeError(f"duplicate/failed metric {path}")
        keys.add(key)
        if {float(m["alpha"]) for m in doc["alpha_metrics"]} != alpha_set: raise RuntimeError(f"alpha mismatch {path}")
        numeric=np.asarray([m[f] for m in doc["alpha_metrics"] for f in ("threshold","precision","recall","f1")],float)
        if not np.isfinite(numeric).all(): raise RuntimeError(f"non-finite metric {path}")
        score=args.output_root/"point_scores"/key[0]/key[1]/f"{key[2]}.npz"
        with np.load(score,allow_pickle=False) as stored:
            values=stored["scores"].astype(float); times=stored["timestamps"]
            metadata=json.loads(str(stored["metadata_json"][0]))
        if len(values)!=int(doc["n_points"]) or len(times)!=len(values) or not np.isfinite(values).all(): raise RuntimeError(f"bad score {score}")
        family=str(config["variants"][key[2]]["family"]).upper()
        if family != "A":
            digest=str(metadata.get("candidate_pool_digest","")); pool.setdefault((key[0],key[1]),set()).add(digest)
            if not digest: raise RuntimeError(f"missing pool digest {score}")
        if family in DIAGNOSTIC_FAMILIES:
            diag_expected += 1; diag=args.output_root/"path_diagnostics"/key[0]/key[1]/f"{key[2]}.npz"
            if not diag.is_file(): raise FileNotFoundError(diag)
            with np.load(diag,allow_pickle=False) as stored:
                if not stored.files or any(not np.isfinite(stored[name]).all() for name in stored.files): raise RuntimeError(f"invalid diagnostic {diag}")
                required={"reference_window_ids","shift","column_costs"} if family in {"T2","T4"} else {"reference_window_ids","path_offsets","reference_columns","local_costs","normalized_path_costs"}
                if not required.issubset(stored.files): raise RuntimeError(f"missing diagnostic keys {diag}")
    if len(pool)!=expected_signals or any(len(v)!=1 for v in pool.values()): raise RuntimeError("paired variants do not share one pool digest per signal")
    diag_paths=list((args.output_root/"path_diagnostics").glob("*/*/*.npz"))
    if len(diag_paths)!=diag_expected: raise RuntimeError(f"expected {diag_expected} diagnostics, found {len(diag_paths)}")
    results=pd.read_csv(args.output_root/"results.csv")
    if len(results)!=expected*len(alpha_set) or not np.isfinite(results[["alpha","f1","precision","recall","threshold"]].to_numpy()).all(): raise RuntimeError("results.csv failed")
    completion=pd.read_csv(args.output_root/"completion.csv")
    if not completion.complete.all(): raise RuntimeError("incomplete groups")
    payload={"status":"passed","mode":"full","metric_files":len(metric_paths),"point_score_files":len(score_paths),"diagnostic_files":len(diag_paths),
             "result_rows":len(results),"signals":expected_signals,"variants":len(config["variants_order"]),"all_scores_finite":True,
             "all_candidate_pools_shared":True,"B_mid_reproduction":baseline}
    target=args.output_root/"validation.json"
    if not publish_json_no_replace(target,payload) and load_json(target)!=payload: raise FileExistsError(target)
    print(json.dumps(payload,indent=2,sort_keys=True)); return 0


def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,default=HERE/"config.json"); p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--previous-root",type=Path,action="append",default=[]); p.add_argument("--baseline-only",action="store_true")
    p.add_argument("--expected-bmid-macro",type=float,default=TARGET_MACRO); return p
if __name__=="__main__": raise SystemExit(run(parser().parse_args()))
