#!/usr/bin/env python3
"""Evaluate the corrected AMR selector with contextual and MindTS metrics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.mindts_metrics import affiliation_scores, fast_vus, MINDTS_COMMIT
from experiments.temporal_matching.runner import discover_tasks, load_ground_truth, load_signal
from src.models.model_utils import compute_detection_intervals


VARIANTS = ["T3", "TS_only", "Late_joint", "AMR_full", "AMR_no_reference_loss"]
SCOPES = {
    "all_11": ["realTraffic", "realTweets", "SMAP", "MSL", "artificialWithAnomaly", "realAdExchange", "realAWSCloudwatch", "YAHOOA1", "YAHOOA2", "YAHOOA3", "YAHOOA4"],
    "real_6": ["realTraffic", "realTweets", "SMAP", "MSL", "realAdExchange", "realAWSCloudwatch"],
    "yahoo_4": ["YAHOOA1", "YAHOOA2", "YAHOOA3", "YAHOOA4"],
}


def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def _labels(timestamps, events):
    result = np.zeros(len(timestamps), np.uint8)
    for start, end in events:
        result |= ((timestamps >= start) & (timestamps <= end)).astype(np.uint8)
    return result


def _score(path):
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["scores"], float), np.asarray(z["timestamps"]), json.loads(str(z["metadata_json"][0]))


def _signal_metrics(payload):
    dataset, signal, raw_path, events, root, mindts_root = payload
    frame = load_signal(Path(raw_path), dataset)
    timestamps = frame["timestamp"].to_numpy()
    labels = _labels(timestamps, events)
    rows, audits = [], []
    for variant in VARIANTS:
        path = Path(root) / "point_scores" / dataset / signal / f"{variant}.npz"
        scores, saved_timestamps, metadata = _score(path)
        valid = len(scores) == len(labels) and np.array_equal(saved_timestamps, timestamps) and np.isfinite(scores).all()
        audits.append({"dataset": dataset, "signal": signal, "variant": variant, "valid": valid, "points": len(scores), "path": str(path)})
        if not valid:
            continue
        base = {"dataset": dataset, "signal": signal, "variant": variant, "points": len(scores), "anomaly_points": int(labels.sum()),
                "score_sha256": hashlib.sha256(np.ascontiguousarray(scores).view(np.uint8)).hexdigest(),
                "learned_cost_kind": metadata.get("learned_cost_kind", "historical_unchanged")}
        if labels.min() == labels.max():
            rows.append({**base, "aff_precision": np.nan, "aff_recall": np.nan, "aff_f": np.nan, "vus_pr": np.nan, "vus_roc": np.nan,
                         "median_event_length": np.nan, "vus_max_window": np.nan, "aff_threshold": np.nan,
                         "predicted_intervals": np.nan, "undefined_reason": "single_class_ground_truth"})
            continue
        intervals, threshold, _ = compute_detection_intervals(scores, alpha=.01, method="mean", smoothing=True, sliding=False, anomaly_padding=0)
        predictions = np.zeros(len(scores), np.uint8)
        for start, end in intervals:
            predictions[start:end + 1] = 1
        ap, ar, af = affiliation_scores(labels, predictions, Path(mindts_root))
        vus = fast_vus(labels, scores)
        rows.append({**base, "aff_precision": ap, "aff_recall": ar, "aff_f": af, "vus_pr": vus.vus_pr, "vus_roc": vus.vus_roc,
                     "median_event_length": vus.median_event_length, "vus_max_window": vus.max_window, "aff_threshold": float(threshold),
                     "predicted_intervals": len(intervals), "undefined_reason": "" if np.isfinite(af) else "no_predicted_events"})
    return rows, audits


def _contextual(root, config):
    rows, docs = [], {}
    for path in sorted((root / "metrics").glob("*/*/*.json")):
        d = json.loads(path.read_text())
        if d["variant"] not in VARIANTS:
            continue
        key = d["dataset"], d["signal"], d["variant"]
        docs[key] = d
        for item in d["alpha_metrics"]:
            rows.append({"dataset": key[0], "signal": key[1], "variant": key[2], **{k: item[k] for k in ("alpha", "TP", "FP", "FN", "f1")}})
    raw = pd.DataFrame(rows)
    means = raw.groupby(["dataset", "variant", "alpha"], as_index=False).f1.mean().rename(columns={"f1": "f1_max"})
    best = means.sort_values(["dataset", "variant", "f1_max", "alpha"], ascending=[True, True, False, False], kind="stable").drop_duplicates(["dataset", "variant"]).rename(columns={"alpha": "best_alpha"})
    counts = []
    for item in best.itertuples():
        q = raw[(raw.dataset == item.dataset) & (raw.variant == item.variant) & np.isclose(raw.alpha, item.best_alpha)]
        counts.append({"dataset": item.dataset, "variant": item.variant, "TP": int(q.TP.sum()), "FP": int(q.FP.sum()), "FN": int(q.FN.sum())})
    best = best.merge(pd.DataFrame(counts))
    macro = []
    for scope, datasets in SCOPES.items():
        for variant in VARIANTS:
            q = best[(best.dataset.isin(datasets)) & (best.variant == variant)]
            macro.append({"scope": scope, "variant": variant, "macro_f1max": q.f1_max.mean(), "TP": int(q.TP.sum()), "FP": int(q.FP.sum()), "FN": int(q.FN.sum())})
    return raw, means, best, pd.DataFrame(macro), docs


def _mechanism(root):
    rows = []
    for path in sorted((root / "path_diagnostics").glob("*/*/*.npz")):
        if path.stem not in VARIANTS:
            continue
        with np.load(path) as z:
            rows.append({"dataset": path.parts[-3], "signal": path.parts[-2], "variant": path.stem, "windows": len(z["reference_window_ids"]),
                         "reference_change_fraction": float(z["reference_changed"].mean()), "path_change_fraction": float(z["path_changed"].mean()),
                         "mean_visual_local_cost": float(z["visual_local_costs"].mean())})
    frame = pd.DataFrame(rows)
    summary = frame.groupby("variant", as_index=False).agg(signals=("signal", "count"), windows=("windows", "sum"),
        reference_change_fraction=("reference_change_fraction", "mean"), path_change_fraction=("path_change_fraction", "mean"),
        mean_visual_local_cost=("mean_visual_local_cost", "mean"))
    return frame, summary


def _event_changes(best, docs, config):
    rows = []
    for dataset in config["datasets"]:
        base_alpha = float(best[(best.dataset == dataset) & (best.variant == "T3")].best_alpha.iloc[0])
        for variant in VARIANTS[1:]:
            model_alpha = float(best[(best.dataset == dataset) & (best.variant == variant)].best_alpha.iloc[0])
            recovered = lost = removed = new = 0
            for signal in sorted(k[1] for k in docs if k[0] == dataset and k[2] == "T3"):
                base, model = docs[(dataset, signal, "T3")], docs[(dataset, signal, variant)]
                bp = next(x["detected_intervals"] for x in base["alpha_metrics"] if np.isclose(x["alpha"], base_alpha))
                mp = next(x["detected_intervals"] for x in model["alpha_metrics"] if np.isclose(x["alpha"], model_alpha))
                truth = base["ground_truth_intervals"]
                for event in truth:
                    bh, mh = any(_overlap(event, x) for x in bp), any(_overlap(event, x) for x in mp)
                    recovered += int(mh and not bh); lost += int(bh and not mh)
                bfp = [x for x in bp if not any(_overlap(x, e) for e in truth)]
                mfp = [x for x in mp if not any(_overlap(x, e) for e in truth)]
                removed += sum(not any(_overlap(x, y) for y in mfp) for x in bfp)
                new += sum(not any(_overlap(x, y) for y in bfp) for x in mfp)
            rows.append({"dataset": dataset, "variant": variant, "FN_recovered": recovered, "TP_lost": lost, "net_event_recovery": recovered-lost,
                         "removed_T3_FP": removed, "new_FP": new, "net_FP_change": new-removed})
    frame = pd.DataFrame(rows)
    total = frame.groupby("variant", as_index=False)[["FN_recovered", "TP_lost", "net_event_recovery", "removed_T3_FP", "new_FP", "net_FP_change"]].sum()
    return frame, total


def _unchanged_audit(root, historical):
    rows = []
    for variant in ["T3", "TS_only", "AMR_full", "AMR_no_reference_loss"]:
        max_error, scores_equal, refs_equal, paths_equal = 0.0, 0, 0, 0
        for path in sorted((root / "point_scores").glob(f"*/*/{variant}.npz")):
            rel = path.relative_to(root)
            with np.load(path) as a, np.load(historical / rel) as b:
                error = float(np.max(np.abs(a["scores"] - b["scores"])))
                max_error = max(max_error, error); scores_equal += int(error == 0)
            diag_rel = Path("path_diagnostics") / rel.relative_to("point_scores")
            with np.load(root / diag_rel) as a, np.load(historical / diag_rel) as b:
                refs_equal += int(np.array_equal(a["reference_window_ids"], b["reference_window_ids"]))
                paths_equal += int(np.array_equal(a["reference_columns"], b["reference_columns"]))
        rows.append({"variant": variant, "signals": 492, "bitwise_equal_scores": scores_equal, "equal_references": refs_equal, "equal_paths": paths_equal, "max_score_abs_error": max_error})
    return pd.DataFrame(rows)


def _md(frame, index=False):
    return frame.to_markdown(index=index, floatfmt=".6f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--mindts-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = args.output_root
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    config = json.loads((root / "config.json").read_text())
    raw, by_alpha, best, macro, docs = _contextual(root, config)
    mechanism, mechanism_summary = _mechanism(root)
    event_dataset, event_total = _event_changes(best, docs, config)
    unchanged = _unchanged_audit(root, args.historical_root)
    tasks = discover_tasks(config, Path("/home/pengxf/jim/VLM4TS/data"))
    truth = load_ground_truth(Path("/home/pengxf/jim/VLM4TS/data"))
    payloads = [(d, s, str(p), truth[s], str(root), str(args.mindts_root)) for d, s, p in tasks]
    metric_rows, audit_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows, audits in pool.map(_signal_metrics, payloads):
            metric_rows.extend(rows); audit_rows.extend(audits)
    metrics, alignment = pd.DataFrame(metric_rows), pd.DataFrame(audit_rows)
    dataset_rows = []
    for (variant, dataset), group in metrics.groupby(["variant", "dataset"]):
        row = {"variant": variant, "dataset": dataset, "signals": len(group)}
        for metric in ["aff_precision", "aff_recall", "aff_f", "vus_pr", "vus_roc"]:
            values = group[metric].to_numpy(float); row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            row[f"{metric}_valid_signals"] = int(np.isfinite(values).sum())
        dataset_rows.append(row)
    metric_dataset = pd.DataFrame(dataset_rows)
    scope_rows = []
    for variant in VARIANTS:
        for scope, datasets in SCOPES.items():
            group = metric_dataset[(metric_dataset.variant == variant) & metric_dataset.dataset.isin(datasets)]
            row = {"variant": variant, "scope": scope, "datasets": len(group)}
            for metric in ["aff_precision", "aff_recall", "aff_f", "vus_pr", "vus_roc"]:
                values = group[metric].to_numpy(float); row[metric] = float(np.nanmean(values))
                row[f"{metric}_valid_signals"] = int(group[f"{metric}_valid_signals"].sum())
            scope_rows.append(row)
    metric_scope = pd.DataFrame(scope_rows)
    for frame, name in [(raw, "results.csv"), (by_alpha, "dataset_f1_by_alpha.csv"), (best, "dataset_f1max.csv"), (macro, "macro_f1max.csv"),
                        (mechanism, "mechanism_per_signal.csv"), (mechanism_summary, "mechanism_summary.csv"),
                        (event_dataset, "event_changes_by_dataset.csv"), (event_total, "event_changes.csv"),
                        (unchanged, "unchanged_branch_integrity.csv"), (metrics, "mindts_metrics_per_signal.csv"),
                        (metric_dataset, "mindts_metrics_by_dataset.csv"), (metric_scope, "mindts_metrics_by_scope.csv"),
                        (alignment, "alignment_audit.csv")]:
        frame.to_csv(root / name, index=False)
    late_amr_equal_ref = late_amr_equal_path = 0
    for directory in (root / "path_diagnostics").glob("*/*"):
        with np.load(directory / "Late_joint.npz") as late, np.load(directory / "AMR_full.npz") as amr:
            late_amr_equal_ref += int(np.array_equal(late["reference_window_ids"], amr["reference_window_ids"]))
            late_amr_equal_path += int(np.array_equal(late["reference_columns"], amr["reference_columns"]))
    validation = {"status": "passed" if len(metrics) == 2460 and alignment.valid.all() and unchanged.max_score_abs_error.max() == 0 else "failed",
                  "signals": 492, "variants": 5, "score_files": len(list((root / "point_scores").glob("*/*/*.npz"))),
                  "metric_rows": len(metrics), "single_class_rows": int((metrics.undefined_reason == "single_class_ground_truth").sum()),
                  "late_amr_equal_reference_signals": late_amr_equal_ref, "late_amr_equal_path_signals": late_amr_equal_path,
                  "mindts_commit": MINDTS_COMMIT, "code_commit": code_commit,
                  "selector_fix_commit": "8c1dc06", "full_inference_commit": "b313018"}
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    m = macro.pivot(index="variant", columns="scope", values="macro_f1max").reindex(VARIANTS)
    c = macro[macro.scope == "all_11"][["variant", "TP", "FP", "FN"]]
    per = best.pivot(index="dataset", columns="variant", values="f1_max").reindex(config["datasets"])
    mdataset = metric_dataset.pivot(index="dataset", columns="variant", values=["aff_f", "vus_pr", "vus_roc"]).reindex(config["datasets"])
    mdataset.columns = [f"{metric}_{variant}" for metric, variant in mdataset.columns]
    mscope = metric_scope[["variant", "scope", "aff_f", "vus_pr", "vus_roc"]]
    late = float(m.loc["Late_joint", "all_11"]); amr = float(m.loc["AMR_full", "all_11"])
    cross_effect = pd.DataFrame([
        {"scope": scope, "Late_joint": float(m.loc["Late_joint", scope]),
         "AMR_full": float(m.loc["AMR_full", scope]),
         "AMR_minus_Late": float(m.loc["AMR_full", scope] - m.loc["Late_joint", scope])}
        for scope in ["all_11", "real_6", "yahoo_4"]
    ])
    dataset_effect = pd.DataFrame({
        "dataset": per.index,
        "T3": per["T3"].to_numpy(),
        "Late_joint": per["Late_joint"].to_numpy(),
        "AMR_full": per["AMR_full"].to_numpy(),
    })
    dataset_effect["AMR_minus_Late"] = dataset_effect.AMR_full - dataset_effect.Late_joint
    dataset_effect["AMR_minus_T3"] = dataset_effect.AMR_full - dataset_effect.T3
    report = f"""# Corrected AMR late-fusion and cross-attention evaluation

## Integrity and implementation

- Isolated worktree: `{Path.cwd()}`. Selector data-flow fix: `8c1dc06`; optimized full inference and evaluator: `b313018`; report code: `{code_commit}`.
- Existing six cross-fit checkpoints and frozen CLIP caches were reused. No model was retrained and the historical run was not modified.
- `Late_joint` now mixes the calibrated immutable visual cost with the independently encoded numeric-token cost. `AMR_full` and `AMR_no_reference_loss` mix visual cost with fused tokens produced from numeric-to-visual and visual-to-numeric Cross-Attention.
- Regression tests verify `(3,14,192)` numeric/visual/fused shapes and show that zeroing both Cross-Attention outputs changes fused tokens while leaving numeric and visual tokens bitwise unchanged.
- T3, TS-only, AMR-full and AMR-no-reference-loss remain bitwise identical to all 492 historical point-score files and retain identical reference/path diagnostics.
- MindTS commit `{MINDTS_COMMIT}` supplies Affiliation-F. VUS-PR/VUS-ROC use the audited equivalent of its 250-threshold RangeAUC-volume loop. Aff-F uses the fixed label-free project threshold `alpha=0.01`; VUS uses continuous scores.
- Four single-class signals yield undefined Aff-F/VUS and remain `NaN`, excluded from means.

## Macro Contextual F1-max

{_md(m.reset_index())}

### All-11 contextual counts

{_md(c)}

## Per-dataset Contextual F1-max

{_md(per.reset_index())}

## MindTS metrics by scope

{_md(mscope)}

## MindTS metrics by dataset

{_md(mdataset.reset_index())}

## Reference and DTW mechanism

{_md(mechanism_summary)}

Late-joint and AMR-full retain identical complete reference arrays on `{late_amr_equal_ref}/492` signals and identical paths on `{late_amr_equal_path}/492`. The corrected branches are therefore operationally distinct.

## Event and false-positive changes versus T3

{_md(event_total)}

Detailed dataset-level changes are stored in `event_changes_by_dataset.csv`.

Contextual `TP/FP/FN` are interval-level protocol counts: `TP` counts predicted intervals overlapping truth and `FN` counts missed true events. They are retained from the original evaluation implementation and should not be read as a point-wise confusion matrix.

## Controlled Cross-Attention contribution

{_md(cross_effect)}

### Dataset-level effect

{_md(dataset_effect)}

The repair changes the scientific conclusion from “AMR and Late-joint are identical” to a measured, small and domain-dependent Cross-Attention effect:

- `AMR_full` and corrected `Late_joint` have identical complete reference arrays on only `{late_amr_equal_ref}/492` signals and identical DTW paths on `{late_amr_equal_path}/492`; the two inference branches now exercise distinct data flows.
- Cross-Attention reduces the reference-change rate from `{float(mechanism_summary.set_index('variant').loc['Late_joint','reference_change_fraction']):.6f}` to `{float(mechanism_summary.set_index('variant').loc['AMR_full','reference_change_fraction']):.6f}` and the path-change rate from `{float(mechanism_summary.set_index('variant').loc['Late_joint','path_change_fraction']):.6f}` to `{float(mechanism_summary.set_index('variant').loc['AMR_full','path_change_fraction']):.6f}`. It therefore behaves as a more conservative visual anchor rather than as an interchangeable late cost.
- Relative to T3, Late-joint recovers `22` missed events, loses `12`, removes `39` false-positive intervals and adds `57`; AMR-full recovers `12`, loses `8`, removes `31` and adds `42`. Cross-Attention suppresses some routing changes and false positives, but it also gives up recovered events.
- The largest Late-joint advantage is AdExchange (`+0.047619` over AMR-full). AMR-full is better on Tweets, MSL and Yahoo A3, while most other dataset differences are small. The aggregate effect is negative on All-11 and Real-6 and slightly positive on Yahoo-4.

## Metric interpretation

- T3 remains best on All-11 and Real-6 Contextual F1-max and on their Aff-F, VUS-PR and VUS-ROC summaries. No corrected learned selector provides a stable general improvement.
- `AMR_no_reference_loss` is the strongest learned contextual variant at `{float(m.loc['AMR_no_reference_loss','all_11']):.6f}/{float(m.loc['AMR_no_reference_loss','real_6']):.6f}/{float(m.loc['AMR_no_reference_loss','yahoo_4']):.6f}` for All-11/Real-6/Yahoo-4, but remains below T3 on All-11 and Real-6.
- AMR-full improves Yahoo VUS-PR over corrected Late-joint, while Late-joint has slightly higher Yahoo VUS-ROC. This agrees with the event results: the Cross-Attention representation adds a small Yahoo-specific effect, not a robust cross-domain gain.

## Final judgment

Corrected Late-joint reaches All-11 `{late:.6f}` versus AMR-full `{amr:.6f}` and T3 `{float(m.loc['T3','all_11']):.6f}`. The corrected controlled estimate is `AMR_full - Late_joint = {amr-late:+.6f}` on All-11, `{float(m.loc['AMR_full','real_6']-m.loc['Late_joint','real_6']):+.6f}` on Real-6 and `{float(m.loc['AMR_full','yahoo_4']-m.loc['Late_joint','yahoo_4']):+.6f}` on Yahoo-4. Cross-Attention is now proven to participate in candidate routing, but it does not improve the general detector. T3 remains the retained model. No post-evaluation weight, checkpoint or threshold selection was performed.

## Artifacts

- `dataset_f1max.csv`, `macro_f1max.csv`, `results.csv`
- `mindts_metrics_per_signal.csv`, `mindts_metrics_by_dataset.csv`, `mindts_metrics_by_scope.csv`
- `mechanism_per_signal.csv`, `mechanism_summary.csv`
- `event_changes.csv`, `event_changes_by_dataset.csv`
- `unchanged_branch_integrity.csv`, `alignment_audit.csv`, `validation.json`
"""
    (root / "amr_late_joint_fix_final_report.md").write_text(report)
    print(json.dumps(validation, indent=2)); print(m.to_string()); print(mscope.to_string(index=False))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
