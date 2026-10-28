#!/usr/bin/env python3
"""Summarize temporal ViT4TS results with strict contextual F1-max."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_DATA_ROOT = Path("/home/pengxf/jim/VLM4TS/data")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/runs/temporal_matching_20261028"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiments.temporal_matching.runner import (
    load_json,
    load_signal,
    overlap,
    publish_bytes_no_replace,
    signal_path,
)


def publish_csv(path: Path, frame: pd.DataFrame) -> bool:
    return publish_bytes_no_replace(path, frame.to_csv(index=False).encode("utf-8"))


def load_documents(root: Path) -> list[dict[str, Any]]:
    documents = []
    for path in sorted((root / "metrics").glob("*/*/*.json")):
        document = load_json(path)
        if document.get("status") == "success":
            document["_path"] = str(path)
            documents.append(document)
    return documents


def flatten(documents: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for document in documents:
        timing = {f"timing_{key}": value for key, value in document.get("timing", {}).items() if isinstance(value, (int, float))}
        for metric in document["alpha_metrics"]:
            rows.append({
                "dataset": document["dataset"], "signal": document["signal"], "model": document["variant"],
                "alpha": float(metric["alpha"]), "precision": float(metric["precision"]), "recall": float(metric["recall"]), "f1": float(metric["f1"]),
                "TP": int(metric["TP"]), "FP": int(metric["FP"]), "FN": int(metric["FN"]), "threshold": float(metric["threshold"]),
                "n_points": int(document["n_points"]), "peak_cuda_memory_bytes": document.get("peak_cuda_memory_bytes"), **timing,
            })
    return pd.DataFrame(rows)


def aggregate(results: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = results.groupby(["dataset", "model", "alpha"], as_index=False).agg(
        mean_precision=("precision", "mean"), mean_recall=("recall", "mean"), mean_f1=("f1", "mean"),
        TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"), n_signals=("signal", "nunique")
    )
    grouped["expected_signals"] = grouped["dataset"].map(config["expected_signal_counts"]).astype(int)
    grouped["complete"] = grouped["n_signals"] == grouped["expected_signals"]
    alpha_order = {float(alpha): index for index, alpha in enumerate(config["evaluation"]["alphas"])}
    eligible = grouped[grouped["complete"]].copy()
    eligible["_order"] = eligible["alpha"].map(alpha_order)
    eligible = eligible.sort_values(["dataset", "model", "mean_f1", "_order"], ascending=[True, True, False, True], kind="stable")
    best = eligible.drop_duplicates(["dataset", "model"]).drop(columns="_order").rename(columns={"alpha": "best_alpha", "mean_f1": "f1_max"})
    return grouped, best


def validate_complete(dataset_alpha: pd.DataFrame, config: Mapping[str, Any]) -> list[str]:
    failures = []
    for dataset in config["datasets"]:
        for model in config["variants_order"]:
            part = dataset_alpha[(dataset_alpha.dataset == dataset) & (dataset_alpha.model == model)]
            if len(part) != len(config["evaluation"]["alphas"]) or not part["complete"].all():
                found = int(part.n_signals.max()) if len(part) else 0
                failures.append(f"{dataset}/{model}: {found}/{config['expected_signal_counts'][dataset]}")
    return failures


def completion_table(results: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for dataset in config["datasets"]:
        for model in config["variants_order"]:
            found = results[(results.dataset == dataset) & (results.model == model)].signal.nunique()
            expected = int(config["expected_signal_counts"][dataset])
            rows.append({"dataset": dataset, "model": model, "completed_signals": found, "expected_signals": expected, "complete": found == expected})
    expected_total = sum(int(config["expected_signal_counts"][dataset]) for dataset in config["datasets"])
    total_label = f"ALL_{expected_total}"
    for model in config["variants_order"]:
        found = len(results[results.model == model][["dataset", "signal"]].drop_duplicates())
        rows.append({"dataset": total_label, "model": model, "completed_signals": found, "expected_signals": expected_total, "complete": found == expected_total})
    return pd.DataFrame(rows)


def macro_scopes(config: Mapping[str, Any]) -> dict[str, list[str]]:
    configured = config.get("macro_scopes")
    if configured is not None:
        scopes = {str(name): list(datasets) for name, datasets in configured.items()}
    else:
        scopes = {}
        if config.get("real_datasets"):
            real = list(config["real_datasets"])
            scopes[f"real_{len(real)}"] = real
        datasets = list(config["datasets"])
        scopes[f"all_{len(datasets)}"] = datasets
    known = set(config["datasets"])
    for name, datasets in scopes.items():
        unknown = sorted(set(datasets) - known)
        if unknown:
            raise ValueError(f"macro scope {name} contains unknown datasets: {unknown}")
        if not datasets:
            raise ValueError(f"macro scope {name} is empty")
    return scopes


def macro_table(best: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for scope, datasets in macro_scopes(config).items():
        part = best[best.dataset.isin(datasets)]
        for model in config["variants_order"]:
            selected = part[part.model == model]
            rows.append({"scope": scope, "model": model, "macro_f1max": selected.f1_max.mean(), "n_datasets": selected.dataset.nunique(), "expected_datasets": len(datasets), "TP": selected.TP.sum(), "FP": selected.FP.sum(), "FN": selected.FN.sum(), "complete": selected.dataset.nunique() == len(datasets)})
    return pd.DataFrame(rows)


def comparisons(best: pd.DataFrame, macro: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    indexed = best.set_index(["dataset", "model"])
    rows = []
    for spec in config["comparisons"]:
        for dataset in config["datasets"]:
            keys = (dataset, spec["left"]), (dataset, spec["right"])
            if all(key in indexed.index for key in keys):
                left = float(indexed.loc[keys[0], "f1_max"]); right = float(indexed.loc[keys[1], "f1_max"])
                rows.append({"scope": "dataset", "dataset": dataset, "comparison": spec["name"], "left": spec["left"], "right": spec["right"], "left_f1max": left, "right_f1max": right, "delta_f1max": left - right})
        for scope in macro_scopes(config):
            left_row = macro[(macro.scope == scope) & (macro.model == spec["left"])]
            right_row = macro[(macro.scope == scope) & (macro.model == spec["right"])]
            if len(left_row) and len(right_row):
                left = float(left_row.iloc[0].macro_f1max); right = float(right_row.iloc[0].macro_f1max)
                rows.append({"scope": scope, "dataset": "", "comparison": spec["name"], "left": spec["left"], "right": spec["right"], "left_f1max": left, "right_f1max": right, "delta_f1max": left - right})
    return pd.DataFrame(rows)


def runtime_tables(results: pd.DataFrame, documents: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timing_columns = sorted(column for column in results if column.startswith("timing_"))
    columns = ["dataset", "signal", "model", "peak_cuda_memory_bytes", *timing_columns]
    signal = results[columns].drop_duplicates(["dataset", "signal", "model"])
    aggregations: dict[str, tuple[str, str]] = {"n_signals": ("signal", "nunique"), "peak_cuda_memory_bytes": ("peak_cuda_memory_bytes", "max")}
    for column in timing_columns:
        aggregations[f"mean_{column}"] = (column, "mean")
        aggregations[f"total_{column}"] = (column, "sum")
    summary = signal.groupby(["dataset", "model"], as_index=False).agg(**aggregations)
    summary.insert(2, "runtime_scope", "backend_reported_per_variant_with_shared_cache")
    curated_aliases = {
        "mean_variant_incremental_seconds": "mean_timing_inference_seconds",
        "mean_shared_visual_pool_seconds": "mean_timing_shared_visual_pool_seconds",
        "mean_shared_visual_patch_distance_seconds": "mean_timing_shared_visual_patch_distance_seconds",
        "mean_stable_shared_seconds": "mean_timing_stable_shared_seconds",
        "mean_score_recovery_seconds": "mean_timing_score_recovery_seconds",
    }
    for target, source in curated_aliases.items():
        summary[target] = summary[source] if source in summary else np.nan

    # Standardized incremental fields remain explicit even if a backend omits
    # them; raw timing keys stay available in the same table.
    aliases = {
        "matching_seconds": ["timing_matching_seconds", "timing_matching"],
        "incremental_matching_seconds": ["timing_incremental_matching_seconds"],
    }
    incremental = signal[["dataset", "signal", "model"]].copy()
    for target, candidates in aliases.items():
        present = next((column for column in candidates if column in signal), None)
        incremental[target] = signal[present] if present else np.nan
    timing_fields = list(aliases)
    incremental_summary = incremental.groupby(["dataset", "model"], as_index=False).agg(
        n_signals=("signal", "nunique"),
        **{f"mean_{field}": (field, "mean") for field in timing_fields},
        **{f"total_{field}": (field, "sum") for field in timing_fields},
    )
    return summary, incremental_summary, incremental


def candidate_pool_audit(documents: list[dict[str, Any]], config: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten the paired-reference pool identity reported by the backend.

    The digest is deliberately opaque: the validator only uses it to prove
    that B--E used the same eligible window IDs for a signal. A may omit all
    candidate fields because it retains the Stable median reference.
    """
    rows = []
    for document in documents:
        model = document["variant"]
        variant = config["variants"].get(model, {})
        if variant.get("family") == "A":
            continue
        metadata = document.get("backend_metadata", {})
        pool = metadata.get("candidate_pool", {}) if isinstance(metadata.get("candidate_pool", {}), Mapping) else {}
        def field(name: str) -> Any:
            return metadata.get(f"candidate_pool_{name}", pool.get(name))
        rows.append({
            "dataset": document["dataset"], "signal": document["signal"], "model": model,
            "family": variant.get("family"), "layer": variant.get("layer"),
            "base_sampling": variant.get("base_sampling"),
            "min_windows": field("min_windows"), "max_windows": field("max_windows"),
            "pool_digest": field("digest"),
            "max_configured_windows": int(config["model"]["candidate_pool"]["max_common_visual_windows"]),
            "overlap_exclusion_radius": int(config["model"]["candidate_pool"]["overlap_exclusion_window_index_radius"]),
        })
    return pd.DataFrame(rows)


def metric_at(document: Mapping[str, Any], alpha: float) -> Mapping[str, Any]:
    return next(metric for metric in document["alpha_metrics"] if np.isclose(metric["alpha"], alpha))


def event_changes(documents: list[dict[str, Any]], best: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_key = {(doc["dataset"], doc["signal"], doc["variant"]): doc for doc in documents}
    alphas = best.set_index(["dataset", "model"])["best_alpha"].to_dict()
    events, totals = [], []
    for spec in config["comparisons"]:
        for dataset in config["datasets"]:
            if (dataset, spec["left"]) not in alphas or (dataset, spec["right"]) not in alphas:
                continue
            recovered = lost = 0
            left_tp = left_fp = left_fn = right_tp = right_fp = right_fn = 0
            signals = sorted(signal for ds, signal, model in by_key if ds == dataset and model == spec["right"])
            for signal in signals:
                left = by_key.get((dataset, signal, spec["left"])); right = by_key.get((dataset, signal, spec["right"]))
                if left is None or right is None: continue
                lm = metric_at(left, float(alphas[(dataset, spec["left"])])); rm = metric_at(right, float(alphas[(dataset, spec["right"])]))
                left_tp += int(lm["TP"]); left_fp += int(lm["FP"]); left_fn += int(lm["FN"])
                right_tp += int(rm["TP"]); right_fp += int(rm["FP"]); right_fn += int(rm["FN"])
                for index, event in enumerate(right["ground_truth_intervals"]):
                    left_hit = any(overlap(event, item) for item in lm["detected_intervals"])
                    right_hit = any(overlap(event, item) for item in rm["detected_intervals"])
                    state = "recovered" if left_hit and not right_hit else "lost" if right_hit and not left_hit else "retained" if left_hit and right_hit else "missed_both"
                    recovered += state == "recovered"; lost += state == "lost"
                    events.append({"comparison": spec["name"], "dataset": dataset, "signal": signal, "event_index": index, "start": event[0], "end": event[1], "right_detected": right_hit, "left_detected": left_hit, "change": state, "right_alpha": rm["alpha"], "left_alpha": lm["alpha"]})
            totals.append({
                "comparison": spec["name"], "dataset": dataset,
                "left_TP": left_tp, "right_TP": right_tp, "TP_change": left_tp - right_tp,
                "left_FN": left_fn, "right_FN": right_fn, "FN_change": left_fn - right_fn,
                "left_FP": left_fp, "right_FP": right_fp, "FP_change": left_fp - right_fp,
                "FN_recovered": recovered, "TP_lost": lost, "net_event_recovery": recovered - lost,
            })
    return pd.DataFrame(events), pd.DataFrame(totals)


def publish_figure(path: Path, figure: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        figure.savefig(temporary, dpi=170, bbox_inches="tight")
        try: os.link(temporary, path)
        except FileExistsError: return False
        return True
    finally: temporary.unlink(missing_ok=True)


def summary_plots(best: pd.DataFrame, macro: pd.DataFrame, output_root: Path) -> None:
    if best.empty or macro.empty:
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["A_stable_final", "B_paired_2d_mid", "T1_independent", "T2_shift_consistent", "T3_constrained_dtw", "T4_bmid_t2_l025", "T5_bmid_t3_l025"]
    view = best[best.model.isin(models)].pivot(index="dataset", columns="model", values="f1_max")
    if not view.empty:
        figure, axis = plt.subplots(figsize=(14, 5.8))
        view.plot(kind="bar", ax=axis, width=.82)
        axis.set_ylabel("Unweighted contextual F1-max"); axis.set_xlabel("")
        axis.grid(axis="y", alpha=.25); axis.legend(frameon=False, fontsize=8, ncol=2)
        figure.tight_layout(); publish_figure(output_root / "plots/dataset_temporal_f1max.png", figure); plt.close(figure)
    all11 = macro[macro.scope == "all_11"].set_index("model")
    if not all11.empty:
        ordered = all11.reindex([name for name in models if name in all11.index])
        figure, axis = plt.subplots(figsize=(9, 4.8)); ordered.macro_f1max.plot(kind="bar", ax=axis, rot=35)
        axis.set_ylabel("All-11 macro F1-max"); axis.set_xlabel(""); axis.grid(axis="y", alpha=.25)
        figure.tight_layout(); publish_figure(output_root / "plots/macro_all11.png", figure); plt.close(figure)


def failure_plots(events: pd.DataFrame, changes: pd.DataFrame, data_root: Path, output_root: Path, config: Mapping[str, Any]) -> None:
    if events.empty: return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Keep each requested comparison represented.  Taking a global head tends
    # to fill every slot with the first dataset/comparison in sort order.
    failures = (
        events[events.change == "lost"]
        .sort_values(["comparison", "dataset", "signal", "event_index"], kind="stable")
        .groupby("comparison", sort=False, group_keys=False)
        .head(2)
    )
    if failures.empty:
        worst = changes[(changes.scope == "dataset") & (changes.delta_f1max < 0)].sort_values("delta_f1max").head(4)
        if worst.empty: return
        failures = events.merge(worst[["comparison", "dataset"]], on=["comparison", "dataset"]).head(8)
    specs = {item["name"]: item for item in config["comparisons"]}
    for _, row in failures.iterrows():
        spec = specs[row.comparison]; dataset = row.dataset; signal = row.signal
        source = load_signal(signal_path(data_root, dataset, signal), dataset)
        arrays = {}
        for side in ["right", "left"]:
            model = spec[side]
            with np.load(output_root / "point_scores" / dataset / signal / f"{model}.npz", allow_pickle=False) as stored:
                arrays[side] = stored["scores"].astype(float); timestamps = stored["timestamps"]
        inside = np.flatnonzero((timestamps >= row.start) & (timestamps <= row.end))
        if not len(inside): continue
        start = max(0, int(inside[0]) - 224); end = min(len(source), int(inside[-1]) + 225); x = np.arange(start, end)
        figure, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
        axes[0].plot(x, source.value.to_numpy()[start:end], color="black", lw=.8); axes[0].set_ylabel("value")
        for side in ["right", "left"]:
            values = arrays[side]; low, high = np.quantile(values, [.01, .99]); normalized = (values-low)/max(high-low, np.finfo(float).eps)
            axes[1].plot(x, normalized[start:end], label=f"{side}: {spec[side]}", lw=1)
        for axis in axes: axis.axvspan(inside[0], inside[-1], color="tab:red", alpha=.15); axis.grid(alpha=.2)
        axes[1].legend(frameon=False); axes[1].set_ylabel("normalized score"); axes[1].set_xlabel("sample index")
        axes[0].set_title(f"{row.comparison} failure: {dataset}/{signal} event {row.event_index}")
        figure.tight_layout(); publish_figure(output_root / "plots/failures" / f"{row.comparison}_{dataset}_{signal}_{int(row.event_index):02d}.png", figure); plt.close(figure)


def path_diagnostics(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate label-free T2/T3 correspondence diagnostics."""
    rows = []
    for path in sorted((output_root / "path_diagnostics").glob("*/*/*.npz")):
        dataset, signal, model = path.parts[-3], path.parts[-2], path.stem
        with np.load(path, allow_pickle=False) as stored:
            if "shift" in stored.files:
                offsets = stored["shift"].astype(float).reshape(-1)
                costs = stored["column_costs"].astype(float).reshape(-1)
                path_costs = np.asarray([], dtype=float); kind = "shift"
            else:
                offsets = stored["path_offsets"].astype(float).reshape(-1)
                costs = stored["local_costs"].astype(float).reshape(-1)
                path_costs = stored["normalized_path_costs"].astype(float).reshape(-1); kind = "dtw"
        rows.append({"dataset": dataset, "signal": signal, "model": model, "path_kind": kind,
                     "n_offsets": len(offsets), "mean_offset": float(offsets.mean()),
                     "mean_abs_offset": float(np.abs(offsets).mean()),
                     "fraction_nonzero_offset": float(np.mean(offsets != 0)),
                     "fraction_band_edge": float(np.mean(np.abs(offsets) >= 2)),
                     "mean_local_cost": float(costs.mean()), "p95_local_cost": float(np.quantile(costs, .95)),
                     "mean_normalized_path_cost": float(path_costs.mean()) if len(path_costs) else np.nan})
    signal = pd.DataFrame(rows)
    if signal.empty: return signal, signal
    dataset = signal.groupby(["dataset", "model", "path_kind"], as_index=False).agg(
        signals=("signal", "nunique"), mean_abs_offset=("mean_abs_offset", "mean"),
        fraction_nonzero_offset=("fraction_nonzero_offset", "mean"), fraction_band_edge=("fraction_band_edge", "mean"),
        mean_local_cost=("mean_local_cost", "mean"), p95_local_cost=("p95_local_cost", "mean"),
        mean_normalized_path_cost=("mean_normalized_path_cost", "mean"))
    return signal, dataset


def path_plots(signal_paths: pd.DataFrame, output_root: Path) -> None:
    if signal_paths.empty: return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for dataset in ("realAdExchange", "MSL", "SMAP", "YAHOOA3", "YAHOOA4"):
        part = signal_paths[(signal_paths.dataset == dataset) & (signal_paths.model == "T3_constrained_dtw")]
        if part.empty: continue
        row = part.sort_values("mean_abs_offset", ascending=False).iloc[0]
        source = output_root / "path_diagnostics" / dataset / row.signal / "T3_constrained_dtw.npz"
        with np.load(source, allow_pickle=False) as stored:
            offsets = stored["path_offsets"].astype(float); costs = stored["local_costs"].astype(float)
        order = np.argsort(np.mean(np.abs(offsets), axis=1))[::-1][:12]
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        im0 = axes[0].imshow(offsets[order].T, aspect="auto", origin="lower", vmin=-2, vmax=2, cmap="coolwarm")
        im1 = axes[1].imshow(costs[order].T, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_ylabel("query column"); axes[0].set_title(f"{dataset}/{row.signal}: DTW offsets")
        axes[1].set_ylabel("query column"); axes[1].set_xlabel("selected query windows"); axes[1].set_title("local visual cost")
        figure.colorbar(im0, ax=axes[0], fraction=.025); figure.colorbar(im1, ax=axes[1], fraction=.025)
        figure.tight_layout(); publish_figure(output_root / "plots/paths" / f"{dataset}_{row.signal}_dtw.png", figure); plt.close(figure)


def report(best: pd.DataFrame, macro: pd.DataFrame, changes: pd.DataFrame, event_totals: pd.DataFrame, runtime: pd.DataFrame, incremental: pd.DataFrame, completion: pd.DataFrame, config: Mapping[str, Any]) -> str:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty: return "No complete results."
        header = "| " + " | ".join(frame.columns) + " |\n| " + " | ".join("---" for _ in frame.columns) + " |"
        rows = ["| " + " | ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
        return "\n".join([header, *rows])
    expected_total = sum(int(config["expected_signal_counts"][dataset]) for dataset in config["datasets"])
    all11 = changes[changes.scope == "all_11"]
    conclusion_rows = []
    for name in ("T2_minus_T1", "T3_minus_T1", "T4_l025_minus_Bmid", "T4_l050_minus_Bmid", "T5_l025_minus_Bmid", "T5_l050_minus_Bmid"):
        selected = all11[all11.comparison == name]
        if len(selected):
            delta = float(selected.iloc[0].delta_f1max)
            conclusion_rows.append(f"- `{name}`: {delta:+.6f} macro F1-max.")
    if not conclusion_rows:
        conclusion_rows = ["- Complete results are required before drawing the temporal matching comparison."]
    return "\n".join([
        "# Temporal Matching ViT4TS results", "", "## Protocol", "", config["evaluation"]["selection_protocol"], "",
        "All paired and temporal variants share the same candidate-window pool. B-mid and T1--T5 use the same frozen CLIP block-6 14x14 features. T1/T2/T3 share the same column-cost matrices. Lambda 0.25 and 0.5 remain separate prespecified variants.", "",
        "## Dataset F1-max and contextual counts", "", table(best[["dataset", "model", "best_alpha", "f1_max", "TP", "FP", "FN"]]), "",
        "## Macro averages", "", table(macro[["scope", "model", "macro_f1max", "TP", "FP", "FN", "complete"]]), "",
        "## Primary controlled conclusions", "", *conclusion_rows, "",
        "## Incremental comparisons", "", table(changes), "", "## FN recovery, TP loss, and FP change", "", table(event_totals), "",
        "## Runtime", "",
        "Runtime fields are backend reported. Shared rendering/CLIP/cache work and per-variant sampling, matching, and score recovery remain separate when the backend exposes them; shared costs must not be summed once per variant.", "",
        table(runtime), "", "## Incremental matching timing", "", table(incremental), "",
        "## Completion", "", f"Each complete model contains all {expected_total} configured signals; the rows below are checked before final reporting.", "", table(completion), ""
    ])


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config); documents = load_documents(args.output_root)
    if not documents: raise RuntimeError("No temporal matching metric artifacts found")
    results = flatten(documents); dataset_alpha, best = aggregate(results, config)
    failures = validate_complete(dataset_alpha, config)
    if failures and not args.allow_incomplete: raise RuntimeError("Incomplete groups:\n" + "\n".join(failures[:30]))
    completion = completion_table(results, config)
    macro = macro_table(best, config); changes = comparisons(best, macro, config)
    runtime, incremental, incremental_per_signal = runtime_tables(results, documents)
    candidates = candidate_pool_audit(documents, config)
    events, event_totals = event_changes(documents, best, config)
    path_signal, path_dataset = path_diagnostics(args.output_root)
    artifacts = {"results.csv": results, "completion.csv": completion, "dataset_f1_by_alpha.csv": dataset_alpha, "dataset_f1max.csv": best, "macro_f1max.csv": macro, "incremental_comparisons.csv": changes, "event_changes.csv": events, "event_change_summary.csv": event_totals, "runtime.csv": runtime, "incremental_timing.csv": incremental, "incremental_timing_per_signal.csv": incremental_per_signal, "candidate_pool_audit.csv": candidates, "path_diagnostics_per_signal.csv": path_signal, "path_diagnostics_by_dataset.csv": path_dataset}
    for name, frame in artifacts.items():
        print(("wrote " if publish_csv(args.output_root / name, frame) else "kept ") + str(args.output_root / name))
    summary_plots(best, macro, args.output_root)
    path_plots(path_signal, args.output_root)
    failure_plots(events, changes, args.data_root, args.output_root, config)
    publish_bytes_no_replace(args.output_root / "experiment_report.md", report(best, macro, changes, event_totals, runtime, incremental, completion, config).encode("utf-8"))
    if failures: print("WARNING incomplete diagnostic summary:\n" + "\n".join(failures))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG); result.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT); result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT); result.add_argument("--allow-incomplete", action="store_true")
    return result


if __name__ == "__main__": raise SystemExit(run(parser().parse_args()))
