#!/usr/bin/env python3
"""Aggregate VFG-ViT4TS artifacts using dataset-level contextual F1-max."""

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
DEFAULT_CONFIG = HERE / "configs.json"
DEFAULT_DATA_ROOT = Path("/home/pengxf/jim/VLM4TS/data")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "runs" / "vfg_vit4ts_20261027"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vfg_vit4ts.runner import (  # noqa: E402
    intervals_overlap,
    load_json,
    publish_bytes_no_replace,
)


def publish_csv_no_replace(path: Path, frame: pd.DataFrame) -> bool:
    return publish_bytes_no_replace(path, frame.to_csv(index=False).encode("utf-8"))


def load_metric_documents(output_root: Path) -> list[dict[str, Any]]:
    documents = []
    for path in sorted((output_root / "metrics").glob("*/*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("status") == "success":
            document["_path"] = str(path)
            documents.append(document)
    return documents


def flatten_results(documents: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in documents:
        for metric in document["alpha_metrics"]:
            rows.append(
                {
                    "dataset": document["dataset"],
                    "signal": document["signal"],
                    "model": document["variant"],
                    "alpha": float(metric["alpha"]),
                    "precision": float(metric["precision"]),
                    "recall": float(metric["recall"]),
                    "f1": float(metric["f1"]),
                    "TP": int(metric["TP"]),
                    "FP": int(metric["FP"]),
                    "FN": int(metric["FN"]),
                    "threshold": float(metric["threshold"]),
                    "n_detected": int(metric["n_detected"]),
                    "n_ground_truth": int(metric["n_ground_truth"]),
                    "n_points": int(document["n_points"]),
                    "inference_seconds": document.get("inference_seconds"),
                    "peak_cuda_memory_bytes": document.get("peak_cuda_memory_bytes"),
                    "status": document["status"],
                }
            )
    return pd.DataFrame(rows)


def aggregate_dataset_level(
    results: pd.DataFrame, config: Mapping[str, Any], selected_variants: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mean signal F1 at each alpha, then max alpha for each dataset/model."""
    expected_counts = config["expected_signal_counts"]
    grouped = (
        results[results["model"].isin(selected_variants)]
        .groupby(["dataset", "model", "alpha"], as_index=False)
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            TP=("TP", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum"),
            n_signals=("signal", "nunique"),
            mean_inference_seconds=("inference_seconds", "mean"),
            total_inference_seconds=("inference_seconds", "sum"),
            peak_cuda_memory_bytes=("peak_cuda_memory_bytes", "max"),
        )
    )
    grouped["expected_signals"] = grouped["dataset"].map(expected_counts).astype(int)
    grouped["complete"] = grouped["n_signals"] == grouped["expected_signals"]

    alpha_order = {float(value): index for index, value in enumerate(config["evaluation"]["alphas"])}
    eligible = grouped[grouped["complete"]].copy()
    eligible["_alpha_order"] = eligible["alpha"].map(alpha_order)
    eligible = eligible.sort_values(
        ["dataset", "model", "mean_f1", "_alpha_order"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    best = eligible.drop_duplicates(["dataset", "model"], keep="first").drop(columns="_alpha_order")
    best = best.rename(columns={"alpha": "best_alpha", "mean_f1": "f1_max"})

    macro = (
        best.groupby("model", as_index=False)
        .agg(
            macro_f1max=("f1_max", "mean"),
            n_datasets=("dataset", "nunique"),
            total_TP=("TP", "sum"),
            total_FP=("FP", "sum"),
            total_FN=("FN", "sum"),
            mean_signal_inference_seconds=("mean_inference_seconds", "mean"),
            peak_cuda_memory_bytes=("peak_cuda_memory_bytes", "max"),
        )
        .sort_values("macro_f1max", ascending=False)
    )
    macro["complete"] = macro["n_datasets"] == len(config["datasets"])
    return grouped, best, macro


def validate_completion(
    dataset_alpha: pd.DataFrame,
    config: Mapping[str, Any],
    selected_variants: list[str],
) -> list[str]:
    failures = []
    for dataset in config["datasets"]:
        for variant in selected_variants:
            subset = dataset_alpha[
                (dataset_alpha["dataset"] == dataset)
                & (dataset_alpha["model"] == variant)
            ]
            expected_alphas = len(config["evaluation"]["alphas"])
            if len(subset) != expected_alphas or not subset["complete"].all():
                found = int(subset["n_signals"].max()) if len(subset) else 0
                expected = int(config["expected_signal_counts"][dataset])
                failures.append(f"{dataset}/{variant}: {found}/{expected} signals")
    return failures


def comparison_table(
    f1max: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    rows = []
    indexed = f1max.set_index(["dataset", "model"])
    for comparison_name, members in config["comparisons"].items():
        member_items = list(members.items())
        reference_label, reference_variant = member_items[0]
        for dataset in config["datasets"]:
            if (dataset, reference_variant) not in indexed.index:
                continue
            reference = float(indexed.loc[(dataset, reference_variant), "f1_max"])
            for label, variant in member_items:
                if (dataset, variant) not in indexed.index:
                    continue
                value = float(indexed.loc[(dataset, variant), "f1_max"])
                rows.append(
                    {
                        "comparison": comparison_name,
                        "dataset": dataset,
                        "member": label,
                        "model": variant,
                        "f1_max": value,
                        "reference_member": reference_label,
                        "delta_vs_reference": value - reference,
                    }
                )
    return pd.DataFrame(rows)


def runtime_table(results: pd.DataFrame) -> pd.DataFrame:
    per_signal = results.drop_duplicates(["dataset", "signal", "model"])
    return (
        per_signal.groupby(["dataset", "model"], as_index=False)
        .agg(
            n_signals=("signal", "nunique"),
            mean_inference_seconds=("inference_seconds", "mean"),
            median_inference_seconds=("inference_seconds", "median"),
            total_inference_seconds=("inference_seconds", "sum"),
            peak_cuda_memory_bytes=("peak_cuda_memory_bytes", "max"),
        )
        .sort_values(["dataset", "model"])
    )


def main_pairwise_table(
    f1max: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    members = config["comparisons"]["base"]
    required = [members["A"], members["C"], members["D"]]
    pivot = f1max[f1max["model"].isin(required)].pivot(
        index="dataset", columns="model", values="f1_max"
    )
    if any(model not in pivot for model in required):
        return pd.DataFrame()
    pivot = pivot.reindex(config["datasets"])
    return pd.DataFrame(
        {
            "dataset": pivot.index,
            "A_stable_f1max": pivot[members["A"]].to_numpy(),
            "C_score_fusion_f1max": pivot[members["C"]].to_numpy(),
            "D_frequency_guided_f1max": pivot[members["D"]].to_numpy(),
            "delta_D_minus_A": (pivot[members["D"]] - pivot[members["A"]]).to_numpy(),
            "delta_D_minus_C": (pivot[members["D"]] - pivot[members["C"]]).to_numpy(),
        }
    )


def publish_figure_no_replace(path: Path, figure: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.png")
    try:
        figure.savefig(temporary, format="png", dpi=170, bbox_inches="tight")
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def plot_comparisons(
    f1max: pd.DataFrame, config: Mapping[str, Any], output_root: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, members in config["comparisons"].items():
        present = [variant for variant in members.values() if variant in set(f1max["model"])]
        if len(present) < 2:
            continue
        pivot = f1max[f1max["model"].isin(present)].pivot(
            index="dataset", columns="model", values="f1_max"
        )
        pivot = pivot.reindex(config["datasets"])
        labels_by_variant = {variant: label for label, variant in members.items()}
        figure, axis = plt.subplots(figsize=(12, 4.8))
        x = np.arange(len(pivot.index))
        width = 0.82 / len(present)
        for index, variant in enumerate(present):
            axis.bar(
                x + (index - (len(present) - 1) / 2) * width,
                pivot[variant],
                width,
                label=labels_by_variant[variant],
            )
        axis.set_xticks(x, pivot.index, rotation=25, ha="right")
        axis.set_ylabel("Dataset-level Unweighted Contextual F1-max")
        axis.set_title(name.replace("_", " "))
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, ncol=min(3, len(present)))
        figure.tight_layout()
        publish_figure_no_replace(output_root / "plots" / f"{name}.png", figure)
        plt.close(figure)


def plot_main_diagnostics(
    f1max: pd.DataFrame,
    runtime: pd.DataFrame,
    config: Mapping[str, Any],
    output_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    members = config["comparisons"]["base"]
    models = [model for model in members.values() if model in set(f1max["model"])]
    labels = {model: label for label, model in members.items()}
    if len(models) < 2:
        return
    selected = f1max[f1max["model"].isin(models)].copy()
    datasets = config["datasets"]

    # Main figure 2: per-dataset change from stable A.
    pivot = selected.pivot(index="dataset", columns="model", values="f1_max").reindex(datasets)
    baseline = members["A"]
    figure, axis = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(datasets))
    compared = [model for model in models if model != baseline]
    width = 0.82 / max(1, len(compared))
    for index, model in enumerate(compared):
        axis.bar(
            x + (index - (len(compared) - 1) / 2) * width,
            pivot[model] - pivot[baseline],
            width,
            label=f"{labels[model]} − A",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, datasets, rotation=25, ha="right")
    axis.set_ylabel("Δ dataset F1-max")
    axis.set_title("A/B/C/D performance change relative to Stable ViT4TS")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    publish_figure_no_replace(output_root / "plots" / "base_delta_vs_A.png", figure)
    plt.close(figure)

    # Main figure 3: contextual event counts at each model's selected alpha.
    counts = selected.groupby("model")[["TP", "FP", "FN"]].sum().reindex(models)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(models))
    width = 0.24
    for offset, column in enumerate(["TP", "FP", "FN"]):
        axis.bar(x + (offset - 1) * width, counts[column], width, label=column)
    axis.set_xticks(x, [labels[model] for model in models], rotation=15, ha="right")
    axis.set_ylabel("Contextual interval count")
    axis.set_title("A/B/C/D TP, FP, and FN at selected dataset α")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    publish_figure_no_replace(output_root / "plots" / "base_contextual_counts.png", figure)
    plt.close(figure)

    # Main figure 4: mean signal inference time from saved artifacts.
    time_values = (
        runtime[runtime["model"].isin(models)]
        .groupby("model")["mean_inference_seconds"]
        .mean()
        .reindex(models)
    )
    figure, axis = plt.subplots(figsize=(8, 4.6))
    axis.bar(np.arange(len(models)), time_values, color="tab:blue", alpha=0.8)
    axis.set_xticks(np.arange(len(models)), [labels[model] for model in models], rotation=15, ha="right")
    axis.set_ylabel("Mean inference seconds per signal")
    axis.set_title("A/B/C/D inference time")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    publish_figure_no_replace(output_root / "plots" / "base_runtime.png", figure)
    plt.close(figure)


def find_alpha_metric(document: Mapping[str, Any], alpha: float) -> Mapping[str, Any]:
    for metric in document["alpha_metrics"]:
        if np.isclose(float(metric["alpha"]), alpha):
            return metric
    raise KeyError(f"alpha={alpha} missing from {document['_path']}")


def event_is_detected(event: list[Any], detected: list[list[Any]]) -> bool:
    return any(intervals_overlap(event, interval) for interval in detected)


def event_analysis(
    documents: list[dict[str, Any]],
    f1max: pd.DataFrame,
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config["event_analysis"]
    baseline = settings["baseline"]
    candidate = settings["candidate"]
    best = f1max.set_index(["dataset", "model"])["best_alpha"].to_dict()
    by_key = {(doc["dataset"], doc["signal"], doc["variant"]): doc for doc in documents}
    rows = []
    fp_rows = []
    annotations = settings.get("semantic_annotations", {})
    for dataset in config["datasets"]:
        if (dataset, baseline) not in best or (dataset, candidate) not in best:
            continue
        alpha_a = float(best[(dataset, baseline)])
        alpha_d = float(best[(dataset, candidate)])
        signal_names = sorted(
            signal for ds, signal, variant in by_key if ds == dataset and variant == baseline
        )
        for signal in signal_names:
            doc_a = by_key[(dataset, signal, baseline)]
            doc_d = by_key.get((dataset, signal, candidate))
            if doc_d is None:
                continue
            metric_a = find_alpha_metric(doc_a, alpha_a)
            metric_d = find_alpha_metric(doc_d, alpha_d)
            fp_rows.append(
                {
                    "dataset": dataset,
                    "signal": signal,
                    "baseline_alpha": alpha_a,
                    "candidate_alpha": alpha_d,
                    "baseline_FP": metric_a["FP"],
                    "candidate_FP": metric_d["FP"],
                    "delta_FP": metric_d["FP"] - metric_a["FP"],
                }
            )
            score_a_path = output_root / "point_scores" / dataset / signal / f"{baseline}.npz"
            score_d_path = output_root / "point_scores" / dataset / signal / f"{candidate}.npz"
            with np.load(score_a_path, allow_pickle=False) as stored:
                timestamps = stored["timestamps"]
                score_a = stored["scores"]
            with np.load(score_d_path, allow_pickle=False) as stored:
                score_d = stored["scores"]
            for event_index, event in enumerate(doc_a["ground_truth_intervals"]):
                inside = (timestamps >= event[0]) & (timestamps <= event[1])
                n_points = int(inside.sum())
                if n_points <= int(settings["short_max_points"]):
                    duration_type = "short"
                elif n_points >= int(settings["long_min_points"]):
                    duration_type = "long"
                else:
                    duration_type = "intermediate"
                key = f"{dataset}/{signal}/{event[0]}/{event[1]}"
                semantic_type = annotations.get(key, "unclassified")
                mean_a = float(np.mean(score_a[inside])) if inside.any() else np.nan
                mean_d = float(np.mean(score_d[inside])) if inside.any() else np.nan
                rank_a = float(np.mean(score_a <= mean_a)) if np.isfinite(mean_a) else np.nan
                rank_d = float(np.mean(score_d <= mean_d)) if np.isfinite(mean_d) else np.nan
                detected_a = event_is_detected(event, metric_a["detected_intervals"])
                detected_d = event_is_detected(event, metric_d["detected_intervals"])
                rows.append(
                    {
                        "dataset": dataset,
                        "signal": signal,
                        "event_index": event_index,
                        "start": event[0],
                        "end": event[1],
                        "n_points": n_points,
                        "duration_type": duration_type,
                        "semantic_type": semantic_type,
                        "baseline_alpha": alpha_a,
                        "candidate_alpha": alpha_d,
                        "baseline_detected": detected_a,
                        "candidate_detected": detected_d,
                        "recovered": (not detected_a) and detected_d,
                        "lost": detected_a and (not detected_d),
                        "baseline_event_score_mean": mean_a,
                        "candidate_event_score_mean": mean_d,
                        "baseline_event_score_percentile": rank_a,
                        "candidate_event_score_percentile": rank_d,
                        "delta_event_score_percentile": rank_d - rank_a,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(fp_rows)


def plot_representative_events(
    events: pd.DataFrame,
    data_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
) -> None:
    if events.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = config["event_analysis"]["baseline"]
    candidate = config["event_analysis"]["candidate"]
    candidates = events[events["recovered"] | events["lost"]].copy()
    if candidates.empty:
        candidates = events.copy()
    candidates["priority"] = (
        candidates["recovered"].astype(int) * 2
        + candidates["lost"].astype(int)
        + candidates["delta_event_score_percentile"].abs().fillna(0)
    )
    candidates = candidates.sort_values("priority", ascending=False).head(4)
    context_points = int(config["model"]["window_size"])
    for _, event in candidates.iterrows():
        dataset = str(event["dataset"])
        signal = str(event["signal"])
        source = pd.read_csv(data_root / dataset / f"{signal}.csv").sort_values(
            "timestamp", kind="stable"
        ).reset_index(drop=True)
        score_dir = output_root / "point_scores" / dataset / signal
        with np.load(score_dir / f"{baseline}.npz", allow_pickle=False) as stored:
            timestamps = stored["timestamps"]
            score_a = stored["scores"].astype(float)
        with np.load(score_dir / f"{candidate}.npz", allow_pickle=False) as stored:
            score_d = stored["scores"].astype(float)
        event_indices = np.flatnonzero(
            (timestamps >= event["start"]) & (timestamps <= event["end"])
        )
        if not len(event_indices):
            continue
        left = max(0, int(event_indices[0]) - context_points)
        right = min(len(source), int(event_indices[-1]) + context_points + 1)
        x = np.arange(left, right)

        def robust_unit(values: np.ndarray) -> np.ndarray:
            low, high = np.quantile(values, [0.01, 0.99])
            return (values - low) / max(float(high - low), np.finfo(float).eps)

        figure, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True)
        axes[0].plot(x, source["value"].to_numpy()[left:right], color="black", linewidth=0.8)
        axes[0].set_ylabel("value")
        axes[0].set_title(
            f"{dataset}/{signal} event {int(event['event_index'])}: "
            f"A={bool(event['baseline_detected'])}, D={bool(event['candidate_detected'])}"
        )
        axes[1].plot(x, robust_unit(score_a)[left:right], label="A stable", linewidth=1)
        axes[1].plot(x, robust_unit(score_d)[left:right], label="D guided", linewidth=1)
        axes[1].set_ylabel("robust normalized score")
        axes[1].set_xlabel("sample index")
        axes[1].legend(frameon=False)
        for axis in axes:
            axis.axvspan(event_indices[0], event_indices[-1], color="tab:red", alpha=0.16)
            axis.grid(alpha=0.2)
        figure.tight_layout()
        filename = f"event_{dataset}_{signal}_{int(event['event_index']):02d}.png"
        publish_figure_no_replace(output_root / "plots" / "representative_events" / filename, figure)
        plt.close(figure)


def build_report(
    f1max: pd.DataFrame,
    macro: pd.DataFrame,
    comparisons: pd.DataFrame,
    main_pairwise: pd.DataFrame,
    runtime: pd.DataFrame,
    events: pd.DataFrame,
    false_positives: pd.DataFrame,
    config: Mapping[str, Any],
) -> str:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No complete results."
        text = frame.copy()
        for column in text.select_dtypes(include=["float"]).columns:
            text[column] = text[column].map(lambda value: f"{value:.6f}")
        headers = [str(column) for column in text.columns]
        rows = [[str(value) for value in row] for row in text.itertuples(index=False, name=None)]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# Visual-Frequency Guided ViT4TS experiment report",
        "",
        "## Evaluation protocol",
        "",
        config["evaluation"]["selection_protocol"],
        "",
        "All methods use α ∈ {0.1, 0.01, 0.001}. Counts are contextual interval TP/FP/FN. Harmonic stabilization is shared by every method and is not counted as a model contribution.",
        "",
        "Local STFT uses 64-point Hann frames at hop 16, centered on the 14 visual columns, with 24-point reflect padding on each side. It retains 32 non-DC log1p magnitude bins and separate shape/energy information.",
        "",
        "## Dataset results",
        "",
    ]
    display_columns = ["dataset", "model", "best_alpha", "f1_max", "TP", "FP", "FN"]
    lines.append(markdown_table(f1max[display_columns]))
    lines.extend(["", "## Macro average", ""])
    macro_columns = ["model", "macro_f1max", "total_TP", "total_FP", "total_FN", "complete"]
    lines.append(markdown_table(macro[macro_columns]))
    lines.extend(["", "## Runtime and memory", ""])
    runtime_columns = [
        "dataset",
        "model",
        "mean_inference_seconds",
        "total_inference_seconds",
        "peak_cuda_memory_bytes",
    ]
    lines.append(markdown_table(runtime[runtime_columns]))

    indexed_macro = macro.set_index("model") if len(macro) else pd.DataFrame()
    baseline = config["event_analysis"]["baseline"]
    candidate = config["event_analysis"]["candidate"]
    fusion = config["comparisons"]["base"]["C"]
    lines.extend(["", "## Main questions", ""])
    for left, right, wording in [
        (candidate, baseline, "D versus A"),
        (candidate, fusion, "D versus C"),
    ]:
        if left in indexed_macro.index and right in indexed_macro.index:
            delta = float(indexed_macro.loc[left, "macro_f1max"] - indexed_macro.loc[right, "macro_f1max"])
            lines.append(f"- {wording}: macro F1-max difference {delta:+.6f}.")
        else:
            lines.append(f"- {wording}: incomplete results.")
    if len(main_pairwise):
        lines.extend(["", "Per-dataset D improvements over A and simple score fusion C:", ""])
        lines.append(markdown_table(main_pairwise))

    core_ablations = comparisons[
        comparisons["comparison"].isin(
            ["E1_fft_vs_stft", "E2_shape_energy", "E3_fusion_location"]
        )
    ] if len(comparisons) else pd.DataFrame()
    lines.extend(["", "## E1/E2/E3 ablations", ""])
    if len(core_ablations):
        lines.append(
            markdown_table(
                core_ablations[
                    ["comparison", "dataset", "member", "f1_max", "delta_vs_reference"]
                ]
            )
        )
    else:
        lines.append("Core ablation results are incomplete.")

    lines.extend(["", "## E5 event accounting", ""])
    if len(events):
        lines.append(
            f"At each model's dataset-selected α, D recovered {int(events['recovered'].sum())} baseline misses and lost {int(events['lost'].sum())} events detected by A."
        )
        lines.append(
            f"The change in contextual false-positive intervals was {int(false_positives['delta_FP'].sum()):+d}."
        )
        lines.append(
            "Short/long classes use fixed point-count boundaries (≤56 and ≥224). Periodic and trend labels are reported only when explicitly annotated; all unclear events remain unclassified."
        )
    else:
        lines.append("A/D results are incomplete, so event accounting was not produced.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `results.csv`: per-signal metrics at every α.",
            "- `dataset_f1_by_alpha.csv`: equal-weight signal means before α selection.",
            "- `dataset_f1max.csv`: selected dataset results and contextual counts.",
            "- `macro_results.csv`: seven-dataset macro results.",
            "- `runtime.csv`: per-dataset/model inference time and peak CUDA allocation.",
            "- `main_pairwise.csv`: per-dataset D−A and D−C performance changes.",
            "- `event_analysis.csv` and `false_positive_changes.csv`: E5 details.",
            "- `point_scores/`: original per-time-point anomaly scores.",
            "",
        ]
    )
    return "\n".join(lines)


def select_variants(config: Mapping[str, Any], suite: str, variants: str | None) -> list[str]:
    if variants:
        chosen = [name.strip() for name in variants.split(",") if name.strip()]
    else:
        chosen = list(config["suites"][suite])
    unknown = sorted(set(chosen) - set(config["variants"]))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return chosen


def run(args: argparse.Namespace) -> int:
    config = load_json(args.config)
    variants = select_variants(config, args.suite, args.variants)
    documents = load_metric_documents(args.output_root)
    if not documents:
        raise RuntimeError(f"No metric JSON files found below {args.output_root / 'metrics'}")
    results = flatten_results(documents)
    dataset_alpha, f1max, macro = aggregate_dataset_level(results, config, variants)
    completion_failures = validate_completion(dataset_alpha, config, variants)
    if completion_failures and not args.allow_incomplete:
        preview = "\n".join(completion_failures[:20])
        raise RuntimeError(
            "Refusing to report F1-max from incomplete dataset/model groups. "
            "Finish the workers or pass --allow-incomplete for diagnostics only:\n" + preview
        )

    comparisons = comparison_table(f1max, config)
    main_pairwise = main_pairwise_table(f1max, config)
    runtime = runtime_table(results[results["model"].isin(variants)])
    events, false_positives = event_analysis(documents, f1max, config, args.output_root)
    artifacts = {
        "results.csv": results,
        "dataset_f1_by_alpha.csv": dataset_alpha,
        "dataset_f1max.csv": f1max,
        "macro_results.csv": macro,
        "comparisons.csv": comparisons,
        "main_pairwise.csv": main_pairwise,
        "runtime.csv": runtime,
        "event_analysis.csv": events,
        "false_positive_changes.csv": false_positives,
    }
    for filename, frame in artifacts.items():
        if publish_csv_no_replace(args.output_root / filename, frame):
            print(f"Wrote {args.output_root / filename}")
        else:
            print(f"Kept existing {args.output_root / filename}")
    plot_comparisons(f1max, config, args.output_root)
    plot_main_diagnostics(f1max, runtime, config, args.output_root)
    plot_representative_events(events, args.data_root, args.output_root, config)
    report = build_report(
        f1max,
        macro,
        comparisons,
        main_pairwise,
        runtime,
        events,
        false_positives,
        config,
    )
    if not publish_bytes_no_replace(args.output_root / "experiment_report.md", report.encode("utf-8")):
        print(f"Kept existing {args.output_root / 'experiment_report.md'}")
    if completion_failures:
        print("WARNING: incomplete diagnostic summary:\n" + "\n".join(completion_failures))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--suite", choices=["base", "core_ablations", "extended_ablations", "all"], default="all")
    parser.add_argument("--variants", help="Comma-separated variant names; overrides --suite")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
