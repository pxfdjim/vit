"""Read-only audit of the fixed ViT4TS first-stage baseline and its inputs.

The cached 224/56 scores were produced by the earlier seven-dataset run.  This
script checks their provenance and reproduces the project's contextual interval
metrics without loading CLIP or changing any model code.
"""

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.evaluate import evaluate_intervals, window_wise_metrics
from models.model_utils import compute_detection_intervals
from preprocessing.data_utils import intervals_from_indices

DATASETS = (
    "artificialWithAnomaly", "realAWSCloudwatch", "realAdExchange",
    "realTraffic", "realTweets", "MSL", "SMAP",
)
ALPHAS = (0.1, 0.01, 0.001)
FINAL = ROOT / "data/runs/20261016_vit4ts_final7"
OLD = ROOT / "data/runs/20261016_vit4ts_diagnostics"
WEIGHT_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_index_and_labels():
    with (ROOT / "data/datasets.csv").open(newline="") as source:
        index = {dataset: sorted(ast.literal_eval(names)) for dataset, names in csv.reader(source)}
    labels = {}
    with (ROOT / "data/anomalies.csv").open() as source:
        next(source)
        for line in source:
            name, intervals = line.strip().split(",", 1)
            labels[name] = ast.literal_eval(intervals.strip('"'))
    return index, labels


def score_path(dataset, signal):
    final = FINAL / "scores" / dataset / signal / "window_224.npz"
    old = OLD / "scores" / dataset / signal / "paper_window_224.npz"
    if final.exists():
        return final, "top"
    if old.exists():
        return old, "corrected"
    raise FileNotFoundError(f"No cached 224/56 first-stage score: {dataset}/{signal}")


def score_metrics(scores, timestamps, labels, alpha):
    index_intervals, _, _ = compute_detection_intervals(scores, alpha=alpha)
    predicted = intervals_from_indices(index_intervals, timestamps, scores)
    predicted = predicted[["start", "end"]].values.tolist() if len(predicted) else []
    metric = evaluate_intervals(labels, predicted)
    counts = window_wise_metrics(labels, predicted)
    return {**counts, "precision": metric["precision"],
            "recall": metric["recall"], "f1": metric["F1"],
            "predicted_intervals": predicted}


def audit_challenge_outputs(output):
    """Check completed diagnostics from saved arrays; this never reruns CLIP."""
    challenge_root = output.parent
    scale_rows = []
    scale_metric_checks = 0
    for case in ("aws", "ad", "traffic", "artificial", "tweets", "msl", "smap"):
        path = challenge_root / "scale" / (case + ".json")
        if not path.exists():
            raise FileNotFoundError(f"Incomplete scale run: {path}")
        report = json.loads(path.read_text())
        target = slice(report["target_index_interval"][0], report["target_index_interval"][1] + 1)
        spike = report["spike_index"]
        if report["spike_distance_from_target"] <= 224:
            raise AssertionError(f"Peak overlaps a target window: {case}")
        if any(a <= spike <= b for a, b in report["original_labels_index_intervals"]):
            raise AssertionError(f"Peak placed in a labelled original anomaly: {case}")
        baseline_path = ROOT / report["records"]["baseline"]["score_file"]
        with np.load(baseline_path) as cached:
            baseline_raw = cached["raw"]
            baseline_proc = cached["processed"]
            baseline_score = cached["scores"]
        for condition, record in report["records"].items():
            with np.load(ROOT / record["score_file"]) as cached:
                raw, proc, score = (cached[key] for key in ("raw", "processed", "scores"))
                timestamps = cached["timestamps"]
            if not np.array_equal(raw[target], baseline_raw[target]):
                raise AssertionError(f"Target raw values changed: {case}/{condition}")
            changed = np.flatnonzero(raw != baseline_raw)
            if condition == "baseline" and len(changed):
                raise AssertionError(f"Baseline changed: {case}")
            if condition != "baseline" and not np.array_equal(changed, [spike]):
                raise AssertionError(f"Intervention changed points outside peak: {case}/{condition}")
            if not np.isclose(score[target].mean(), record["target_score_mean"], rtol=0, atol=1e-12):
                raise AssertionError(f"Reported target score inconsistent: {case}/{condition}")
            if condition == "10_fixed_trend_scale_fixedref" and not np.allclose(
                proc[target], baseline_proc[target], rtol=0, atol=1e-12
            ):
                raise AssertionError(f"Fixed trend/scale failed to hold target: {case}")
            truth = [(timestamps[a], timestamps[b])
                     for a, b in report["original_labels_index_intervals"]]
            truth_with_peak = truth + ([] if condition == "baseline" else
                                       [(timestamps[spike], timestamps[spike])])
            if [float(x["alpha"]) for x in record["metrics"]] != list(ALPHAS):
                raise AssertionError(f"Original alpha set changed: {case}/{condition}")
            for stored in record["metrics"]:
                found, _, _ = compute_detection_intervals(score, alpha=stored["alpha"])
                if [list(x) for x in found] != stored["detected_index_intervals"]:
                    raise AssertionError(f"Scale intervals mismatch: {case}/{condition}")
                predicted = [(timestamps[a], timestamps[b]) for a, b in found]
                result = evaluate_intervals(truth, predicted)
                counts = window_wise_metrics(truth, predicted)
                augmented = evaluate_intervals(truth_with_peak, predicted)
                if (not np.isclose(result["F1"], stored["f1"])
                        or not np.isclose(augmented["F1"], stored["contextual_f1_with_peak_label"])
                        or (counts["TP"], counts["FP"], counts["FN"])
                        != (stored["TP"], stored["FP"], stored["FN"])):
                    raise AssertionError(f"Scale F1/TP/FP/FN mismatch: {case}/{condition}")
                scale_metric_checks += 1
            scale_rows.append({"case": case, "condition": condition,
                               "target_score_mean": float(score[target].mean()),
                               "target_score_ratio_to_baseline": float(
                                   score[target].mean() / baseline_score[target].mean()),
                               "target_pixel_mae": record["target_window_pixel_mae_vs_baseline"],
                               "target_hit_alpha_0p1": record["metrics"][0]["target_detected"]})

    reference_checks = {}
    for kind in ("reference", "reference_real"):
        reference = challenge_root / kind
        metadata = json.loads((reference / "metadata.json").read_text())
        if metadata["sequence_level_image_max_abs_diff_20pct"] > 1e-6:
            raise AssertionError(f"{kind}: assembled images differ from direct rendering")
        if kind == "reference" and metadata["source_score_max_abs_diff_0pct"] > 1e-6:
            raise AssertionError("Reference diagnostic differs from source scorer")
        summary = pd.read_csv(reference / "summary.csv")
        alpha_rows = pd.read_csv(reference / "threshold_details.csv")
        target_start, target_end = metadata["target_interval"]
        target_slice = slice(target_start, target_end + 1)
        with np.load(reference / "raw_scores.npz") as raw_scores:
            for level in (0, 5, 10, 20, 40):
                subset = summary[summary.level_percent == level].set_index("mode")
                self_extra = len(metadata["target_windows"]) / 80 * 100
                for mode, expected_percent in (("external_reference", level),
                                               ("self_reference", level + self_extra), ("fixed", 0)):
                    observed = float(subset.loc[mode, "actual_total_window_percent"])
                    if observed != expected_percent:
                        raise AssertionError(f"{kind}: bad reference-window percentage: "
                                             f"{level}/{mode}: {observed}")
                if not np.array_equal(raw_scores[f"fixed_{level}"], raw_scores["fixed_0"]):
                    raise AssertionError(f"{kind}: fixed reference scores changed: {level}")
                for mode in ("external_reference", "self_reference", "fixed"):
                    vec = raw_scores[f"{mode}_{level}"]
                    row = subset.loc[mode]
                    if not np.isclose(vec[target_slice].mean(), row.target_mean_score, atol=1e-12):
                        raise AssertionError(f"{kind}: target mean mismatch: {level}/{mode}")
                    matching = alpha_rows[(alpha_rows.level_percent == level)
                                          & (alpha_rows["mode"] == mode)]
                    if len(matching) != 3 or not np.isclose(matching.f1.max(), row.f1_max):
                        raise AssertionError(f"{kind}: F1-max mismatch: {level}/{mode}")
                    for observed in matching.itertuples():
                        found, _, _ = compute_detection_intervals(vec, alpha=observed.alpha)
                        if [list(x) for x in found] != json.loads(observed.detected_intervals):
                            raise AssertionError(f"{kind}: prediction intervals mismatch: {level}/{mode}")
                        metric = evaluate_intervals([(target_start, target_end)], found)
                        counts = window_wise_metrics([(target_start, target_end)], found)
                        if (not np.isclose(metric["F1"], observed.f1)
                                or (counts["TP"], counts["FP"], counts["FN"])
                                != (observed.TP, observed.FP, observed.FN)):
                            raise AssertionError(f"{kind}: interval metrics mismatch: {level}/{mode}")
        if kind == "reference_real":
            source = pd.read_csv(metadata["source"]).sort_values("timestamp").reset_index(drop=True)
            crop_start = metadata["crop_start"]
            raw_target = source.value.to_numpy()[crop_start + target_start:crop_start + target_end + 1]
            if len(raw_target) != target_end - target_start + 1:
                raise AssertionError("Real reference target falls outside source crop")
            if metadata["target_timestamps"] != [int(source.timestamp.iloc[crop_start + x])
                                                 for x in (target_start, target_end)]:
                raise AssertionError("Real reference target timestamp mismatch")
            _, all_labels = load_index_and_labels()
            if all_labels["A-6"] != [metadata["target_timestamps"]]:
                raise AssertionError("Real reference target does not match original label")
            if any(not (end < target_start or start > target_end)
                   for start, end in metadata["injection_intervals"]):
                raise AssertionError("Real reference copied anomaly into target")
            for (start, end), windows in zip(metadata["injection_intervals"],
                                             metadata["injection_windows"]):
                actual = [i for i in range(80) if i * 56 <= end and i * 56 + 223 >= start]
                if actual != windows or set(actual) & set(metadata["target_windows"]):
                    raise AssertionError("Real reference injection window accounting mismatch")
            if set(metadata["donor_windows"]) & set(metadata["target_windows"]):
                raise AssertionError("Real reference normal donors overlap target")
        reference_checks[kind] = {"sequence_level_image_max_abs_diff_20pct":
                                  metadata["sequence_level_image_max_abs_diff_20pct"],
                                  "reference_rows": len(summary),
                                  "f1_max_rows_verified": len(summary)}
        if kind == "reference":
            reference_checks[kind]["source_score_max_abs_diff_0pct"] = (
                metadata["source_score_max_abs_diff_0pct"])
    checks = {"scale": scale_rows, "scale_metric_rows_verified": scale_metric_checks,
              **reference_checks}
    (output / "challenge_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/runs/vit4ts_challenges/audit")
    parser.add_argument("--check-challenges", action="store_true",
                        help="Validate completed scale/reference outputs without rerunning CLIP")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = ROOT / "data/model_cache/ViT-B-16.pt"
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != WEIGHT_SHA256:
        raise RuntimeError("OpenAI CLIP ViT-B/16 checkpoint SHA-256 mismatch")
    prior_manifest = json.loads((FINAL / "audit_manifest.json").read_text())
    cached_source_sha = prior_manifest["source_sha256"]
    current_source_sha = {name: file_sha256(ROOT / name) for name in cached_source_sha}
    changed_sources = [name for name in cached_source_sha
                       if current_source_sha[name] != cached_source_sha[name]]
    core_sources = ("src/models/vit4ts.py", "src/models/model_utils.py",
                    "src/preprocessing/preprocess.py", "src/evaluation/evaluate.py")
    if any(name in changed_sources for name in core_sources):
        raise RuntimeError("ViT4TS core differs from the cached baseline; recompute scores")

    index, labels = load_index_and_labels()
    rows, candidate_rows, dataset_rows = [], [], []
    provenance = {"final7_cache": 0, "diagnostics_cache": 0}
    metric_mismatches = []
    for dataset in DATASETS:
        n_events = 0
        n_points = 0
        n_duplicate_timestamps = 0
        for signal in index[dataset]:
            frame = pd.read_csv(ROOT / "data" / dataset / (signal + ".csv"))
            frame = frame.sort_values("timestamp")
            ts = frame.timestamp.to_numpy()
            n_points += len(frame)
            n_duplicate_timestamps += int(pd.Index(ts).duplicated().sum())
            truth = labels[signal]
            n_events += len(truth)
            path, key = score_path(dataset, signal)
            provenance["final7_cache" if key == "top" else "diagnostics_cache"] += 1
            with np.load(path) as cached:
                scores = np.asarray(cached[key], dtype=float)
                cached_ts = cached["timestamps"]
            if len(scores) != len(ts) or not np.array_equal(ts, cached_ts):
                raise RuntimeError(f"Cached score alignment mismatch: {dataset}/{signal}")
            if not np.isfinite(scores).all():
                raise RuntimeError(f"Nonfinite cached score: {dataset}/{signal}")
            record = json.loads((FINAL / "results" / dataset / (signal + ".json")).read_text())
            if (record["configs"]["top_224"]["window"] != 224
                    or record["configs"]["top_224"]["stride"] != 56
                    or record["clip"] != "OpenAI CLIP ViT-B/16"
                    or record["preprocess"] != "detrend_minmax"):
                raise RuntimeError(f"Unexpected baseline config: {dataset}/{signal}")
            stored = {float(row["alpha"]): row for row in record["configs"]["top_224"]["metrics"]}
            for alpha in ALPHAS:
                observed = score_metrics(scores, ts, truth, alpha)
                expected = stored[alpha]
                if (observed["predicted_intervals"] != expected["predicted_intervals"]
                        or any(abs(observed[name] - expected[name]) > 1e-10
                               for name in ("TP", "FP", "FN", "f1"))):
                    metric_mismatches.append({"dataset": dataset, "signal": signal, "alpha": alpha})
                rows.append({"dataset": dataset, "signal": signal, "alpha": alpha,
                             "TP": observed["TP"], "FP": observed["FP"],
                             "FN": observed["FN"], "f1": observed["f1"],
                             "score_source": str(path.relative_to(ROOT))})
            # These are screening candidates only; score rank is not a definition
            # of weak ground-truth amplitude.
            for event_id, (start, end) in enumerate(truth):
                mask = (ts >= start) & (ts <= end)
                if not mask.any():
                    continue
                event_scores = scores[mask]
                rank = float(np.mean(scores <= np.median(event_scores)))
                hit_001 = any(not (pred_end < start or end < pred_start)
                              for pred_start, pred_end in stored[0.01]["predicted_intervals"])
                candidate_rows.append({"dataset": dataset, "signal": signal,
                                       "event_id": event_id, "start": start, "end": end,
                                       "event_points": int(mask.sum()),
                                       "median_score": float(np.median(event_scores)),
                                       "median_score_percentile": rank,
                                       "missed_at_alpha_0p01": int(not hit_001)})
        dataset_rows.append({"dataset": dataset, "signals": len(index[dataset]),
                             "points": n_points, "ground_truth_intervals": n_events,
                             "duplicate_timestamps": n_duplicate_timestamps})

    metrics = pd.DataFrame(rows)
    # Unweighted Contextual F1-max: mean signal F1 at each fixed alpha,
    # followed by the maximum over the paper's three fixed alpha values.
    by_alpha = metrics.groupby(["dataset", "alpha"], as_index=False).agg(
        mean_signal_f1=("f1", "mean"), n_signals=("signal", "nunique"))
    f1max = by_alpha.loc[by_alpha.groupby("dataset").mean_signal_f1.idxmax()]
    metrics.to_csv(output / "per_signal_alpha.csv", index=False)
    by_alpha.to_csv(output / "unweighted_contextual_f1_by_alpha.csv", index=False)
    f1max.to_csv(output / "unweighted_contextual_f1max.csv", index=False)
    pd.DataFrame(candidate_rows).sort_values(
        ["missed_at_alpha_0p01", "median_score_percentile"], ascending=[False, True]
    ).to_csv(output / "low_score_event_candidates.csv", index=False)
    audit = {"datasets": dataset_rows, "score_cache_provenance": provenance,
             "checkpoint_sha256": checkpoint_sha,
             "current_source_sha256": current_source_sha,
             "cached_source_sha256": cached_source_sha,
             "source_changes_since_cache": changed_sources,
             "metrics_recomputed": len(rows), "metric_mismatches": metric_mismatches,
             "protocol": {"window": 224, "stride": 56, "image_size": [224, 224],
                          "clip": "OpenAI CLIP ViT-B/16", "alphas": list(ALPHAS),
                          "f1max": "max over alpha of unweighted mean signal contextual F1"}}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"signals": len(metrics) // len(ALPHAS),
                      "metrics_recomputed": len(rows),
                      "metric_mismatches": len(metric_mismatches),
                      "cache_provenance": provenance}, indent=2))
    if metric_mismatches:
        raise RuntimeError("Cached baseline metric mismatches; see audit.json")
    if args.check_challenges:
        checks = audit_challenge_outputs(output)
        print(json.dumps({"scale_rows_verified": len(checks["scale"]),
                          "scale_metric_rows_verified": checks["scale_metric_rows_verified"],
                          "reference": checks["reference"],
                          "reference_real": checks["reference_real"]}, indent=2))


if __name__ == "__main__":
    main()
