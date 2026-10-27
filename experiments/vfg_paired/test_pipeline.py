#!/usr/bin/env python3
"""CPU-only synthetic tests; no CLIP model is imported."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from experiments.vfg_paired.runner import call_backend, publish_bytes_no_replace
from experiments.vfg_paired.summarize import (
    aggregate,
    completion_table,
    comparisons,
    event_changes,
    failure_plots,
    macro_table,
    report,
    runtime_tables,
)


def test_atomic_no_replace() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact"
        assert publish_bytes_no_replace(path, b"first")
        assert not publish_bytes_no_replace(path, b"second")
        assert path.read_bytes() == b"first"


def test_adapter() -> None:
    class Backend:
        def predict_variants(self, data, variants, context): return {name: {"scores": [0.0] * len(data)} for name in variants}
    module = SimpleNamespace(create_backend=lambda config, device, cache_dir: Backend())
    state = {}; data = pd.DataFrame({"timestamp": [1, 2], "value": [3.0, 4.0]})
    output = call_backend(module, state, {}, "cpu", Path("/tmp/unused"), data, {"A": {}}, {"signal": "toy"})
    assert set(output) == {"A"} and "instance" in state


def synthetic_config() -> dict:
    return {"datasets": ["real", "artificialWithAnomaly"], "real_datasets": ["real"], "expected_signal_counts": {"real": 2, "artificialWithAnomaly": 1}, "evaluation": {"alphas": [0.1, 0.01], "selection_protocol": "synthetic strict protocol"}, "variants_order": ["A", "B", "C", "D", "E"], "comparisons": [{"name": "B_minus_A", "left": "B", "right": "A"}]}


def test_strict_aggregation_and_macros() -> None:
    rows = []
    for model in ["A", "B", "C", "D", "E"]:
        for dataset, signals in {"real": ["s1", "s2"], "artificialWithAnomaly": ["s3"]}.items():
            for signal in signals:
                for alpha in [0.1, 0.01]:
                    # A/real: alpha .1 mean .5; alpha .01 mean .6. Per-signal
                    # max then mean would differ and must never be used.
                    f1 = 0.6 if alpha == 0.01 else (1.0 if signal == "s1" else 0.0)
                    if model == "B": f1 += 0.1
                    rows.append({"dataset": dataset, "signal": signal, "model": model, "alpha": alpha, "precision": f1, "recall": f1, "f1": f1, "TP": 1, "FP": 0, "FN": 0})
    config = synthetic_config(); _, best = aggregate(pd.DataFrame(rows), config)
    chosen = best[(best.dataset == "real") & (best.model == "A")].iloc[0]
    assert chosen.best_alpha == 0.01 and abs(chosen.f1_max - 0.6) < 1e-12
    macro = macro_table(best, config); changes = comparisons(best, macro, config)
    assert set(macro.scope) == {"real_6", "all_7"}
    assert abs(changes[(changes.scope == "real_6")].iloc[0].delta_f1max - 0.1) < 1e-12
    completion = completion_table(pd.DataFrame(rows), config)
    rendered = report(best, macro, changes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), completion, config)
    assert "real_6" in rendered and "all_7" in rendered and "Completion" in rendered


def test_incremental_timing() -> None:
    frame = pd.DataFrame([{"dataset": "real", "signal": "s1", "model": "A", "alpha": 0.1, "peak_cuda_memory_bytes": 10, "timing_inference_seconds": 5.5, "timing_matching_seconds": 2.0, "timing_frequency_seconds": 3.0}])
    runtime, incremental, per_signal = runtime_tables(frame, [])
    assert runtime.iloc[0].runtime_scope == "cache_only_incremental_no_clip"
    assert runtime.iloc[0].mean_variant_incremental_seconds == 5.5
    assert incremental.iloc[0].mean_matching_seconds == 2.0
    assert per_signal.iloc[0].frequency_seconds == 3.0


def test_event_accounting_and_failure_plot() -> None:
    config = {
        "datasets": ["toy"],
        "comparisons": [{"name": "B_minus_A", "left": "B", "right": "A"}],
    }
    truth = [[2, 3], [6, 7]]
    def document(model: str, detected: list[list[int]], tp: int, fp: int, fn: int) -> dict:
        return {"dataset": "toy", "signal": "s", "variant": model, "ground_truth_intervals": truth, "alpha_metrics": [{"alpha": 0.1, "TP": tp, "FP": fp, "FN": fn, "detected_intervals": detected}]}
    documents = [document("A", [[2, 3], [20, 20]], 1, 2, 1), document("B", [[6, 7], [20, 20]], 1, 3, 1)]
    best = pd.DataFrame([{"dataset": "toy", "model": "A", "best_alpha": 0.1}, {"dataset": "toy", "model": "B", "best_alpha": 0.1}])
    events, totals = event_changes(documents, best, config)
    assert set(events.change) == {"lost", "recovered"}
    summary = totals.iloc[0]
    assert summary.FN_recovered == 1 and summary.TP_lost == 1 and summary.FP_change == 1

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); data_root = root / "data"; output_root = root / "output"
        (data_root / "toy").mkdir(parents=True)
        pd.DataFrame({"timestamp": np.arange(32), "value": np.sin(np.arange(32))}).to_csv(data_root / "toy/s.csv", index=False)
        score_dir = output_root / "point_scores/toy/s"; score_dir.mkdir(parents=True)
        np.savez_compressed(score_dir / "A.npz", scores=np.linspace(0, 1, 32), timestamps=np.arange(32))
        np.savez_compressed(score_dir / "B.npz", scores=np.linspace(1, 0, 32), timestamps=np.arange(32))
        changes = pd.DataFrame([{"scope": "dataset", "dataset": "toy", "comparison": "B_minus_A", "delta_f1max": -0.1}])
        failure_plots(events, changes, data_root, output_root, config)
        assert len(list((output_root / "plots/failures").glob("*.png"))) == 1


if __name__ == "__main__":
    test_atomic_no_replace(); test_adapter(); test_strict_aggregation_and_macros(); test_incremental_timing(); test_event_accounting_and_failure_plot(); print("paired pipeline checks passed")
