#!/usr/bin/env python3
"""CPU-only checks for orchestration invariants; this never imports CLIP."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vfg_vit4ts.runner import call_backend, load_json, publish_bytes_no_replace
from experiments.vfg_vit4ts.summarize import (
    aggregate_dataset_level,
    plot_comparisons,
    plot_main_diagnostics,
)


def test_atomic_no_replace() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "artifact.txt"
        assert publish_bytes_no_replace(target, b"first")
        assert not publish_bytes_no_replace(target, b"second")
        assert target.read_bytes() == b"first"


def test_dataset_level_alpha_order() -> None:
    # Per-signal max then mean would incorrectly yield 0.8. The required
    # mean-per-alpha then max gives 0.6.
    rows = []
    for signal, values in {"s1": {0.1: 1.0, 0.01: 0.6}, "s2": {0.1: 0.0, 0.01: 0.6}}.items():
        for alpha, f1 in values.items():
            rows.append(
                {
                    "dataset": "toy",
                    "model": "candidate",
                    "signal": signal,
                    "alpha": alpha,
                    "precision": f1,
                    "recall": f1,
                    "f1": f1,
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "inference_seconds": 1.0,
                    "peak_cuda_memory_bytes": None,
                }
            )
    config = {
        "datasets": ["toy"],
        "expected_signal_counts": {"toy": 2},
        "evaluation": {"alphas": [0.1, 0.01]},
    }
    _, best, macro = aggregate_dataset_level(pd.DataFrame(rows), config, ["candidate"])
    assert len(best) == 1
    assert best.iloc[0]["best_alpha"] == 0.01
    assert abs(best.iloc[0]["f1_max"] - 0.6) < 1e-12
    assert abs(macro.iloc[0]["macro_f1max"] - 0.6) < 1e-12


def test_backend_adapter() -> None:
    calls = []

    class FakeBackend:
        def predict_variants(self, data, variants, context):
            calls.append((len(data), sorted(variants), context["signal"]))
            return {name: [0.0] * len(data) for name in variants}

    module = SimpleNamespace(
        create_backend=lambda config, device, cache_dir: FakeBackend()
    )
    state = {}
    output = call_backend(
        module=module,
        data=pd.DataFrame({"timestamp": [1, 2], "value": [3.0, 4.0]}),
        variant_configs={"A": {}},
        experiment_config={},
        device="cpu",
        cache_dir=Path("/tmp/unused-vfg-test-cache"),
        context={"signal": "toy"},
        backend_state=state,
    )
    assert output == {"A": [0.0, 0.0]}
    assert calls == [(2, ["A"], "toy")]
    assert "instance" in state


def test_plot_entry_points() -> None:
    config = load_json(Path(__file__).with_name("configs.json"))
    config["datasets"] = ["toy"]
    variants = config["suites"]["all"]
    f1max = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": model,
                "f1_max": 0.5 + index / 100,
                "TP": 10 + index,
                "FP": 2 + index,
                "FN": 3,
            }
            for index, model in enumerate(variants)
        ]
    )
    runtime = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "model": model,
                "mean_inference_seconds": 1.0 + index / 10,
            }
            for index, model in enumerate(variants)
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plot_comparisons(f1max, config, root)
        plot_main_diagnostics(f1max, runtime, config, root)
        expected = {
            "base.png",
            "base_delta_vs_A.png",
            "base_contextual_counts.png",
            "base_runtime.png",
            "E1_fft_vs_stft.png",
            "E2_shape_energy.png",
            "E3_fusion_location.png",
        }
        found = {path.name for path in (root / "plots").glob("*.png")}
        assert expected.issubset(found), (expected, found)


if __name__ == "__main__":
    test_atomic_no_replace()
    test_dataset_level_alpha_order()
    test_backend_adapter()
    test_plot_entry_points()
    print("pipeline checks passed")
