from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.curve_aware.runner import (
    call_backend,
    configure_final_cache_roots,
    discover_tasks,
    load_signal,
    publish_bytes_no_replace,
)
from experiments.curve_aware.summarize import (
    aggregate,
    candidate_pool_audit,
    comparisons,
    completion_table,
    macro_table,
    report,
)


def tiny_config() -> dict:
    variants = {
        "A_stable_final": {"family": "A", "layer": "final", "base_sampling": "full_2d_patch"},
        "B_paired_2d_final": {"family": "B", "layer": "final", "base_sampling": "full_2d_patch"},
        "D_curve_bilinear_final": {"family": "D", "layer": "final", "base_sampling": "curve_bilinear"},
        "E_curve_gaussian_mid": {"family": "E", "layer": "mid", "base_sampling": "curve_gaussian"},
    }
    return {
        "datasets": ["real", "YAHOOA1"],
        "expected_signal_counts": {"real": 2, "YAHOOA1": 1},
        "macro_scopes": {"real_1": ["real"], "all_2": ["real", "YAHOOA1"]},
        "evaluation": {"alphas": [0.1, 0.01], "selection_protocol": "strict toy protocol"},
        "model": {"candidate_pool": {"max_common_visual_windows": 32, "overlap_exclusion_window_index_radius": 4}},
        "variants": variants,
        "variants_order": list(variants),
        "comparisons": [{"name": "D_minus_B", "left": "D_curve_bilinear_final", "right": "B_paired_2d_final"}],
    }


def test_config_has_492_signals_and_nine_variants() -> None:
    config = json.loads((Path(__file__).parents[1] / "experiments/curve_aware/config.json").read_text())
    assert sum(config["expected_signal_counts"].values()) == 492
    assert len(config["datasets"]) == 11 and len(config["variants_order"]) == 9
    assert set(config["macro_scopes"]) == {"real_6", "original_7", "yahoo_4", "all_11"}
    paired = [config["variants"][name] for name in config["variants_order"] if name != "A_stable_final"]
    assert {item["layer"] for item in paired} == {"final", "mid"}


def test_atomic_publish_and_adapter(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    assert publish_bytes_no_replace(target, b"first")
    assert not publish_bytes_no_replace(target, b"second")
    assert target.read_bytes() == b"first"

    class Backend:
        def predict_variants(self, data, variants, context):
            return {name: {"scores": np.zeros(len(data))} for name in variants}

    module = SimpleNamespace(create_backend=lambda config, device, cache_dir: Backend())
    state = {}
    data = pd.DataFrame({"timestamp": [1, 2], "value": [3.0, 4.0]})
    result = call_backend(module, state, {}, "cpu", tmp_path, data, {"A": {}}, {"signal": "s"})
    assert set(result) == {"A"} and "instance" in state
    config = {"cache": {"never_replace": True}}
    roots = configure_final_cache_roots(config, [tmp_path / "one", tmp_path / "two"])
    assert config["cache"]["final_cache_roots"] == roots
    assert roots == [str((tmp_path / "one").resolve()), str((tmp_path / "two").resolve())]


def test_yahoo_index_is_authoritative_and_uses_row_positions(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    for name in ("r1", "r2"):
        pd.DataFrame({"timestamp": [2, 1], "value": [2.0, 1.0]}).to_csv(tmp_path / "real" / f"{name}.csv", index=False)
    yahoo = tmp_path / "Yahoo_S5/A1Benchmark"; yahoo.mkdir(parents=True)
    pd.DataFrame({"timestamp": [100, 50], "value": [5.0, 4.0], "is_anomaly": [0, 1]}).to_csv(yahoo / "y1.csv", index=False)
    pd.DataFrame({"timestamp": [1], "value": [99.0], "is_anomaly": [0]}).to_csv(yahoo / "aggregate.csv", index=False)
    (tmp_path / "datasets.csv").write_text('YAHOOA1,"(\'y1\',)"\n', encoding="utf-8")
    config = {"datasets": ["real", "YAHOOA1"], "expected_signal_counts": {"real": 2, "YAHOOA1": 1}}
    tasks = discover_tasks(config, tmp_path)
    assert [(d, s) for d, s, _ in tasks] == [("real", "r1"), ("real", "r2"), ("YAHOOA1", "y1")]
    loaded = load_signal(yahoo / "y1.csv", "YAHOOA1")
    assert loaded.timestamp.tolist() == [1, 2] and loaded.value.tolist() == [5.0, 4.0]


def test_strict_dataset_alpha_selection_and_report() -> None:
    config = tiny_config(); rows = []
    for model in config["variants_order"]:
        for dataset, signals in {"real": ["s1", "s2"], "YAHOOA1": ["s3"]}.items():
            for signal in signals:
                for alpha in config["evaluation"]["alphas"]:
                    value = 0.6 if alpha == 0.01 else (1.0 if signal == "s1" else 0.0)
                    if model.startswith("D_"): value += 0.1
                    rows.append({"dataset": dataset, "signal": signal, "model": model, "alpha": alpha,
                                 "precision": value, "recall": value, "f1": value, "TP": 1, "FP": 0, "FN": 0})
    results = pd.DataFrame(rows); _, best = aggregate(results, config)
    selected = best[(best.dataset == "real") & (best.model == "B_paired_2d_final")].iloc[0]
    assert selected.best_alpha == 0.01 and selected.f1_max == 0.6
    macro = macro_table(best, config); changes = comparisons(best, macro, config)
    assert abs(changes[changes.scope == "all_2"].iloc[0].delta_f1max - 0.1) < 1e-12
    completion = completion_table(results, config)
    text = report(best, macro, changes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), completion, config)
    assert "Curve-aware ViT4TS" in text and "strict toy protocol" in text


def test_candidate_audit_flattens_shared_pool_metadata() -> None:
    config = tiny_config(); documents = []
    for model in config["variants_order"]:
        documents.append({
            "dataset": "real", "signal": "s1", "variant": model,
            "backend_metadata": {"candidate_pool_min_windows": 3, "candidate_pool_max_windows": 32,
                                 "candidate_pool_digest": "same-window-ids"},
        })
    audit = candidate_pool_audit(documents, config)
    assert len(audit) == 3
    assert audit.pool_digest.nunique() == 1
    assert audit.min_windows.min() == 3 and audit.max_windows.max() == 32
