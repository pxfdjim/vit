"""Independent checkpoint, score, and frozen-base integrity checks."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.orionbench_protocol import DATASETS, signal_names


def main():
    run = Path(sys.argv[1]).resolve()
    config = json.loads((ROOT / "configs/model.json").read_text())
    baseline = ROOT / "data/runs/model_refactor_v2_20261116_eval"
    summary = {"checkpoints": 0, "signals": 0, "point_scores": 0,
               "max_base_error": 0.0, "max_formula_error": 0.0,
               "status": "passed"}
    for mode in ("posterior", "no_reference", "t3_direct"):
        inferred = run / "inference" / mode
        metrics = pd.read_csv(inferred / "metrics_per_signal.csv")
        if len(metrics) != 984 or len(metrics[metrics.variant == "Model"]) != 492:
            raise RuntimeError(f"incomplete evaluation for {mode}")
        if metrics[metrics.variant == "Model"].aff_f1.notna().sum() != 488:
            raise RuntimeError(f"unexpected valid signal count for {mode}")
        for dataset in DATASETS:
            path = run / "checkpoints" / mode / f"{dataset}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["model_config"]["mode"] != mode or \
                    payload["model_config"]["architecture"] != "reference_conditioned_v1" or \
                    payload["seed"] != 20261104 or len(payload["history"]) != 16 or \
                    payload["labels_used"] or payload["checkpoint_selection"] != "fixed_final_epoch":
                raise RuntimeError(f"invalid checkpoint: {path}")
            summary["checkpoints"] += 1
            for signal in signal_names(Path(config["data_root"]), dataset):
                scored = inferred / "point_scores" / dataset / signal
                with np.load(scored / "B3.npz", allow_pickle=False) as src, \
                        np.load(scored / "Model.npz", allow_pickle=False) as dst, \
                        np.load(baseline / "point_scores" / dataset / signal /
                                "B3.npz", allow_pickle=False) as original, \
                        np.load(inferred / "feature_diagnostics" / dataset /
                                signal / "evidence.npz", allow_pickle=False) as features:
                    base, final, reference = src["scores"], dst["scores"], original["scores"]
                    delta = features["point_delta"]
                    if not np.array_equal(src["timestamps"], dst["timestamps"]) or \
                            not np.array_equal(src["timestamps"], original["timestamps"]):
                        raise RuntimeError(f"timestamp mismatch: {mode}/{dataset}/{signal}")
                    summary["max_base_error"] = max(summary["max_base_error"],
                                                     float(np.max(np.abs(base - reference))))
                    summary["max_formula_error"] = max(summary["max_formula_error"],
                                                        float(np.max(np.abs(final -
                                                            (base + .05 * delta)))))
                    summary["point_scores"] += len(final)
                    summary["signals"] += 1
    if summary["max_base_error"] != 0 or summary["max_formula_error"] > 1e-12:
        raise RuntimeError(summary)
    (run / "validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
