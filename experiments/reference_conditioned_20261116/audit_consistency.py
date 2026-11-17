"""Audit train/inference inputs and posterior outputs on a transductive signal."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reference_conditioned_20261116.model import load_model_checkpoint
from experiments.reference_conditioned_20261116.predict import (load_visual_features,
                                                                  stored_t3_reference)
from src.core.reference import fitted_standardized_windows, patch_tokens
from src.evaluation.orionbench_protocol import evaluation_frame, signal_names


@torch.inference_mode()
def main():
    run = Path(sys.argv[1]).resolve()
    device = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda:0")
    config = json.loads((ROOT / "configs/model.json").read_text())
    dataset = "realTraffic"
    signal = signal_names(Path(config["data_root"]), dataset)[0]
    cache = torch.load(ROOT / config["synthetic_training_cache_root"] /
                       f"{dataset}.pt", map_location="cpu", weights_only=False)
    indices = [i for i, row in enumerate(json.loads(cache["records_json"]))
               if row["signal"] == signal]
    frame = evaluation_frame(Path(config["data_root"]), dataset, signal)
    raw = frame.value.to_numpy(np.float64)
    stats = json.loads(Path(config["normalization_stats"]).read_text())[dataset][signal]
    inferred_numeric = fitted_standardized_windows(raw, stats, 0)
    block, _ = load_visual_features(raw, {"dataset": dataset, "signal": signal}, config)
    inferred_visual = patch_tokens(block["intermediate_patch"]).half()
    train_numeric = cache["raw"][indices]
    train_visual = cache["visual"][indices]
    if inferred_numeric.shape != train_numeric.shape or inferred_visual.shape != train_visual.shape:
        raise RuntimeError("training/inference window count differs")
    frozen = ROOT / config["frozen_b3_root"] / dataset / f"{signal}.npz"
    with np.load(frozen, allow_pickle=False) as stored:
        train_ids = stored["clean_reference_ids"].astype(np.int64)
        train_anchor = stored["clean_reference_columns"].astype(np.float32)
        if not np.array_equal(stored["source_indices"], indices):
            raise RuntimeError("window source order differs")
    infer_ids, infer_anchor = stored_t3_reference(
        Path(config["feature_scoring_run"]) / "feature_diagnostics" /
        dataset / signal / "evidence.npz", len(indices))
    input_errors = {
        "raw_windows_max_abs": float((inferred_numeric - train_numeric).abs().max()),
        "visual_patches_max_abs": float((inferred_visual.float() - train_visual.float()).abs().max()),
        "reference_ids_equal": bool(np.array_equal(train_ids, infer_ids.numpy())),
        "t3_centers_max_abs": float(np.max(np.abs(train_anchor - infer_anchor.numpy()))),
    }
    model, _ = load_model_checkpoint(run / "checkpoints/posterior/realTraffic.pt", device)
    model.eval()
    train_n = model.numeric(train_numeric.to(device))
    infer_n = model.numeric(inferred_numeric.to(device))
    train_v = model.visual(train_visual.float().to(device))
    infer_v = model.visual(inferred_visual.float().to(device))
    train_output = model.forward_tokens(train_n, train_v, train_n[torch.from_numpy(train_ids).to(device)],
                                        train_v[torch.from_numpy(train_ids).to(device)],
                                        torch.from_numpy(train_anchor).to(device))
    infer_output = model.forward_tokens(infer_n, infer_v, infer_n[infer_ids.to(device)],
                                        infer_v[infer_ids.to(device)], infer_anchor.to(device))
    output_errors = {key: float((train_output[key] - infer_output[key]).abs().max())
                     for key in ("prior", "delta")}
    output_errors["posterior"] = float((train_output["alignment"]["soft_match"] -
                                        infer_output["alignment"]["soft_match"]).abs().max())
    output_errors["learned_residual"] = float((train_output["representation"]["joint"] -
                                               infer_output["representation"]["joint"]).abs().max())
    result = {"dataset": dataset, "signal": signal,
              "input_errors": input_errors, "output_errors": output_errors,
              "status": "passed"}
    if not input_errors["reference_ids_equal"] or \
            max(value for key, value in input_errors.items() if key != "reference_ids_equal") > 1e-5 or \
            max(output_errors.values()) > 1e-5:
        result["status"] = "failed"
    (run / "train_inference_consistency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if result["status"] != "passed":
        raise RuntimeError("training/inference consistency failed")


if __name__ == "__main__":
    main()
