#!/usr/bin/env python3
"""Measure actual Full-checkpoint gradient paths on one unlabeled batch per dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.train_amr_ablation import gradient_audit
from src.models.amr_training_ablation import AblationReferenceModel, AlignmentLossConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = []
    for dataset in config["datasets"]:
        cache = torch.load(Path(config["training_cache_root"]) / f"{dataset}.pt",
                           map_location="cpu", weights_only=False)
        checkpoint = torch.load(Path(config["full_checkpoint_root"]) / f"{dataset}.pt",
                                map_location="cpu", weights_only=True)
        model = AblationReferenceModel("cross").to(args.device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        count = int(config["training"]["batch_size"])
        raw = cache["raw"][:count].to(args.device)
        visual = cache["visual"][:count].float().to(args.device)
        audit = gradient_audit(model, raw, visual, "cross_modal_alignment",
                               AlignmentLossConfig(),
                               float(config["numeric_self_supervision"]["mask_fraction"]))
        rows.extend({"dataset": dataset, "variant": "Full", "audit_state": "final_checkpoint",
                     **item} for item in audit)
        print(dataset, len(audit), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "gradient_path_audit_full.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
