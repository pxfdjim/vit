#!/usr/bin/env python3
"""Audit TREM input use before and after APRA's bounded residual capacity."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.train_apra_trem import projected_visual, reference_raw
from src.models.apra_trem import APRATREM, b3_token_evidence


@torch.inference_mode()
def audit(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cached = torch.load(payload["source_cache"], map_location="cpu", weights_only=False)
    model = APRATREM(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    raw = cached["raw"].to(device)
    visual = projected_visual(model, cached["visual"], device)
    count = min(32, len(raw)); indices = torch.arange(count, device=device)
    ids = cached["reference_ids"].to(device)[indices]
    columns = cached["reference_columns"].to(device)[indices]
    expanded = columns[..., None]
    nq = model.numeric(raw[indices]); nr_full = model.numeric(raw[ids])
    nr = nr_full.gather(1, expanded.expand(-1, -1, nr_full.shape[-1]))
    vq = visual[indices].float(); vr_full = visual[ids].float()
    vr = vr_full.gather(1, expanded.expand(-1, -1, vr_full.shape[-1]))
    rq = raw[indices].reshape(count, 14, 16); rr = reference_raw(raw, ids, columns)
    base = b3_token_evidence(nq, vq, nr, vr, rq, rr)
    output = model.forward_tokens(nq, vq, nr, vr, base)
    residual = dict(output["alignment"])
    residual["structural_numeric"] = residual["structural_numeric"] + .1
    changed_residual = model.trem(residual, output["apra_delta"], output["b3_evidence"])
    reliability = dict(output["alignment"])
    reliability["reliability"] = 1.0 - reliability["reliability"]
    changed_reliability = model.trem(reliability, output["apra_delta"], output["b3_evidence"])
    capacity = (1.0 - output["apra_delta"].abs()).clamp_min(0.0)
    return {
        "variant": payload["variant"], "dataset": payload["dataset"],
        "audit_points": int(capacity.numel()),
        "capacity_mean": float(capacity.mean()),
        "zero_capacity_fraction": float((capacity == 0).float().mean()),
        "trem_raw_abs_mean": float(output["trem_raw"].abs().mean()),
        "trem_delta_abs_mean": float(output["trem_delta"].abs().mean()),
        "residual_raw_effect": float((changed_residual["trem_raw"] - output["trem_raw"]).abs().mean()),
        "reliability_raw_effect": float((changed_reliability["trem_raw"] - output["trem_raw"]).abs().mean()),
        "residual_bounded_effect": float((changed_residual["trem_delta"] - output["trem_delta"]).abs().mean()),
        "reliability_bounded_effect": float((changed_reliability["trem_delta"] - output["trem_delta"]).abs().mean()),
        "labels_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); device = torch.device(args.device)
    rows = []
    for path in sorted((args.output_root / "checkpoints").glob("*/*.pt")):
        row = audit(path, device); rows.append(row); print(row, flush=True)
    output = args.output_root / "trem_input_mechanism_audit.csv"
    temporary = output.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False); temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
