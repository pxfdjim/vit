"""Frozen APRA with an independently calibrated temporal residual branch."""
from __future__ import annotations

from typing import Mapping
import torch
from torch import nn
import torch.nn.functional as F

from src.models.apra_acef import HIDDEN, APRAEvidenceHead, AnomalyPreservingReferenceAlignment
from src.models.apra_trem import DilatedResidualBlock, _difference
from src.models.bimodal_reference import NumericPatchEncoder, VisualColumnEncoder


class IndependentTREM(nn.Module):
    def __init__(self, hidden=HIDDEN, width=128, *, use_local=True,
                 use_multiscale=True, use_reliability=True, input_mode="apra"):
        super().__init__()
        if not (use_local or use_multiscale):
            raise ValueError("at least one temporal branch is required")
        if input_mode not in {"apra", "numeric"}:
            raise ValueError(input_mode)
        self.use_local = bool(use_local)
        self.use_multiscale = bool(use_multiscale)
        self.use_reliability = bool(use_reliability)
        self.input_mode = input_mode
        self.residual_projection = nn.Sequential(
            nn.Linear(hidden * 4, width), nn.GELU(), nn.LayerNorm(width))
        self.local = (nn.Sequential(
            nn.Conv1d(width * 3, width, 3, padding=1), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1), nn.GELU())
            if use_local else None)
        self.context_input = (nn.Sequential(
            nn.Linear(width + (2 if input_mode == "apra" else 0), width),
            nn.GELU(), nn.LayerNorm(width)) if use_multiscale else None)
        self.context = (nn.ModuleList([DilatedResidualBlock(width, d) for d in (1, 2, 4)])
                        if use_multiscale else None)
        branches = int(use_local) + int(use_multiscale)
        self.reliability_fusion = (nn.Sequential(
            nn.Linear(4, width // 2), nn.GELU(), nn.Linear(width // 2, branches))
            if use_reliability else None)
        self.reliability_modulation = (nn.Sequential(nn.Linear(4, width), nn.Tanh())
                                       if use_reliability else None)
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, alignment: Mapping[str, torch.Tensor], apra_score: torch.Tensor,
                b3_evidence: torch.Tensor, nq: torch.Tensor, nr: torch.Tensor):
        if self.input_mode == "apra":
            source = torch.cat((alignment["structural_numeric"], alignment["structural_visual"],
                                alignment["unmatched_numeric"], alignment["unmatched_visual"]), -1)
        else:
            difference = nq - nr
            source = torch.cat((difference, difference.abs(), nq, nr), -1)
        projected = self.residual_projection(source)
        first = _difference(projected)
        second = _difference(first)
        local = torch.zeros_like(projected)
        context = torch.zeros_like(projected)
        branches = []
        if self.local is not None:
            local = self.local(torch.cat((projected, first, second), -1).transpose(1, 2)).transpose(1, 2)
            branches.append(local)
        if self.context is not None:
            context_source = (torch.cat((projected, apra_score[..., None], b3_evidence[..., None]), -1)
                              if self.input_mode == "apra" else projected)
            context = self.context_input(context_source)
            for block in self.context:
                context = block(context)
            branches.append(context)
        condition = torch.stack((alignment["reliability"], alignment["rejection"],
                                 apra_score.detach(), b3_evidence.detach()), -1)
        if self.reliability_fusion is None:
            weights = projected.new_full((*projected.shape[:2], len(branches)), 1.0 / len(branches))
            modulation = torch.ones_like(projected)
        else:
            weights = self.reliability_fusion(condition).softmax(-1)
            modulation = 1.0 + .25 * self.reliability_modulation(condition)
        fused = sum(weights[..., i, None] * value for i, value in enumerate(branches)) * modulation
        pre_tanh = self.output(self.output_norm(fused)).squeeze(-1)
        raw = torch.tanh(pre_tanh)
        return {"trem_raw": raw, "trem_delta": raw, "input_features": source,
                "residual_projection": projected, "first_difference": first,
                "second_difference": second, "local_dynamics": local,
                "multiscale_context": context, "fusion_weights": weights,
                "reliability_modulation": modulation, "fused_features": fused,
                "pre_tanh": pre_tanh}


class FrozenAPRAIndependentTREM(nn.Module):
    def __init__(self, *, hidden=HIDDEN, heads=4, layers=2, trem_width=128,
                 use_local=True, use_multiscale=True, use_reliability=True,
                 input_mode="apra"):
        super().__init__()
        self.numeric = NumericPatchEncoder(hidden, layers, heads)
        self.visual = VisualColumnEncoder(hidden)
        self.aligner = AnomalyPreservingReferenceAlignment(hidden, heads, True)
        self.apra_head = APRAEvidenceHead(hidden, heads)
        self.trem = IndependentTREM(hidden, trem_width, use_local=use_local,
            use_multiscale=use_multiscale, use_reliability=use_reliability,
            input_mode=input_mode)

    def freeze_base(self):
        for module in (self.numeric, self.visual, self.aligner, self.apra_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.numeric, self.visual, self.aligner, self.apra_head):
            module.eval()
        return self

    def fixed_outputs(self, nq, vq, nr, vr):
        with torch.no_grad():
            alignment = self.aligner(nq, vq, nr, vr)
            apra = self.apra_head(nq, vq, alignment)["delta"]
        return {k: v.detach() for k, v in alignment.items()}, apra.detach()

    def forward_tokens(self, nq, vq, nr, vr, b3_evidence):
        alignment, apra = self.fixed_outputs(nq, vq, nr, vr)
        trem = self.trem(alignment, apra, b3_evidence.detach(), nq.detach(), nr.detach())
        return {"alignment": alignment, "apra_delta": apra,
                "delta": apra + trem["trem_raw"], **trem,
                "numeric_query": nq, "numeric_reference": nr,
                "visual_query": vq, "visual_reference": vr,
                "b3_evidence": b3_evidence}


def load_apra_only(model: FrozenAPRAIndependentTREM, checkpoint: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mapped = {}
    for name, value in payload["state_dict"].items():
        if name.startswith("fusion."):
            mapped["apra_head." + name.removeprefix("fusion.")] = value
        elif name.startswith(("numeric.", "visual.", "aligner.")):
            mapped[name] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if unexpected or any(not name.startswith("trem.") for name in missing):
        raise RuntimeError((missing, unexpected))
    model.freeze_base()


def load_old_trem(model: FrozenAPRAIndependentTREM, checkpoint: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {name.removeprefix("trem."): value for name, value in payload["state_dict"].items()
             if name.startswith("trem.") and not name.startswith("trem.residual_limit")}
    missing, unexpected = model.trem.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError((missing, unexpected))


def stage_gradient_norms(model):
    groups = {"projection": model.trem.residual_projection,
              "local": model.trem.local, "multiscale_input": model.trem.context_input,
              "multiscale": model.trem.context, "reliability_fusion": model.trem.reliability_fusion,
              "reliability_modulation": model.trem.reliability_modulation,
              "output_norm": model.trem.output_norm, "output": model.trem.output}
    result = {}
    for name, module in groups.items():
        if module is None:
            result[name] = 0.0
            continue
        total = sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)
        result[name] = total ** .5
    return result


def robust_signed_calibration(point, *, mad_multiplier=1.4826, divisor=3., floor=.05, clip=1.):
    import numpy as np
    finite = np.asarray(point)[np.isfinite(point)]
    center = float(np.median(finite)) if len(finite) else 0.0
    mad = float(np.median(np.abs(finite - center))) if len(finite) else 0.0
    scale = max(mad_multiplier * mad, floor)
    calibrated = np.clip((np.asarray(point) - center) / (divisor * scale), -clip, clip)
    return calibrated, {"center": center, "mad": mad, "scale": scale}
