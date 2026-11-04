"""APRA with temporal residual evolution modeling (TREM)."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from src.models.apra_acef import (
    HIDDEN, APRAEvidenceHead, AnomalyPreservingReferenceAlignment,
)
from src.models.bimodal_reference import NumericPatchEncoder, VisualColumnEncoder


def _difference(value: torch.Tensor) -> torch.Tensor:
    """Left anchored temporal difference without circular wraparound."""
    return torch.cat((torch.zeros_like(value[:, :1]), value[:, 1:] - value[:, :-1]), dim=1)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.pointwise(F.gelu(self.depthwise(value.transpose(1, 2)))).transpose(1, 2)
        return self.norm(value + update)


class TemporalResidualEvolutionModel(nn.Module):
    """Model local and multi-scale evolution of APRA residual evidence."""
    def __init__(self, hidden: int = HIDDEN, width: int = 128, *,
                 use_local: bool = True, use_multiscale: bool = True,
                 residual_limit: float = .25):
        super().__init__()
        if not (use_local or use_multiscale):
            raise ValueError("TREM requires at least one temporal branch")
        self.use_local = use_local
        self.use_multiscale = use_multiscale
        self.residual_limit = float(residual_limit)
        # The four inputs are APRA outputs. No independent query classifier is used.
        self.residual_projection = nn.Sequential(
            nn.Linear(hidden * 4, width), nn.GELU(), nn.LayerNorm(width),
        )
        if use_local:
            self.local = nn.Sequential(
                nn.Conv1d(width * 3, width, kernel_size=3, padding=1),
                nn.GELU(), nn.Conv1d(width, width, kernel_size=3, padding=1),
                nn.GELU(),
            )
        else:
            self.local = None
        if use_multiscale:
            self.context_input = nn.Sequential(
                nn.Linear(width + 2, width), nn.GELU(), nn.LayerNorm(width),
            )
            self.context = nn.ModuleList(
                [DilatedResidualBlock(width, dilation) for dilation in (1, 2, 4)]
            )
        else:
            self.context_input = None
            self.context = None
        branches = int(use_local) + int(use_multiscale)
        self.reliability_fusion = nn.Sequential(
            nn.Linear(4, width // 2), nn.GELU(), nn.Linear(width // 2, branches),
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, alignment: Mapping[str, torch.Tensor],
                apra_score: torch.Tensor, b3_evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = torch.cat((
            alignment["structural_numeric"], alignment["structural_visual"],
            alignment["unmatched_numeric"], alignment["unmatched_visual"],
        ), dim=-1)
        projected = self.residual_projection(residual)
        branch_values = []
        local_value = torch.zeros_like(projected)
        context_value = torch.zeros_like(projected)
        if self.local is not None:
            first = _difference(projected)
            second = _difference(first)
            local_value = self.local(torch.cat((projected, first, second), -1).transpose(1, 2)).transpose(1, 2)
            branch_values.append(local_value)
        if self.context is not None:
            context_value = self.context_input(torch.cat((
                projected, apra_score[..., None], b3_evidence[..., None],
            ), dim=-1))
            for block in self.context:
                context_value = block(context_value)
            branch_values.append(context_value)
        condition = torch.stack((
            alignment["reliability"], alignment["rejection"],
            apra_score.detach(), b3_evidence.detach(),
        ), dim=-1)
        weights = self.reliability_fusion(condition).softmax(-1)
        fused = sum(weights[..., index, None] * value
                    for index, value in enumerate(branch_values))
        raw = torch.tanh(self.output(self.output_norm(fused)).squeeze(-1))
        # Residual capacity shrinks near APRA's [-1, 1] boundary, proving that
        # the combined correction remains bounded while preserving APRA at zero.
        capacity = (1.0 - apra_score.abs()).clamp_min(0.0)
        delta = self.residual_limit * capacity * raw
        return {
            "trem_delta": delta,
            "trem_raw": raw,
            "local_dynamics": local_value,
            "multiscale_context": context_value,
            "fusion_weights": weights,
            "residual_projection": projected,
        }


class APRATREM(nn.Module):
    """Frozen architecture boundary: existing APRA plus a bounded TREM residual."""
    def __init__(self, *, use_local: bool = True, use_multiscale: bool = True,
                 hidden: int = HIDDEN, heads: int = 4, layers: int = 2,
                 trem_width: int = 128, residual_limit: float = .25):
        super().__init__()
        self.numeric = NumericPatchEncoder(hidden, layers, heads)
        self.visual = VisualColumnEncoder(hidden)
        self.aligner = AnomalyPreservingReferenceAlignment(hidden, heads, True)
        self.apra_head = APRAEvidenceHead(hidden, heads)
        self.trem = TemporalResidualEvolutionModel(
            hidden, trem_width, use_local=use_local,
            use_multiscale=use_multiscale, residual_limit=residual_limit,
        )

    def forward_tokens(self, nq: torch.Tensor, vq: torch.Tensor,
                       nr: torch.Tensor, vr: torch.Tensor,
                       b3_evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        alignment = self.aligner(nq, vq, nr, vr)
        apra = self.apra_head(nq, vq, alignment)["delta"]
        trem = self.trem(alignment, apra, b3_evidence)
        combined = apra + trem["trem_delta"]
        return {
            "alignment": alignment,
            "apra_delta": apra,
            "delta": combined,
            **trem,
            "numeric_query": nq,
            "visual_query": vq,
            "numeric_reference": nr,
            "visual_reference": vr,
            "b3_evidence": b3_evidence,
        }


def load_apra_only(model: APRATREM, checkpoint: str) -> None:
    """Load the immutable validated APRA-only model and leave TREM at zero."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    mapped = {}
    for name, value in state.items():
        if name.startswith("fusion."):
            mapped["apra_head." + name.removeprefix("fusion.")] = value
        elif name.startswith(("numeric.", "visual.", "aligner.")):
            mapped[name] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if unexpected or any(not name.startswith("trem.") for name in missing):
        raise RuntimeError(f"APRA-only initialization mismatch: missing={missing}, unexpected={unexpected}")


def b3_token_evidence(nq: torch.Tensor, vq: torch.Tensor,
                      nr: torch.Tensor, vr: torch.Tensor,
                      raw_query: torch.Tensor, raw_reference: torch.Tensor) -> torch.Tensor:
    """Frozen 0.90/0.05/0.05 token proxy for the three B3 evidence sources."""
    visual = 1.0 - F.cosine_similarity(vq, vr, dim=-1)
    numeric = 1.0 - F.cosine_similarity(nq, nr, dim=-1)
    level = (raw_query - raw_reference).abs().mean(-1)
    qdiff = raw_query[..., 1:] - raw_query[..., :-1]
    rdiff = raw_reference[..., 1:] - raw_reference[..., :-1]
    dynamics = (qdiff - rdiff).abs().mean(-1)
    raw = .5 * (torch.log1p(level) + torch.log1p(dynamics))
    return .90 * visual + .05 * raw + .05 * numeric


def parameter_group(name: str) -> str:
    if name.startswith("numeric."):
        return "numeric"
    if name.startswith("visual."):
        return "visual"
    if name.startswith("aligner."):
        return "apra"
    if name.startswith("apra_head."):
        return "apra_head"
    if name.startswith("trem."):
        return "trem"
    return "other"


def gradient_norms(model: nn.Module) -> dict[str, float]:
    squares: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group(name)
        squares[group] = squares.get(group, 0.0) + float(parameter.grad.square().sum())
    result = {key: value ** .5 for key, value in squares.items()}
    for key in ("numeric", "visual", "apra", "apra_head", "trem", "other"):
        result.setdefault(key, 0.0)
    result["all"] = sum(squares.values()) ** .5
    return result

