"""Architectures and unlabeled objectives for the AMR training ablations.

The frozen visual projection and the numeric encoder are identical to the
published AMR model.  Architecture changes are confined to the interaction
module.  Scoring always reads ``numeric`` tokens, so all variants share the
same inference representation and dimensionality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from src.models.bimodal_reference import NumericPatchEncoder, VisualColumnEncoder


ARCHITECTURES = frozenset(("cross", "no_cross", "late", "numeric"))


class AblationReferenceModel(nn.Module):
    def __init__(self, architecture: str, hidden: int = 192, heads: int = 4,
                 layers: int = 2):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.architecture = architecture
        self.numeric = NumericPatchEncoder(hidden=hidden, layers=layers, heads=heads)
        if architecture != "numeric":
            self.visual = VisualColumnEncoder(hidden=hidden)
        if architecture == "cross":
            self.n_to_v = nn.MultiheadAttention(hidden, heads, dropout=0.1, batch_first=True)
            self.v_to_n = nn.MultiheadAttention(hidden, heads, dropout=0.1, batch_first=True)
            self.fuse = nn.Sequential(nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
                                      nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden))
            # Retained for exact state compatibility with the immutable Full
            # checkpoints. It is not read by fixed-B3 scoring.
            self.log_temperature = nn.Parameter(torch.tensor(-2.3))
        elif architecture == "late":
            self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden * 2), nn.GELU(),
                                      nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden))
        elif architecture == "numeric":
            # Training-only masked Patch reconstruction head.  The head is not
            # used by B3 inference and adds no labels or visual targets.
            self.decoder = nn.Linear(hidden, 16)

    def forward(self, raw: torch.Tensor, visual: torch.Tensor | None = None):
        n = self.numeric(raw)
        output = {"numeric": n}
        if self.architecture == "numeric":
            output["reconstruction"] = self.decoder(n)
            return output
        if visual is None:
            raise ValueError(f"{self.architecture} requires frozen visual tokens")
        v = self.visual(visual)
        output["visual"] = v
        if self.architecture == "cross":
            nv = self.n_to_v(n, v, v, need_weights=False)[0]
            vn = self.v_to_n(v, n, n, need_weights=False)[0]
            output["fused"] = F.normalize(
                self.fuse(torch.cat((n, v, nv, vn), dim=-1)), dim=-1)
        elif self.architecture == "late":
            output["fused"] = F.normalize(self.fuse(torch.cat((n, v), dim=-1)), dim=-1)
        return output


@dataclass(frozen=True)
class AlignmentLossConfig:
    patch: bool = True
    temporal_relation: bool = True
    transition_geometry: bool = True
    reference: bool = True

    @property
    def active(self) -> bool:
        return any((self.patch, self.temporal_relation,
                    self.transition_geometry, self.reference))


def alignment_loss(outputs: Mapping[str, torch.Tensor], config: AlignmentLossConfig):
    """Return the four preregistered cross-modal consistency groups.

    Patch Consistency contains both exact numeric/visual Patch alignment and,
    when an interaction representation exists, the original fused-token Patch
    term.  This makes L1 and L5 remove every Patch-level alignment path.
    """
    n, v = outputs["numeric"], outputs["visual"]
    local = (1.0 - (n * v).sum(-1)).mean()
    relation = F.smooth_l1_loss(
        torch.einsum("bid,bjd->bij", n, n),
        torch.einsum("bid,bjd->bij", v, v),
    )
    dn = 1.0 - (n[:, 1:] * n[:, :-1]).sum(-1)
    dv = 1.0 - (v[:, 1:] * v[:, :-1]).sum(-1)
    transition = F.smooth_l1_loss(dn, dv)
    reference = torch.zeros((), device=n.device)
    if n.shape[0] > 1:
        ng, vg = F.normalize(n.mean(1), dim=-1), F.normalize(v.mean(1), dim=-1)
        reference = F.smooth_l1_loss(ng @ ng.T, vg @ vg.T)
    fused = torch.zeros((), device=n.device)
    if "fused" in outputs:
        fused = 1.0 - (outputs["fused"] * 0.5 * (n + v)).sum(-1).mean()
    groups = {
        "patch": local + 0.25 * fused,
        "temporal_relation": 0.5 * relation,
        "transition_geometry": 0.25 * transition,
        "reference": 0.5 * reference,
    }
    active = {
        "patch": config.patch,
        "temporal_relation": config.temporal_relation,
        "transition_geometry": config.transition_geometry,
        "reference": config.reference,
    }
    total = sum(value for name, value in groups.items() if active[name]) if config.active else None
    diagnostics = {
        "patch_local_unweighted": local,
        "patch_fused_unweighted": fused,
        "temporal_relation_unweighted": relation,
        "transition_geometry_unweighted": transition,
        "reference_unweighted": reference,
        **{f"weighted_{name}": value for name, value in groups.items()},
    }
    return total, groups, diagnostics


def masked_numeric_batch(raw: torch.Tensor, mask_fraction: float,
                         generator: torch.Generator | None = None):
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between zero and one")
    batch = raw.shape[0]
    random = torch.rand((batch, 14), device=raw.device, generator=generator)
    mask = random < mask_fraction
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty, random[empty].argmin(dim=1)] = True
    patch_mask = mask[:, :, None].expand(-1, -1, 16).reshape(batch, 224)
    masked = raw.masked_fill(patch_mask, 0.0)
    target = raw.reshape(batch, 14, 16)
    return masked, target, mask


def masked_reconstruction_loss(output: Mapping[str, torch.Tensor], target: torch.Tensor,
                               patch_mask: torch.Tensor):
    per_value = F.smooth_l1_loss(output["reconstruction"], target, reduction="none").mean(-1)
    return per_value[patch_mask].mean()


def parameter_group(name: str) -> str:
    if name.startswith("numeric."):
        return "numeric_transformer"
    if name.startswith(("n_to_v.", "v_to_n.")):
        return "cross_attention"
    if name.startswith("fuse."):
        return "fusion"
    if name.startswith("decoder."):
        return "reconstruction_head"
    return "other"


def gradient_norms(model: nn.Module) -> dict[str, float]:
    squares: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group(name)
        squares[group] = squares.get(group, 0.0) + float(parameter.grad.detach().square().sum())
    result = {name: value ** 0.5 for name, value in squares.items()}
    result["all"] = sum(squares.values()) ** 0.5
    for group in ("numeric_transformer", "cross_attention", "fusion",
                  "reconstruction_head", "other"):
        result.setdefault(group, 0.0)
    return result


def trainable_parameter_counts(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        counts[group] = counts.get(group, 0) + parameter.numel()
    counts["all"] = sum(counts.values())
    counts["inference_numeric"] = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith("numeric."))
    return counts

