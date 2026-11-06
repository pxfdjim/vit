"""Final anomaly preserving reference alignment model.

This module contains the production E1 APRA path only.  The model keeps the
three-state differentiable monotone alignment, structural evidence head, and
the original L_path + L_detect training objective used by the immutable E1
checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from src.models.bimodal_reference import NumericPatchEncoder, VisualColumnEncoder


TOKENS = 14
HIDDEN = 192


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-8)


def _monotone_marginals(match: torch.Tensor, reject: torch.Tensor,
                        skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact Match, Query Reject, and Reference Skip path marginals."""
    batch, query_steps, reference_steps = match.shape
    neg = match.new_full((batch,), -1e4)
    alpha: list[list[torch.Tensor]] = [[neg for _ in range(reference_steps + 1)]
                                       for _ in range(query_steps + 1)]
    alpha[0][0] = match.new_zeros(batch)
    for i in range(query_steps + 1):
        for j in range(reference_steps + 1):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0 and j > 0:
                candidates.append(alpha[i - 1][j - 1] + match[:, i - 1, j - 1])
            if i > 0:
                candidates.append(alpha[i - 1][j] + reject[:, i - 1])
            if j > 0:
                candidates.append(alpha[i][j - 1] + skip[:, j - 1])
            alpha[i][j] = torch.logsumexp(torch.stack(candidates), dim=0)
    beta: list[list[torch.Tensor]] = [[neg for _ in range(reference_steps + 1)]
                                      for _ in range(query_steps + 1)]
    beta[query_steps][reference_steps] = match.new_zeros(batch)
    for i in range(query_steps, -1, -1):
        for j in range(reference_steps, -1, -1):
            if i == query_steps and j == reference_steps:
                continue
            candidates = []
            if i < query_steps and j < reference_steps:
                candidates.append(match[:, i, j] + beta[i + 1][j + 1])
            if i < query_steps:
                candidates.append(reject[:, i] + beta[i + 1][j])
            if j < reference_steps:
                candidates.append(skip[:, j] + beta[i][j + 1])
            beta[i][j] = torch.logsumexp(torch.stack(candidates), dim=0)
    log_z = alpha[query_steps][reference_steps]
    match_rows, reject_rows, skip_rows = [], [], []
    for i in range(query_steps):
        match_rows.append(torch.stack([
            (alpha[i][j] + match[:, i, j] + beta[i + 1][j + 1] - log_z).exp()
            for j in range(reference_steps)], dim=-1))
        reject_rows.append(torch.stack([
            (alpha[i][j] + reject[:, i] + beta[i + 1][j] - log_z).exp()
            for j in range(reference_steps + 1)], dim=-1).sum(-1))
    for j in range(reference_steps):
        skip_rows.append(torch.stack([
            (alpha[i][j] + skip[:, j] + beta[i][j + 1] - log_z).exp()
            for i in range(query_steps + 1)], dim=-1).sum(-1))
    return (torch.stack(match_rows, dim=1), torch.stack(reject_rows, dim=1),
            torch.stack(skip_rows, dim=1))


class AnomalyPreservingReferenceAlignment(nn.Module):
    """DTW-anchored monotone soft matching with two explicit gap states."""
    def __init__(self, hidden: int = HIDDEN, heads: int = 4):
        super().__init__()
        del heads
        self.query_projection = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                              nn.LayerNorm(hidden))
        self.reference_projection = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                                  nn.LayerNorm(hidden))
        self.query_structure = nn.Linear(hidden * 2, hidden, bias=False)
        self.reference_structure = nn.Linear(hidden * 2, hidden, bias=False)
        self.reject_head = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(),
                                         nn.Linear(hidden, 1))
        self.skip_head = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.GELU(),
                                       nn.Linear(hidden // 2, 1))
        self.distance_penalty = nn.Parameter(torch.tensor(0.75))
        self.reliability = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, nq: torch.Tensor, vq: torch.Tensor,
                nr: torch.Tensor, vr: torch.Tensor) -> dict[str, torch.Tensor]:
        q = F.normalize(self.query_projection(torch.cat((nq, vq), -1)), dim=-1)
        r = F.normalize(self.reference_projection(torch.cat((nr, vr), -1)), dim=-1)
        q_delta = torch.cat((nq - nq.roll(1, 1), vq - vq.roll(1, 1)), -1)
        r_delta = torch.cat((nr - nr.roll(1, 1), vr - vr.roll(1, 1)), -1)
        qs = F.normalize(self.query_structure(q_delta), dim=-1)
        rs = F.normalize(self.reference_structure(r_delta), dim=-1)
        match_logits = 3.0 * torch.einsum("bih,bjh->bij", q, r)
        match_logits = match_logits + torch.einsum("bih,bjh->bij", qs, rs)
        index = torch.arange(nq.shape[1], device=nq.device, dtype=nq.dtype)
        distance = (index[:, None] - index[None, :]).abs()
        match_logits = match_logits - F.softplus(self.distance_penalty) * distance.square() / 4.0
        match_logits = match_logits.masked_fill(distance[None] > 3, -1e4)
        reject_logits = self.reject_head(torch.cat((nq, vq, q_delta), -1)).squeeze(-1)
        skip_logits = self.skip_head(torch.cat((nr, vr), -1)).squeeze(-1) - 1.5
        soft_match, rejection, reference_skip = _monotone_marginals(
            match_logits, reject_logits, skip_logits)
        row_match = soft_match.sum(-1)
        normalized = _normalize_rows(soft_match)
        aligned_n = normalized @ nr
        aligned_v = normalized @ vr
        entropy = -(normalized.clamp_min(1e-8) * normalized.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / torch.log(torch.tensor(float(nq.shape[1]), device=nq.device))
        expected = (normalized * index[None, None]).sum(-1)
        local_smoothness = torch.cat((torch.ones_like(expected[:, :1]),
                                      (expected[:, 1:] - expected[:, :-1]).abs()), 1)
        coverage = 1.0 - reference_skip.mean(1, keepdim=True).expand_as(expected)
        reliability = torch.sigmoid(self.reliability(torch.stack((
            1.0 - entropy, row_match, coverage, torch.exp(-local_smoothness)), -1)).squeeze(-1))
        unmatched_n = nq - row_match[..., None] * aligned_n
        unmatched_v = vq - row_match[..., None] * aligned_v
        structural_n = (nq - nq.roll(1, 1)) - (aligned_n - aligned_n.roll(1, 1))
        structural_v = (vq - vq.roll(1, 1)) - (aligned_v - aligned_v.roll(1, 1))
        return {"soft_match": soft_match, "rejection": rejection,
                "reference_skip": reference_skip, "reliability": reliability,
                "aligned_numeric": aligned_n, "aligned_visual": aligned_v,
                "unmatched_numeric": unmatched_n, "unmatched_visual": unmatched_v,
                "structural_numeric": structural_n, "structural_visual": structural_v}


class APRAEvidenceHead(nn.Module):
    """Convert aligned, unmatched, and structural evidence into a signed delta."""
    def __init__(self, hidden: int = HIDDEN, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(hidden * 6 + 2, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=.1,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, nq: torch.Tensor, vq: torch.Tensor,
                alignment: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del nq, vq
        features = torch.cat((alignment["unmatched_numeric"], alignment["unmatched_visual"],
            alignment["structural_numeric"], alignment["structural_visual"],
            alignment["aligned_numeric"], alignment["aligned_visual"],
            alignment["rejection"][..., None], alignment["reliability"][..., None]), -1)
        delta = torch.tanh(self.out(self.temporal(F.gelu(self.input(features)))).squeeze(-1))
        return {"delta": delta}


class APRA(nn.Module):
    """Final E1 APRA network with a fixed production topology."""
    def __init__(self, hidden: int = HIDDEN, heads: int = 4, layers: int = 2):
        super().__init__()
        self.numeric = NumericPatchEncoder(hidden, layers, heads)
        self.visual = VisualColumnEncoder(hidden)
        self.aligner = AnomalyPreservingReferenceAlignment(hidden, heads)
        self.fusion = APRAEvidenceHead(hidden, heads)

    def forward_tokens(self, nq: torch.Tensor, vq: torch.Tensor,
                       nr: torch.Tensor, vr: torch.Tensor) -> dict[str, torch.Tensor]:
        alignment = self.aligner(nq, vq, nr, vr)
        fusion = self.fusion(nq, vq, alignment)
        return {"alignment": alignment, **fusion, "numeric_query": nq,
                "visual_query": vq, "numeric_reference": nr, "visual_reference": vr}


def load_apra_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[APRA, dict]:
    """Load either an original E1 checkpoint or a checkpoint from this module."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    legacy = payload.get("model_config", {})
    if legacy.get("aligner", "apra") != "apra" or legacy.get("fusion", "apra_head") != "apra_head":
        raise ValueError("checkpoint is not an E1 APRA model")
    if legacy.get("reject", True) is not True:
        raise ValueError("checkpoint does not contain the E1 three-state APRA topology")
    model = APRA(hidden=int(legacy.get("hidden", HIDDEN)), heads=int(legacy.get("heads", 4)),
                 layers=int(legacy.get("layers", 2))).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def load_numeric_from_full(model: APRA, checkpoint: str | Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("numeric."): value for key, value in payload["state_dict"].items()
             if key.startswith("numeric.")}
    model.numeric.load_state_dict(state, strict=True)


def reversible_warp(tokens: torch.Tensor, amplitudes: torch.Tensor | None = None):
    """Apply the original E1 known invertible monotone temporal interpolation."""
    batch, steps, _ = tokens.shape
    x = torch.arange(steps, device=tokens.device, dtype=tokens.dtype)
    if amplitudes is None:
        amplitudes = (torch.rand(batch, device=tokens.device, dtype=tokens.dtype) * 2 - 1) * 1.5
    coordinates = x[None] + amplitudes[:, None] * torch.sin(torch.pi * x / (steps - 1))[None]
    left = coordinates.floor().long().clamp(0, steps - 1)
    right = (left + 1).clamp(0, steps - 1)
    fraction = coordinates - left
    target = tokens.new_zeros(batch, steps, steps)
    target.scatter_add_(2, left[..., None], (1 - fraction)[..., None])
    target.scatter_add_(2, right[..., None], fraction[..., None])
    return target @ tokens, target


@dataclass(frozen=True)
class LossWeights:
    path: float = 1.0
    detect: float = 1.0
    detect_margin: float = .35
    gamma: float = .05


def path_loss(output: Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Original E1 path cross entropy, reject prior, and cycle loss."""
    match = _normalize_rows(output["alignment"]["soft_match"])
    cross_entropy = -(target * match.clamp_min(1e-8).log()).sum(-1).mean()
    reject = output["alignment"]["rejection"].mean()
    inverse = _normalize_rows(target.transpose(1, 2))
    identity = torch.eye(target.shape[1], device=target.device)[None]
    cycle = F.smooth_l1_loss(match @ inverse, identity.expand_as(match))
    return cross_entropy + .25 * reject + .25 * cycle


def detection_loss(clean, synthetic, warped, clean_base, synthetic_base,
                   altered: torch.Tensor, weights: LossWeights) -> torch.Tensor:
    """Original E1 hard-synthetic ranking and stability objective."""
    clean_score = clean_base.detach() + weights.gamma * clean["delta"]
    synthetic_score = synthetic_base.detach() + weights.gamma * synthetic["delta"]
    difficulty = (weights.detect_margin - (synthetic_base - clean_base).detach()).relu()
    hard_weight = 1.0 + 2.0 * difficulty / difficulty.mean().clamp_min(1e-6)
    ranking = (hard_weight[altered] * F.softplus(
        weights.detect_margin - (synthetic_score - clean_score)[altered])).mean()
    stable = F.smooth_l1_loss(synthetic_score[~altered], clean_score[~altered])
    warp_stable = F.smooth_l1_loss(clean_base.detach() + weights.gamma * warped["delta"], clean_score)
    return ranking + .5 * stable + .5 * warp_stable


def parameter_group(name: str) -> str:
    if name.startswith("numeric."):
        return "numeric"
    if name.startswith("visual."):
        return "visual"
    if name.startswith("aligner."):
        return "apra"
    if name.startswith("fusion."):
        return "scoring_head"
    return "other"


def gradient_norms(model: nn.Module) -> dict[str, float]:
    squares: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group(name)
        squares[group] = squares.get(group, 0.0) + float(parameter.grad.square().sum())
    result = {key: value ** .5 for key, value in squares.items()}
    for key in ("numeric", "visual", "apra", "scoring_head", "other"):
        result.setdefault(key, 0.0)
    result["all"] = sum(squares.values()) ** .5
    return result
