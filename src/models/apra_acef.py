"""Anomaly preserving reference alignment and conditional explanation fusion."""
from __future__ import annotations

from dataclasses import dataclass
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
    """Exact marginals of monotone match/query-reject/reference-skip paths."""
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
    """DTW-anchored monotone soft matching with an explicit query reject state."""
    def __init__(self, hidden: int = HIDDEN, heads: int = 4, reject_enabled: bool = True):
        super().__init__()
        del heads
        self.reject_enabled = reject_enabled
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
        if not self.reject_enabled:
            reject_logits = torch.full_like(reject_logits, -1e4)
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


class FixedDTWAlignment(nn.Module):
    def forward(self, nq, vq, nr, vr):
        batch, steps, _ = nq.shape
        matrix = torch.eye(steps, device=nq.device, dtype=nq.dtype)[None].expand(batch, -1, -1)
        zeros = nq.new_zeros(batch, steps)
        ones = nq.new_ones(batch, steps)
        return {"soft_match": matrix, "rejection": zeros, "reference_skip": zeros,
                "reliability": ones, "aligned_numeric": nr, "aligned_visual": vr,
                "unmatched_numeric": nq - nr, "unmatched_visual": vq - vr,
                "structural_numeric": (nq - nq.roll(1, 1)) - (nr - nr.roll(1, 1)),
                "structural_visual": (vq - vq.roll(1, 1)) - (vr - vr.roll(1, 1))}


class OrdinaryCrossAttentionAlignment(nn.Module):
    def __init__(self, hidden: int = HIDDEN, heads: int = 4):
        super().__init__()
        self.q = nn.Linear(hidden * 2, hidden)
        self.k = nn.Linear(hidden * 2, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=.1, batch_first=True)

    def forward(self, nq, vq, nr, vr):
        q = self.q(torch.cat((nq, vq), -1)); k = self.k(torch.cat((nr, vr), -1))
        index = torch.arange(nq.shape[1], device=nq.device)
        mask = (index[:, None] - index[None, :]).abs() > 3
        _, weights = self.attention(q, k, k, attn_mask=mask, need_weights=True,
                                    average_attn_weights=True)
        aligned_n, aligned_v = weights @ nr, weights @ vr
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        reliability = 1.0 - entropy / torch.log(torch.tensor(float(nq.shape[1]), device=nq.device))
        zeros = reliability.new_zeros(reliability.shape)
        return {"soft_match": weights, "rejection": zeros, "reference_skip": zeros,
                "reliability": reliability, "aligned_numeric": aligned_n,
                "aligned_visual": aligned_v, "unmatched_numeric": nq-aligned_n,
                "unmatched_visual": vq-aligned_v,
                "structural_numeric": (nq-nq.roll(1,1))-(aligned_n-aligned_n.roll(1,1)),
                "structural_visual": (vq-vq.roll(1,1))-(aligned_v-aligned_v.roll(1,1))}


class ConditionalAttention(nn.Module):
    def __init__(self, hidden=HIDDEN, heads=4):
        super().__init__(); self.heads = heads; self.width = hidden // heads
        self.q = nn.Linear(hidden, hidden); self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden); self.out = nn.Linear(hidden, hidden)

    def forward(self, query, context, prior):
        batch, steps, hidden = query.shape
        def heads(x):
            return x.view(batch, steps, self.heads, self.width).transpose(1, 2)
        q, k, v = heads(self.q(query)), heads(self.k(context)), heads(self.v(context))
        logits = q @ k.transpose(-1, -2) / self.width ** .5
        logits = logits + prior.clamp_min(1e-6).log()[:, None]
        attention = logits.softmax(-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, steps, hidden)
        return self.out(output), attention


class ACEF(nn.Module):
    """Bidirectional conditional explanation with shared/complementary evidence."""
    def __init__(self, hidden=HIDDEN, heads=4):
        super().__init__()
        self.visual_explains_numeric = ConditionalAttention(hidden, heads)
        self.numeric_explains_visual = ConditionalAttention(hidden, heads)
        self.predict_numeric = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.predict_visual = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.numeric_residual = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.visual_residual = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.complement = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU())
        self.input = nn.Sequential(nn.Linear(hidden * 8 + 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=.1,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, 1)
        self.correction = nn.Linear(hidden, 1)
        nn.init.zeros_(self.correction.weight); nn.init.zeros_(self.correction.bias)

    def forward(self, nq, vq, alignment):
        prior = _normalize_rows(alignment["soft_match"])
        n_context, n_attention = self.visual_explains_numeric(
            nq, alignment["aligned_visual"], prior)
        v_context, v_attention = self.numeric_explains_visual(
            vq, alignment["aligned_numeric"], prior)
        predicted_n = self.predict_numeric(n_context)
        predicted_v = self.predict_visual(v_context)
        rn = self.numeric_residual(nq - predicted_n)
        rv = self.visual_residual(vq - predicted_v)
        shared = self.shared(rn * rv)
        complement = self.complement(torch.cat((rn, rv, (rn-rv).abs()), -1))
        features = torch.cat((shared, complement,
            alignment["unmatched_numeric"], alignment["unmatched_visual"],
            alignment["structural_numeric"], alignment["structural_visual"],
            alignment["aligned_numeric"], alignment["aligned_visual"],
            alignment["rejection"][..., None], alignment["reliability"][..., None]), -1)
        hidden = self.temporal(self.input(features))
        return {"delta": torch.tanh(self.correction(hidden).squeeze(-1)),
                "predicted_numeric": predicted_n, "predicted_visual": predicted_v,
                "shared_evidence": shared, "complementary_evidence": complement,
                "numeric_explanation_attention": n_attention,
                "visual_explanation_attention": v_attention}


class APRAEvidenceHead(nn.Module):
    def __init__(self, hidden=HIDDEN, heads=4):
        super().__init__()
        self.input = nn.Linear(hidden * 6 + 2, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden*2, dropout=.1,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1); self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, nq, vq, alignment):
        features = torch.cat((alignment["unmatched_numeric"], alignment["unmatched_visual"],
            alignment["structural_numeric"], alignment["structural_visual"],
            alignment["aligned_numeric"], alignment["aligned_visual"],
            alignment["rejection"][...,None], alignment["reliability"][...,None]), -1)
        delta = torch.tanh(self.out(self.temporal(F.gelu(self.input(features)))).squeeze(-1))
        return {"delta": delta}


class GatedFusion(nn.Module):
    def __init__(self, hidden=HIDDEN, heads=4):
        super().__init__(); del heads
        self.n = nn.Linear(hidden, hidden); self.v = nn.Linear(hidden, hidden)
        self.gate = nn.Sequential(nn.Linear(hidden*2+2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.out = nn.Linear(hidden, 1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, nq, vq, alignment):
        n = F.gelu(self.n(nq-alignment["aligned_numeric"]))
        v = F.gelu(self.v(vq-alignment["aligned_visual"]))
        gate = torch.sigmoid(self.gate(torch.cat((n, v, alignment["rejection"][...,None],
                                                  alignment["reliability"][...,None]), -1)))
        return {"delta": torch.tanh(self.out(gate*n + (1-gate)*v).squeeze(-1)), "gate": gate}


class APRAACEF(nn.Module):
    def __init__(self, aligner="apra", fusion="acef", reject=True,
                 hidden=HIDDEN, heads=4, layers=2):
        super().__init__(); self.aligner_name = aligner; self.fusion_name = fusion
        self.numeric = NumericPatchEncoder(hidden, layers, heads)
        self.visual = VisualColumnEncoder(hidden)
        if aligner == "apra":
            self.aligner = AnomalyPreservingReferenceAlignment(hidden, heads, reject)
        elif aligner == "fixed_dtw": self.aligner = FixedDTWAlignment()
        elif aligner == "cross_attention": self.aligner = OrdinaryCrossAttentionAlignment(hidden, heads)
        else: raise ValueError(aligner)
        if fusion == "acef": self.fusion = ACEF(hidden, heads)
        elif fusion == "apra_head": self.fusion = APRAEvidenceHead(hidden, heads)
        elif fusion == "gated": self.fusion = GatedFusion(hidden, heads)
        else: raise ValueError(fusion)

    def forward_tokens(self, nq, vq, nr, vr):
        alignment = self.aligner(nq, vq, nr, vr)
        fusion = self.fusion(nq, vq, alignment)
        return {"alignment": alignment, **fusion, "numeric_query": nq,
                "visual_query": vq, "numeric_reference": nr, "visual_reference": vr}


def load_numeric_from_full(model: APRAACEF, checkpoint: str, *, random_numeric=False):
    if random_numeric:
        for parameter in model.numeric.parameters(): parameter.requires_grad_(False)
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k.removeprefix("numeric."): v for k,v in payload["state_dict"].items()
             if k.startswith("numeric.")}
    model.numeric.load_state_dict(state, strict=True)


def reversible_warp(tokens: torch.Tensor, amplitudes: torch.Tensor | None = None):
    """Apply a known invertible monotone temporal interpolation to token features."""
    batch, steps, _ = tokens.shape
    x = torch.arange(steps, device=tokens.device, dtype=tokens.dtype)
    if amplitudes is None:
        amplitudes = (torch.rand(batch, device=tokens.device, dtype=tokens.dtype) * 2 - 1) * 1.5
    coordinates = x[None] + amplitudes[:,None] * torch.sin(torch.pi*x/(steps-1))[None]
    left = coordinates.floor().long().clamp(0, steps-1); right = (left+1).clamp(0, steps-1)
    fraction = coordinates-left
    target = tokens.new_zeros(batch, steps, steps)
    target.scatter_add_(2, left[...,None], (1-fraction)[...,None])
    target.scatter_add_(2, right[...,None], fraction[...,None])
    return target @ tokens, target


@dataclass(frozen=True)
class LossWeights:
    path: float = 1.0
    explain: float = .5
    detect: float = 1.0
    detect_margin: float = .35
    gamma: float = .05


def path_loss(output: Mapping[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    match = _normalize_rows(output["alignment"]["soft_match"])
    cross_entropy = -(target * match.clamp_min(1e-8).log()).sum(-1).mean()
    reject = output["alignment"]["rejection"].mean()
    inverse = _normalize_rows(target.transpose(1,2))
    identity = torch.eye(target.shape[1], device=target.device)[None]
    cycle = F.smooth_l1_loss(match @ inverse, identity.expand_as(match))
    return cross_entropy + .25*reject + .25*cycle


def explanation_loss(output: Mapping[str, torch.Tensor], untouched: torch.Tensor) -> torch.Tensor:
    if "predicted_numeric" not in output:
        return output["delta"].sum() * 0
    trusted = untouched.bool() & (output["alignment"]["reliability"].detach() >= .25)
    if not trusted.any(): trusted = untouched.bool()
    return .5 * (F.smooth_l1_loss(output["predicted_numeric"][trusted],
                                  output["numeric_query"][trusted]) +
                 F.smooth_l1_loss(output["predicted_visual"][trusted],
                                  output["visual_query"][trusted]))


def detection_loss(clean, synthetic, warped, clean_base, synthetic_base,
                   altered: torch.Tensor, weights: LossWeights) -> torch.Tensor:
    clean_score = clean_base.detach() + weights.gamma*clean["delta"]
    synthetic_score = synthetic_base.detach() + weights.gamma*synthetic["delta"]
    difficulty = (weights.detect_margin - (synthetic_base-clean_base).detach()).relu()
    hard_weight = 1.0 + 2.0*difficulty/difficulty.mean().clamp_min(1e-6)
    ranking = (hard_weight[altered] * F.softplus(
        weights.detect_margin-(synthetic_score-clean_score)[altered])).mean()
    stable = F.smooth_l1_loss(synthetic_score[~altered], clean_score[~altered])
    warp_stable = F.smooth_l1_loss(clean_base.detach()+weights.gamma*warped["delta"], clean_score)
    return ranking + .5*stable + .5*warp_stable


def parameter_group(name: str) -> str:
    if name.startswith("numeric."): return "numeric"
    if name.startswith("visual."): return "visual"
    if name.startswith("aligner."): return "apra"
    if name.startswith("fusion."): return "acef"
    return "other"


def gradient_norms(model: nn.Module) -> dict[str, float]:
    squares = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None: continue
        group = parameter_group(name)
        squares[group] = squares.get(group, 0.0) + float(parameter.grad.square().sum())
    result = {key: value**.5 for key,value in squares.items()}
    for key in ("numeric","visual","apra","acef","other"): result.setdefault(key,0.0)
    result["all"] = sum(squares.values())**.5
    return result
