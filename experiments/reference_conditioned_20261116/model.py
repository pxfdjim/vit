"""Posterior-conditioned, two-view residual representation experiment.

This file lives in the ignored experiment tree; the released Model is untouched.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from src.models.model import (AnomalyPreservingReferenceAlignment,
                              NumericPatchEncoder, VisualColumnEncoder,
                              _monotone_marginals)


class PosteriorOnlyAlignment(AnomalyPreservingReferenceAlignment):
    """The same three-state posterior without the obsolete six-score statistics."""

    def forward(self, nq: torch.Tensor, vq: torch.Tensor,
                nr: torch.Tensor, vr: torch.Tensor,
                anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        if nq.shape != vq.shape or nr.shape != vr.shape or nq.ndim != 3:
            raise ValueError("query/reference views must be matching [batch,time,hidden] tensors")
        if nq.shape[0] != nr.shape[0] or nq.shape[2] != nr.shape[2]:
            raise ValueError("query and reference token shapes are incompatible")
        if anchor.shape != nq.shape[:2] or not torch.isfinite(anchor).all():
            raise ValueError("DTW anchor must have finite shape [batch,query_time]")
        if torch.any(anchor < 0) or torch.any(anchor > nr.shape[1] - 1):
            raise ValueError("DTW anchor is outside the reference window")
        nq, vq = F.normalize(nq, dim=-1), F.normalize(vq, dim=-1)
        nr, vr = F.normalize(nr, dim=-1), F.normalize(vr, dim=-1)
        numeric_similarity = torch.einsum("bih,bjh->bij", nq, nr).clamp(-1, 1)
        visual_similarity = torch.einsum("bih,bjh->bij", vq, vr).clamp(-1, 1)
        columns = torch.arange(nr.shape[1], dtype=nq.dtype, device=nq.device)
        anchor_distance = (columns[None, None] - anchor[..., None]).square()
        match_logits = (numeric_similarity + visual_similarity) / 2 - \
            F.softplus(self.anchor_cost) * anchor_distance
        gap_logit = -F.softplus(self.gap_cost)
        rejection_logits = gap_logit.expand(nq.shape[:2])
        skip_logits = gap_logit.expand(nr.shape[:2])
        soft_match, rejection, reference_skip = _monotone_marginals(
            match_logits, rejection_logits, skip_logits)
        return {"soft_match": soft_match, "rejection": rejection,
                "reference_skip": reference_skip,
                "reliability": soft_match.sum(-1)}


class ReferenceConditionedRepresentation(nn.Module):
    def __init__(self, mode: str, token_dim: int = 192, dim: int = 64):
        super().__init__()
        if mode not in {"posterior", "no_reference", "t3_direct"}:
            raise ValueError(mode)
        self.mode = mode
        self.numeric_projection = nn.Linear(token_dim, dim)
        self.visual_projection = nn.Linear(token_dim, dim)
        self.relation = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(),
                                      nn.Linear(dim, dim), nn.GELU())
        self.joint = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(),
                                   nn.Linear(dim, dim), nn.GELU())
        self.out = nn.Linear(dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, nq: torch.Tensor, vq: torch.Tensor,
                nr: torch.Tensor, vr: torch.Tensor,
                posterior: torch.Tensor, rejection: torch.Tensor,
                prior: torch.Tensor, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        p = posterior.sum(-1)
        u = rejection
        if self.mode == "no_reference":
            p = torch.ones_like(p)
            u = torch.zeros_like(u)
        columns = anchor.round().long().clamp(0, nr.shape[1] - 1)

        def view(query: torch.Tensor, reference: torch.Tensor, projection: nn.Linear):
            q = projection(query)
            r = projection(reference)
            if self.mode == "no_reference":
                matched = torch.zeros_like(q)
                prior_reference = matched
            else:
                matched = (posterior @ r) / p[..., None].clamp_min(1e-8)
                prior_reference = prior @ r
                if self.mode == "t3_direct":
                    matched = torch.gather(r, 1, columns[..., None].expand(-1, -1, r.shape[-1]))

            def relation(a: torch.Tensor, b: torch.Tensor):
                return self.relation(torch.cat((a, b, a - b, (a - b).abs()), dim=-1))

            h_match = relation(q, matched)
            h_unmatch = relation(q, prior_reference)
            h = p[..., None] * h_match + u[..., None] * h_unmatch
            return h, matched, prior_reference, h_match, h_unmatch

        hn, rn, pn, hnm, hnu = view(nq, nr, self.numeric_projection)
        hv, rv, pv, hvm, hvu = view(vq, vr, self.visual_projection)
        joint = self.joint(torch.cat((hn, hv, (hn - hv).abs()), dim=-1))
        delta = torch.tanh(self.out(joint).squeeze(-1))
        return {"delta": delta, "joint": joint, "numeric": hn, "visual": hv,
                "numeric_match": hnm, "numeric_unmatch": hnu,
                "visual_match": hvm, "visual_unmatch": hvu,
                "numeric_reference_match": rn, "numeric_reference_prior": pn,
                "visual_reference_match": rv, "visual_reference_prior": pv,
                "match_confidence": p, "rejection": u}


class Model(nn.Module):
    def __init__(self, mode: str, hidden: int = 192, heads: int = 4, layers: int = 2):
        super().__init__()
        self.numeric = NumericPatchEncoder(hidden, layers, heads)
        self.visual = VisualColumnEncoder(hidden)
        self.aligner = PosteriorOnlyAlignment()
        self.fusion = ReferenceConditionedRepresentation(mode, hidden)

    def forward_tokens(self, nq: torch.Tensor, vq: torch.Tensor,
                       nr: torch.Tensor, vr: torch.Tensor,
                       anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        alignment = self.aligner(nq, vq, nr, vr, anchor)
        columns = torch.arange(nr.shape[1], dtype=anchor.dtype, device=anchor.device)
        cost = (columns[None, None] - anchor[..., None]).square()
        prior = torch.softmax(-F.softplus(self.aligner.anchor_cost) * cost, dim=-1)
        representation = self.fusion(nq, vq, nr, vr, alignment["soft_match"],
                                     alignment["rejection"], prior, anchor)
        return {"alignment": alignment, "prior": prior,
                "representation": representation, "delta": representation["delta"]}


def load_model_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["model_config"]
    if config.get("architecture") != "reference_conditioned_v1":
        raise ValueError("wrong experiment checkpoint architecture")
    model = Model(config["mode"], config["hidden"], config["heads"], config["layers"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload
