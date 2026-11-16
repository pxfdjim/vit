"""Reference-prior three-state alignment with explicit anomaly evidence."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F



class NumericPatchEncoder(nn.Module):
    """Encode sequence-normalized numeric windows as fourteen time tokens."""

    def __init__(self, hidden: int = 192, layers: int = 2, heads: int = 4):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 96, 3, padding=1), nn.GELU(),
        )
        self.patch = nn.Linear(96 * 16, hidden)
        self.position = nn.Parameter(torch.zeros(1, 14, hidden))
        block = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 3, 0.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = F.pad(values[:, 1:] - values[:, :-1], (1, 0))
        second = F.pad(first[:, 1:] - first[:, :-1], (1, 0))
        trend = F.avg_pool1d(values[:, None], 17, stride=1, padding=8)[:, 0]
        encoded = self.local(torch.stack((values, first, second, trend), dim=1))
        encoded = encoded.transpose(1, 2).reshape(values.shape[0], 14, 16 * 96)
        encoded = self.patch(encoded) + self.position
        return F.normalize(self.norm(self.temporal(encoded)), dim=-1)


class VisualColumnEncoder(nn.Module):
    """Apply the fixed projection used by every validated model checkpoint."""

    def __init__(self, hidden: int = 192):
        super().__init__()
        generator = torch.Generator().manual_seed(20261029)
        projection = torch.randn(14 * 768, hidden, generator=generator)
        self.register_buffer("projection", projection / (14 * 768) ** 0.5)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        grid = visual.reshape(-1, 14, 14, 768).permute(0, 2, 1, 3)
        encoded = grid.reshape(grid.shape[0], 14, 14 * 768) @ self.projection
        return F.normalize(F.layer_norm(encoded, (encoded.shape[-1],)), dim=-1)


HIDDEN = 192


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


def _first_difference(tokens: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(tokens[:, :1]), tokens[:, 1:] - tokens[:, :-1]), dim=1)


class AnomalyPreservingReferenceAlignment(nn.Module):
    """Match original reference columns under a frozen DTW position prior."""

    def __init__(self):
        super().__init__()
        self.anchor_cost = nn.Parameter(torch.zeros(()))
        # Query Reject and Reference Skip have equal total mass on equal-length
        # endpoint-constrained paths; one shared cost avoids two unidentifiable biases.
        self.gap_cost = nn.Parameter(torch.ones(()))

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
        match_mass = soft_match.sum(-1)
        conditional = soft_match / match_mass[..., None].clamp_min(1e-8)

        def view_evidence(query: torch.Tensor, reference: torch.Tensor,
                          similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                             torch.Tensor]:
            pair_difference = (1 - similarity) / 2
            matched_difference = (soft_match * pair_difference).sum(-1)
            # A rejected query still carries content. Compare it with the
            # whole reference window without assigning it a forced match.
            unmatched_difference = rejection * pair_difference.mean(-1)
            query_change = _first_difference(query)
            reference_change = _first_difference(reference)
            squared_change = (query_change.square().sum(-1)[..., None]
                              + reference_change.square().sum(-1)[:, None]
                              - 2 * torch.einsum("bih,bjh->bij", query_change,
                                                 reference_change)).clamp_min(0)
            matched_structure = (soft_match * squared_change / 16).sum(-1)
            unmatched_structure = rejection * query_change.square().sum(-1) / 4
            structure = matched_structure + unmatched_structure
            return matched_difference, unmatched_difference, structure

        numeric_match, numeric_unmatched, numeric_structure = view_evidence(
            nq, nr, numeric_similarity)
        visual_match, visual_unmatched, visual_structure = view_evidence(
            vq, vr, visual_similarity)
        evidence = torch.stack((numeric_match, visual_match,
                                numeric_unmatched, visual_unmatched,
                                numeric_structure, visual_structure), dim=-1)
        return {"soft_match": soft_match, "rejection": rejection,
                "reference_skip": reference_skip, "reliability": match_mass,
                "aligned_numeric": conditional @ nr,
                "aligned_visual": conditional @ vr,
                "evidence": evidence}


class EvidenceHead(nn.Module):
    """Bounded signed correction from six posterior anomaly statistics."""
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(6, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.out(evidence).squeeze(-1))


class Model(nn.Module):
    """T3 prior, three-state posterior, and explicit six-statistic readout."""
    def __init__(self, hidden: int = HIDDEN, heads: int = 4, layers: int = 2):
        super().__init__()
        self.numeric = NumericPatchEncoder(hidden, layers, heads)
        self.visual = VisualColumnEncoder(hidden)
        self.aligner = AnomalyPreservingReferenceAlignment()
        self.fusion = EvidenceHead()

    def forward_tokens(self, nq: torch.Tensor, vq: torch.Tensor,
                       nr: torch.Tensor, vr: torch.Tensor,
                       anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        alignment = self.aligner(nq, vq, nr, vr, anchor)
        return {"alignment": alignment, "delta": self.fusion(alignment["evidence"])}


def load_model_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[Model, dict]:
    """Strictly load the redesigned topology; old models remain in Git history."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    configuration = payload.get("model_config", {})
    if configuration.get("architecture") != 2:
        raise ValueError("checkpoint belongs to the previous architecture; retrain the alignment head")
    model = Model(hidden=int(configuration.get("hidden", HIDDEN)),
                  heads=int(configuration.get("heads", 4)),
                  layers=int(configuration.get("layers", 2))).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def load_numeric_from_full(model: Model, checkpoint: str | Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("numeric."): value for key, value in payload["state_dict"].items()
             if key.startswith("numeric.")}
    model.numeric.load_state_dict(state, strict=True)


def reversible_warp(tokens: torch.Tensor, amplitudes: torch.Tensor | None = None):
    """Apply the validated invertible monotone temporal interpolation."""
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


def parameter_group(name: str) -> str:
    if name.startswith("numeric."):
        return "numeric"
    if name.startswith("visual."):
        return "visual"
    if name.startswith("aligner."):
        return "alignment"
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
    for key in ("numeric", "visual", "alignment", "scoring_head", "other"):
        result.setdefault(key, 0.0)
    result["all"] = sum(squares.values()) ** .5
    return result
