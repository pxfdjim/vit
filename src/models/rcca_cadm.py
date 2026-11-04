"""Reference conditioned cross-attention and anomaly-difference modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from src.models.bimodal_reference import NumericPatchEncoder, VisualColumnEncoder


ARCHITECTURES = frozenset(("rcca", "no_rcca", "same_window"))


def gather_reference(tokens: torch.Tensor, ids: torch.Tensor,
                     columns: torch.Tensor) -> torch.Tensor:
    """Gather one DTW-aligned reference token for each query time position."""
    if ids.ndim != 1 or columns.ndim != 2 or columns.shape != (len(ids), 14):
        raise ValueError("reference ids/columns must be [B] and [B,14]")
    selected = tokens[ids]
    return selected.gather(1, columns[..., None].expand(-1, -1, tokens.shape[-1]))


class ReferenceConditionedCrossAttention(nn.Module):
    def __init__(self, hidden: int = 192, heads: int = 4,
                 mode: str = "rcca") -> None:
        super().__init__()
        if mode not in {"rcca", "same_window"}:
            raise ValueError(mode)
        self.mode = mode
        self.position = nn.Parameter(torch.zeros(1, 14, hidden))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.n_to_v = nn.MultiheadAttention(hidden, heads, dropout=0.1,
                                             batch_first=True)
        self.v_to_n = nn.MultiheadAttention(hidden, heads, dropout=0.1,
                                             batch_first=True)
        self.n_query_norm = nn.LayerNorm(hidden)
        self.v_query_norm = nn.LayerNorm(hidden)
        self.n_residual_norm = nn.LayerNorm(hidden)
        self.v_residual_norm = nn.LayerNorm(hidden)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
        )
        self.output_norm = nn.LayerNorm(hidden)

    def _condition(self, n_query: torch.Tensor, v_query: torch.Tensor,
                   n_context: torch.Tensor, v_context: torch.Tensor):
        position = self.position
        nq = self.n_query_norm(n_query + position)
        vq = self.v_query_norm(v_query + position)
        nk = self.n_query_norm(n_context + position)
        vk = self.v_query_norm(v_context + position)
        nv, n_weights = self.n_to_v(nq, vk, v_context, need_weights=True,
                                     average_attn_weights=False)
        vn, v_weights = self.v_to_n(vq, nk, n_context, need_weights=True,
                                     average_attn_weights=False)
        nv = self.n_residual_norm(n_query + nv)
        vn = self.v_residual_norm(v_query + vn)
        fused = self.fuse(torch.cat((n_query, v_query, nv, vn), dim=-1))
        fused = F.normalize(self.output_norm(fused + 0.5 * (n_query + v_query)), dim=-1)
        return fused, n_weights, v_weights

    def forward(self, n_query: torch.Tensor, v_query: torch.Tensor,
                n_reference: torch.Tensor, v_reference: torch.Tensor):
        if self.mode == "rcca":
            q_context_n, q_context_v = n_reference, v_reference
        else:
            q_context_n, q_context_v = n_query, v_query
        query, q_nv, q_vn = self._condition(
            n_query, v_query, q_context_n, q_context_v)
        reference, r_nv, r_vn = self._condition(
            n_reference, v_reference, n_reference, v_reference)
        return {
            "query": query, "reference": reference,
            "query_n_to_v_attention": q_nv,
            "query_v_to_n_attention": q_vn,
            "reference_n_to_v_attention": r_nv,
            "reference_v_to_n_attention": r_vn,
        }


class CrossModalAnomalyDifference(nn.Module):
    """Model local query/reference differences and their temporal context."""
    def __init__(self, hidden: int = 192, heads: int = 4) -> None:
        super().__init__()
        # signed visual diff, signed numeric diff, conditional diff and two
        # cross-modal reference matching products: 5 * hidden.
        self.input = nn.Sequential(nn.Linear(hidden * 5, hidden), nn.GELU(),
                                   nn.LayerNorm(hidden))
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout=0.1, batch_first=True,
            norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
                                  nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, n_query: torch.Tensor, v_query: torch.Tensor,
                n_reference: torch.Tensor, v_reference: torch.Tensor,
                conditional_difference: torch.Tensor) -> torch.Tensor:
        features = torch.cat((
            v_query - v_reference,
            n_query - n_reference,
            conditional_difference,
            n_query * v_reference,
            v_query * n_reference,
        ), dim=-1)
        return self.head(self.temporal(self.input(features))).squeeze(-1)


class RCCACADM(nn.Module):
    def __init__(self, architecture: str = "rcca", hidden: int = 192,
                 heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        self.architecture = architecture
        self.numeric = NumericPatchEncoder(hidden, layers, heads)
        self.visual = VisualColumnEncoder(hidden)
        if architecture != "no_rcca":
            self.rcca = ReferenceConditionedCrossAttention(
                hidden, heads, mode=architecture)
        self.cadm = CrossModalAnomalyDifference(hidden, heads)

    def encode(self, raw: torch.Tensor, visual: torch.Tensor):
        return self.numeric(raw), self.visual(visual)

    def forward_tokens(self, n_query: torch.Tensor, v_query: torch.Tensor,
                       n_reference: torch.Tensor, v_reference: torch.Tensor):
        if self.architecture == "no_rcca":
            conditional = torch.zeros_like(n_query)
            attention = {}
        else:
            conditioned = self.rcca(n_query, v_query, n_reference, v_reference)
            conditional = conditioned["query"] - conditioned["reference"]
            attention = {name: value for name, value in conditioned.items()
                         if name.endswith("attention")}
        logits = self.cadm(n_query, v_query, n_reference, v_reference, conditional)
        return {"logits": logits, "conditional_difference": conditional,
                "attention": attention}

    def forward(self, query_raw: torch.Tensor, reference_raw: torch.Tensor,
                query_visual: torch.Tensor, reference_visual: torch.Tensor,
                reference_columns: torch.Tensor):
        n_query, v_query = self.encode(query_raw, query_visual)
        n_reference_full, v_reference_full = self.encode(reference_raw, reference_visual)
        columns = reference_columns[..., None]
        n_reference = n_reference_full.gather(1, columns.expand(-1, -1, n_query.shape[-1]))
        v_reference = v_reference_full.gather(1, columns.expand(-1, -1, v_query.shape[-1]))
        result = self.forward_tokens(n_query, v_query, n_reference, v_reference)
        result.update({"numeric_query": n_query, "visual_query": v_query,
                       "numeric_reference": n_reference, "visual_reference": v_reference})
        return result


def load_numeric_from_full(model: RCCACADM, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    numeric = {name.removeprefix("numeric."): value for name, value in state.items()
               if name.startswith("numeric.")}
    missing, unexpected = model.numeric.load_state_dict(numeric, strict=True)
    if missing or unexpected:
        raise RuntimeError((missing, unexpected))


@dataclass(frozen=True)
class LossWeights:
    patch: float = 1.0
    synthetic: float = 1.0
    ranking: float = 0.5
    consistency: float = 0.25
    ranking_margin: float = 0.5


def reference_conditioned_loss(
    clean: Mapping[str, torch.Tensor], synthetic: Mapping[str, torch.Tensor],
    patch_mask: torch.Tensor, weights: LossWeights,
    *, use_synthetic: bool = True, use_ranking: bool = True,
):
    mask = patch_mask.bool()
    if mask.shape != clean["logits"].shape or not mask.any() or not (~mask).any():
        raise ValueError("each loss batch needs altered and untouched Patches")
    patch = torch.stack([
        1.0 - (clean["numeric_query"] * clean["visual_query"]).sum(-1).mean(),
        1.0 - (clean["numeric_reference"] * clean["visual_reference"]).sum(-1).mean(),
        1.0 - (synthetic["numeric_query"] * synthetic["visual_query"]).sum(-1).mean(),
    ]).mean()
    positive = mask.sum().float()
    negative = (~mask).sum().float()
    pos_weight = (negative / positive).detach().clamp(0.25, 4.0)
    synthetic_loss = F.binary_cross_entropy_with_logits(
        synthetic["logits"], mask.float(), pos_weight=pos_weight)
    ranking = F.relu(weights.ranking_margin -
                     (synthetic["logits"][mask] - clean["logits"][mask])).mean()
    consistency = F.smooth_l1_loss(
        synthetic["logits"][~mask], clean["logits"][~mask])
    total = weights.patch * patch + weights.consistency * consistency
    if use_synthetic:
        total = total + weights.synthetic * synthetic_loss
    if use_ranking:
        total = total + weights.ranking * ranking
    return total, {"patch": patch, "synthetic": synthetic_loss,
                   "ranking": ranking, "consistency": consistency}


def parameter_group(name: str) -> str:
    if name.startswith("numeric."):
        return "numeric"
    if name.startswith("rcca."):
        return "rcca"
    if name.startswith("cadm."):
        return "cadm"
    return "other"


def gradient_norms(model: nn.Module) -> dict[str, float]:
    squares: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group(name)
        squares[group] = squares.get(group, 0.0) + float(parameter.grad.square().sum())
    result = {name: value ** 0.5 for name, value in squares.items()}
    for name in ("numeric", "rcca", "cadm", "other"):
        result.setdefault(name, 0.0)
    result["all"] = sum(squares.values()) ** 0.5
    return result

