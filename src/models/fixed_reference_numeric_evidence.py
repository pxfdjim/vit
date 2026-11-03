"""Fixed-T3-path numeric evidence and label-free reference reliability.

All quantities are computed from the current unlabeled signal, the immutable
T3 candidate pool, and T3's exact constrained-DTW paths.  No anomaly labels or
model selection signals enter these functions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from src.models.numeric_temporal_matching import EPS, PathMatch, penalty_from_costs
from src.models.temporal_matching import _dtw_one


@dataclass(frozen=True)
class NumericEvidence:
    level: list[torch.Tensor]
    dynamic: list[torch.Tensor]
    combined: list[torch.Tensor]
    scales: dict[str, float]


def positive_median(values: list[torch.Tensor] | np.ndarray, default: float = 1.0) -> float:
    if isinstance(values, np.ndarray):
        flat = np.asarray(values, np.float64).reshape(-1)
    else:
        flat = torch.cat([item.detach().reshape(-1).cpu() for item in values]).numpy()
    flat = flat[np.isfinite(flat) & (flat > EPS)]
    return float(np.median(flat)) if flat.size else float(default)


def _component_scale(values: list[torch.Tensor]) -> float:
    return max(positive_median(values), EPS)


def _patch_statistics(patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered = patches - patches.mean(-1, keepdim=True)
    positions = torch.linspace(-0.5, 0.5, patches.shape[-1], dtype=patches.dtype)
    slope = (centered * positions).sum(-1) / positions.square().sum().clamp_min(EPS)
    spread = 1.4826 * (patches - patches.median(-1, keepdim=True).values).abs().median(-1).values
    endpoint = patches[..., -1] - patches[..., 0]
    return slope, spread, endpoint


@torch.inference_mode()
def build_numeric_evidence(standardized_windows: torch.Tensor, pool: Any) -> NumericEvidence:
    """Build level and local-dynamics candidate matrices.

    ``standardized_windows`` is fitted with the protocol-authorized unlabeled
    statistics before this function.  It is never normalized per Patch.
    """
    windows = standardized_windows.detach().cpu().float()
    if windows.ndim != 2 or windows.shape[1] != 224:
        raise ValueError("standardized_windows must have shape [windows,224]")
    patches = windows.reshape(len(windows), 14, 16)
    first = patches.diff(dim=-1)
    slope, spread, endpoint = _patch_statistics(patches)
    ids_all = torch.as_tensor(pool.window_ids, dtype=torch.long).cpu()
    valid_all = torch.as_tensor(pool.valid, dtype=torch.bool).cpu()
    raw: dict[str, list[torch.Tensor]] = {name: [] for name in
        ("level", "first_difference", "slope", "spread", "endpoint")}
    for query in range(len(windows)):
        refs = ids_all[query, valid_all[query]]
        q, r = patches[query], patches[refs]
        raw["level"].append((q[None, :, None] - r[:, None]).abs().median(-1).values)
        qd, rd = first[query], first[refs]
        raw["first_difference"].append((qd[None, :, None] - rd[:, None]).abs().median(-1).values)
        for name, values in (("slope", slope), ("spread", spread), ("endpoint", endpoint)):
            raw[name].append((values[query][None, :, None] - values[refs][:, None]).abs())
    scales = {name: _component_scale(items) for name, items in raw.items()}
    level = [torch.log1p(item / scales["level"]) for item in raw["level"]]
    dynamic = []
    combined = []
    for index in range(len(level)):
        components = [torch.log1p(raw[name][index] / scales[name]) for name in
                      ("first_difference", "slope", "spread", "endpoint")]
        dyn = torch.stack(components).mean(0)
        dynamic.append(dyn)
        combined.append(0.5 * (level[index] + dyn))
    if any(not torch.isfinite(item).all() for group in (level, dynamic, combined) for item in group):
        raise RuntimeError("non-finite numeric evidence")
    return NumericEvidence(level, dynamic, combined, scales)


def score_on_fixed_paths(costs: list[torch.Tensor], match: PathMatch) -> torch.Tensor:
    rows = []
    for query, matrices in enumerate(costs):
        selected = int(match.selected_pool_positions[query])
        matrix, mask = matrices[selected], match.path_masks[query]
        rows.append(torch.stack([matrix[column, mask[column]].mean() for column in range(14)]))
    return torch.stack(rows)


def align_point_score_to_t3(numeric: np.ndarray, t3: np.ndarray) -> tuple[np.ndarray, float]:
    numeric, t3 = np.asarray(numeric, np.float64), np.asarray(t3, np.float64)
    scale = positive_median(t3) / max(positive_median(numeric), EPS)
    scale = float(np.clip(scale, 1e-4, 1e4))
    return numeric * scale, scale


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(order), np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks


def rank_agreement(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = _rank(left), _rank(right)
    if len(left_rank) < 2:
        return 1.0
    rho = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return float(np.clip(0.5 * (rho + 1.0), 0.0, 1.0))


def margin_confidence(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, np.float64))
    if len(ordered) < 2:
        return 1.0
    gap = max(float(ordered[1] - ordered[0]), 0.0)
    denominator = max(float(np.median(np.abs(ordered))), EPS)
    return float(np.clip(gap / denominator, 0.0, 1.0))


def representation_reliability(visual: np.ndarray, numeric: np.ndarray, representation: np.ndarray) -> float:
    support = 1.0 - float(_rank(representation)[int(np.argmin(visual))]) / max(len(visual) - 1, 1)
    agreement = 0.5 * (rank_agreement(visual, representation) + rank_agreement(numeric, representation))
    return float(math.sqrt(max(support * agreement, 0.0)))


def statistical_reliability(visual: np.ndarray, numeric: np.ndarray) -> float:
    support = 1.0 - float(_rank(numeric)[int(np.argmin(visual))]) / max(len(visual) - 1, 1)
    terms = [rank_agreement(visual, numeric), support,
             margin_confidence(visual), margin_confidence(numeric)]
    product = float(np.prod(np.clip(terms, 0.0, 1.0)))
    return float(product ** 0.25)


@torch.inference_mode()
def candidate_reliability(
    visual_costs: list[torch.Tensor],
    numeric_costs: list[torch.Tensor],
    learned_numeric_costs: list[torch.Tensor],
    fused_costs: list[torch.Tensor],
    pool: Any,
    *,
    top_k: int = 5,
    penalty_scale: float = 0.1,
    late_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return E3-A/B/C gates while retaining the exact T3 visual paths."""
    visual_penalty = penalty_from_costs(visual_costs, penalty_scale)
    gates_stat, gates_cross, gates_late = [], [], []
    details = {name: [] for name in
               ("visual_numeric_agreement", "visual_margin", "numeric_margin",
                "cross_representation_reliability", "late_representation_reliability")}
    for query, visual_matrices in enumerate(visual_costs):
        candidate_values = {name: [] for name in ("visual", "numeric", "learned_numeric", "fused")}
        for candidate in range(len(visual_matrices)):
            objective, path = _dtw_one(visual_matrices[candidate], band=2,
                                       non_diagonal_penalty=visual_penalty)
            candidate_values["visual"].append(float(objective))
            for name, matrices in (("numeric", numeric_costs),
                                   ("learned_numeric", learned_numeric_costs),
                                   ("fused", fused_costs)):
                matrix = matrices[query][candidate]
                candidate_values[name].append(float(np.mean([float(matrix[i, j]) for i, j in path])))
        arrays = {name: np.asarray(values, np.float64) for name, values in candidate_values.items()}
        selected = np.argsort(arrays["visual"], kind="stable")[:top_k]
        visual = arrays["visual"][selected]
        numeric = arrays["numeric"][selected]
        cross = ((1.0 - late_weight) * arrays["visual"] + late_weight * arrays["fused"])[selected]
        late = ((1.0 - late_weight) * arrays["visual"] + late_weight * arrays["learned_numeric"])[selected]
        stat = statistical_reliability(visual, numeric)
        cross_rel = representation_reliability(visual, numeric, cross)
        late_rel = representation_reliability(visual, numeric, late)
        gates_stat.append(stat)
        gates_cross.append(math.sqrt(max(stat * cross_rel, 0.0)))
        gates_late.append(math.sqrt(max(stat * late_rel, 0.0)))
        details["visual_numeric_agreement"].append(rank_agreement(visual, numeric))
        details["visual_margin"].append(margin_confidence(visual))
        details["numeric_margin"].append(margin_confidence(numeric))
        details["cross_representation_reliability"].append(cross_rel)
        details["late_representation_reliability"].append(late_rel)
    converted = {name: np.asarray(values, np.float32) for name, values in details.items()}
    return (np.asarray(gates_stat, np.float32), np.asarray(gates_cross, np.float32),
            np.asarray(gates_late, np.float32), converted)


__all__ = [
    "NumericEvidence", "align_point_score_to_t3", "build_numeric_evidence",
    "candidate_reliability", "margin_confidence", "positive_median",
    "rank_agreement", "representation_reliability", "score_on_fixed_paths",
    "statistical_reliability",
]
