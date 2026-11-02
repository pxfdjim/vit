"""Label-free numeric Patch matching controls for the validated T3 detector.

The module keeps T3's candidate windows and 14-column constrained-DTW
protocol.  It only adds a robust numeric cost for the exact 16 samples that
belong to each visual time column.  No per-Patch normalization is used: level,
amplitude and slope remain observable after sequence-level detrending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.signal import detrend

from src.models.temporal_matching import (
    GRID_SIZE,
    _dtw_one,
    column_cost_matrix,
    constrained_dtw_match,
)

WINDOW_SIZE = 224
STEP_SIZE = 56
PATCH_POINTS = 16
EPS = 1e-8


@dataclass(frozen=True)
class NumericPatchFeatures:
    centered: torch.Tensor       # [W,14,16], sequence-scale units
    differences: torch.Tensor    # [W,14,15], sequence-scale units
    behavior: torch.Tensor       # [W,14,6]
    sequence_center: float
    sequence_scale: float


@dataclass(frozen=True)
class CostBundle:
    visual: list[torch.Tensor]
    numeric: list[torch.Tensor]
    numeric_raw_components: dict[str, list[torch.Tensor]]
    component_scales: dict[str, float]
    numeric_to_visual_scale: float


@dataclass(frozen=True)
class PathMatch:
    scores: torch.Tensor
    reference_ids: torch.Tensor
    reference_columns: torch.Tensor
    path_masks: torch.Tensor
    normalized_path_costs: torch.Tensor
    path_lengths: torch.Tensor
    selected_pool_positions: torch.Tensor
    penalty: float


def _positive_median(values: list[torch.Tensor], default: float = 1.0) -> float:
    flat = torch.cat([value.detach().reshape(-1).to("cpu", torch.float32) for value in values])
    flat = flat[torch.isfinite(flat) & (flat > EPS)]
    return float(flat.median()) if flat.numel() else float(default)


def numeric_patch_features(raw: np.ndarray) -> NumericPatchFeatures:
    """Create 14x16 numeric tokens with a robust *sequence-level* scale."""

    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or values.size < WINDOW_SIZE or not np.isfinite(values).all():
        raise ValueError("raw must be one finite sequence of at least 224 points")
    residual = np.asarray(detrend(values), dtype=np.float64)
    center = float(np.median(residual))
    mad = float(1.4826 * np.median(np.abs(residual - center)))
    q25, q75 = np.quantile(residual, [0.25, 0.75])
    iqr = float((q75 - q25) / 1.349)
    std = float(np.std(residual))
    scale = max(mad, iqr, std * 0.1, EPS)
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    standardized = torch.from_numpy(((residual - center) / scale).astype(np.float32))
    windows = standardized.unfold(0, WINDOW_SIZE, STEP_SIZE).contiguous()
    patches = windows.reshape(windows.shape[0], GRID_SIZE, PATCH_POINTS)
    medians = patches.median(dim=-1, keepdim=True).values
    centered = patches - medians
    differences = patches.diff(dim=-1)
    q10 = torch.quantile(patches, 0.10, dim=-1)
    q90 = torch.quantile(patches, 0.90, dim=-1)
    patch_mad = 1.4826 * (patches - medians).abs().median(dim=-1).values
    patch_std = patches.std(dim=-1, unbiased=False)
    positions = torch.linspace(-0.5, 0.5, PATCH_POINTS, dtype=patches.dtype)
    slope = (centered * positions).sum(dim=-1) / positions.square().sum().clamp_min(EPS)
    endpoint = patches[..., -1] - patches[..., 0]
    behavior = torch.stack(
        [medians.squeeze(-1), q90 - q10, patch_mad, patch_std, slope, endpoint], dim=-1
    )
    for name, value in {"centered": centered, "differences": differences, "behavior": behavior}.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite numeric feature: {name}")
    return NumericPatchFeatures(centered, differences, behavior, center, scale)


def _raw_numeric_components(
    features: NumericPatchFeatures, query: int, references: torch.Tensor
) -> dict[str, torch.Tensor]:
    refs = references.to("cpu", torch.long)
    q_centered = features.centered[query]
    r_centered = features.centered[refs]
    point = (q_centered[None, :, None, :] - r_centered[:, None, :, :]).abs().median(dim=-1).values
    q_diff = features.differences[query]
    r_diff = features.differences[refs]
    rate = (q_diff[None, :, None, :] - r_diff[:, None, :, :]).abs().median(dim=-1).values
    q_behavior = features.behavior[query]
    r_behavior = features.behavior[refs]
    behavior_delta = (q_behavior[None, :, None, :] - r_behavior[:, None, :, :]).abs()
    result = {"shape_level_free": point, "change_rate": rate}
    behavior_names = ("level", "amplitude", "mad", "std", "slope", "endpoint_change")
    result.update({name: behavior_delta[..., index] for index, name in enumerate(behavior_names)})
    return result


@torch.inference_mode()
def build_cost_bundle(tokens: torch.Tensor, pool: Any, raw: np.ndarray) -> CostBundle:
    """Compute shared visual and numeric candidate matrices and calibrations."""

    if tokens.ndim == 3 and tokens.shape[1] == GRID_SIZE * GRID_SIZE:
        grids = tokens.reshape(tokens.shape[0], GRID_SIZE, GRID_SIZE, -1)
    elif tokens.ndim == 4 and tokens.shape[1:3] == (GRID_SIZE, GRID_SIZE):
        grids = tokens
    else:
        raise ValueError("tokens must be [W,196,D] or [W,14,14,D]")
    window_ids = torch.as_tensor(pool.window_ids, dtype=torch.long, device=grids.device)
    valid = torch.as_tensor(pool.valid, dtype=torch.bool, device=grids.device)
    features = numeric_patch_features(raw)
    if features.centered.shape[0] != grids.shape[0]:
        raise RuntimeError("numeric and visual window counts differ")

    visual: list[torch.Tensor] = []
    ids_cpu: list[torch.Tensor] = []
    components: dict[str, list[torch.Tensor]] = {}
    for query in range(grids.shape[0]):
        ids = window_ids[query, valid[query]]
        ids_cpu.append(ids.to("cpu"))
        visual.append(column_cost_matrix(grids[query], grids[ids]).to("cpu", torch.float32))
        raw_components = _raw_numeric_components(features, query, ids)
        for name, value in raw_components.items():
            components.setdefault(name, []).append(value.to(torch.float32))

    scales = {name: _positive_median(values) for name, values in components.items()}
    numeric: list[torch.Tensor] = []
    behavior_names = ("level", "amplitude", "mad", "std", "slope", "endpoint_change")
    for index in range(len(visual)):
        shape = 0.5 * (
            torch.log1p(components["shape_level_free"][index] / max(scales["shape_level_free"], EPS))
            + torch.log1p(components["change_rate"][index] / max(scales["change_rate"], EPS))
        )
        behavior = torch.stack(
            [torch.log1p(components[name][index] / max(scales[name], EPS)) for name in behavior_names],
            dim=0,
        ).mean(dim=0)
        numeric.append(0.5 * (shape + behavior))
    visual_median = _positive_median(visual)
    numeric_median = _positive_median(numeric)
    calibration = float(np.clip(visual_median / max(numeric_median, EPS), 1e-4, 1e4))
    numeric = [value * calibration for value in numeric]
    if any(not torch.isfinite(value).all() for value in numeric):
        raise RuntimeError("non-finite calibrated numeric costs")
    scales["visual_positive_median"] = visual_median
    scales["numeric_positive_median_before_visual_calibration"] = numeric_median
    return CostBundle(visual, numeric, components, scales, calibration)


def penalty_from_costs(costs: list[torch.Tensor], scale: float = 0.1) -> float:
    return float(scale * _positive_median(costs, default=0.0))


def match_paths(
    costs: list[torch.Tensor], pool: Any, *, band: int = 2, penalty_scale: float = 0.1
) -> PathMatch:
    """Run the exact T3 selector while retaining each discrete DTW path."""

    ids_all = torch.as_tensor(pool.window_ids, dtype=torch.long, device="cpu")
    valid_all = torch.as_tensor(pool.valid, dtype=torch.bool, device="cpu")
    penalty = penalty_from_costs(costs, penalty_scale)
    scores, references, columns, masks, objectives, lengths, positions = [], [], [], [], [], [], []
    for query, matrices in enumerate(costs):
        ids = ids_all[query, valid_all[query]]
        result = constrained_dtw_match(matrices, ids, band=band, non_diagonal_penalty=penalty)
        selected = int(torch.nonzero(ids == result.reference_id, as_tuple=False)[0, 0])
        objective, path = _dtw_one(matrices[selected], band=band, non_diagonal_penalty=penalty)
        mask = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.bool)
        for i, j in path:
            mask[i, j] = True
        scores.append(result.scores)
        references.append(result.reference_id)
        columns.append(result.reference_columns)
        masks.append(mask)
        objectives.append(objective)
        lengths.append(len(path))
        positions.append(selected)
    return PathMatch(
        torch.stack(scores), torch.stack(references), torch.stack(columns), torch.stack(masks),
        torch.tensor(objectives, dtype=torch.float32), torch.tensor(lengths, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long), penalty,
    )


def score_costs_on_paths(costs: list[torch.Tensor], match: PathMatch) -> torch.Tensor:
    """Average an alternate modality along an already selected exact path."""

    rows = []
    for query, matrices in enumerate(costs):
        selected = int(match.selected_pool_positions[query])
        matrix = matrices[selected]
        mask = match.path_masks[query]
        values = []
        for i in range(GRID_SIZE):
            chosen = matrix[i, mask[i]]
            if chosen.numel() == 0:
                raise RuntimeError("path does not cover every query column")
            values.append(chosen.mean())
        rows.append(torch.stack(values))
    return torch.stack(rows)


def mix_costs(visual: list[torch.Tensor], numeric: list[torch.Tensor], weight: float) -> list[torch.Tensor]:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0,1]")
    return [(1.0 - weight) * left + weight * right for left, right in zip(visual, numeric)]


__all__ = [
    "CostBundle", "NumericPatchFeatures", "PathMatch", "build_cost_bundle", "match_paths",
    "mix_costs", "numeric_patch_features", "penalty_from_costs", "score_costs_on_paths",
]
