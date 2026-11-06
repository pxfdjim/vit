"""T3 reference retrieval and score restoration used by final APRA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


WINDOW_SIZE = 224
STEP_SIZE = 56
GRID_SIZE = 14
PATCH_COUNT = GRID_SIZE * GRID_SIZE
OVERLAP_RADIUS = WINDOW_SIZE // STEP_SIZE
MAX_CANDIDATES = 32
EPS = 1e-8


def fitted_standardized_windows(
    raw: np.ndarray, stats: dict[str, float], start_index: int
) -> torch.Tensor:
    """Apply normalization fitted on the declared unlabeled training source."""

    values = np.asarray(raw, np.float64)
    x = start_index + np.arange(len(values), dtype=np.float64)
    trend = float(stats["trend_intercept"]) + float(stats["trend_slope"]) * x
    normalized = (values - trend - float(stats["center"])) / float(stats["scale"])
    return torch.from_numpy(normalized.astype(np.float32)).unfold(
        0, WINDOW_SIZE, STEP_SIZE
    ).contiguous()


def patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 4 and tokens.shape[-2] == 1:
        tokens = tokens.squeeze(-2)
    if tokens.ndim != 3 or tokens.shape[1] != PATCH_COUNT:
        raise ValueError("patch tokens must have shape [windows, 196, dim]")
    return tokens


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_n = F.normalize(left, dim=-1, eps=1e-12)
    right_n = F.normalize(right, dim=-1, eps=1e-12)
    return (1.0 - left_n @ right_n.transpose(-1, -2)).clamp_min(0.0)


def _eligible_reference_mask(
    windows: int,
    *,
    overlap_radius: int = OVERLAP_RADIUS,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    if windows < 1:
        raise ValueError("windows must be positive")
    if overlap_radius < 1:
        raise ValueError("overlap_radius must be positive")
    indices = torch.arange(windows, device=device)
    return (indices[:, None] - indices[None, :]).abs() >= overlap_radius


@dataclass(frozen=True)
class ReferencePool:
    window_ids: torch.Tensor
    valid: torch.Tensor
    visual_distances: torch.Tensor

    def __post_init__(self) -> None:
        if self.window_ids.ndim != 2:
            raise ValueError("window_ids must have shape [query, candidates]")
        if self.valid.shape != self.window_ids.shape:
            raise ValueError("valid shape differs from window_ids")
        if self.visual_distances.shape != self.window_ids.shape:
            raise ValueError("visual_distances shape differs from window_ids")
        if not torch.equal(self.valid, self.window_ids >= 0):
            raise ValueError("valid mask and padded window IDs disagree")


def select_reference_windows(
    visual_windows: torch.Tensor,
    *,
    max_candidates: int = MAX_CANDIDATES,
    overlap_radius: int = OVERLAP_RADIUS,
) -> ReferencePool:
    """Select non-overlapping T3 candidates by mean-token cosine distance."""

    tokens = patch_tokens(visual_windows)
    windows = tokens.shape[0]
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    eligible = _eligible_reference_mask(
        windows, overlap_radius=overlap_radius, device=tokens.device
    )
    counts = eligible.sum(dim=1)
    if torch.any(counts == 0):
        failed = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            "paired reference requires a non-overlapping reference for every query; "
            f"none for query windows {failed}"
        )
    means = tokens.mean(dim=1)
    distances = _cosine_distance(means.unsqueeze(0), means)[0]
    distances = distances.masked_fill(~eligible, torch.inf)
    width = min(max_candidates, int(counts.max().item()))
    values, ids = distances.topk(width, dim=-1, largest=False, sorted=True)
    valid = torch.isfinite(values)
    return ReferencePool(ids.masked_fill(~valid, -1), valid, values)


def _single_grid(tokens: torch.Tensor, *, name: str) -> torch.Tensor:
    if tokens.ndim == 2 and tokens.shape[0] == PATCH_COUNT:
        tokens = tokens.reshape(GRID_SIZE, GRID_SIZE, -1)
    if tokens.ndim != 3 or tokens.shape[:2] != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"{name} must have shape [196,D] or [14,14,D]")
    if not tokens.is_floating_point():
        tokens = tokens.float()
    if not torch.isfinite(tokens).all():
        raise ValueError(f"{name} contains non-finite values")
    return tokens


def _reference_grids(tokens: torch.Tensor) -> tuple[torch.Tensor, bool]:
    single = tokens.ndim in (2, 3) and (
        tokens.ndim == 2 or tokens.shape[:2] == (GRID_SIZE, GRID_SIZE)
    )
    if single:
        return _single_grid(tokens, name="reference").unsqueeze(0), True
    if tokens.ndim == 3 and tokens.shape[1] == PATCH_COUNT:
        tokens = tokens.reshape(tokens.shape[0], GRID_SIZE, GRID_SIZE, -1)
    if tokens.ndim != 4 or tokens.shape[1:3] != (GRID_SIZE, GRID_SIZE):
        raise ValueError("reference must have shape [R,196,D] or [R,14,14,D]")
    if not tokens.is_floating_point():
        tokens = tokens.float()
    if not torch.isfinite(tokens).all():
        raise ValueError("reference contains non-finite values")
    return tokens, False


def column_cost_matrix(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compute the exact visual cost for every query/reference column pair."""

    query_grid = _single_grid(query, name="query")
    references, single = _reference_grids(reference)
    if references.shape[-1] != query_grid.shape[-1]:
        raise ValueError("query and reference feature dimensions differ")
    query_columns = F.normalize(query_grid.permute(1, 0, 2), dim=-1, eps=1e-12)
    reference_columns = F.normalize(
        references.permute(0, 2, 1, 3), dim=-1, eps=1e-12
    )
    similarities = torch.einsum(
        "iad,rjbd->rijab", query_columns, reference_columns
    )
    costs = (1.0 - similarities).clamp(min=0.0, max=2.0).amin(dim=-1).mean(dim=-1)
    return costs[0] if single else costs


def visual_costs(grids: torch.Tensor, pool: ReferencePool) -> list[torch.Tensor]:
    ids_all = torch.as_tensor(pool.window_ids, dtype=torch.long, device=grids.device)
    valid_all = torch.as_tensor(pool.valid, dtype=torch.bool, device=grids.device)
    return [
        column_cost_matrix(grids[q], grids[ids_all[q, valid_all[q]]]).cpu()
        for q in range(grids.shape[0])
    ]


def _validate_costs(
    costs: torch.Tensor, reference_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if costs.ndim == 2:
        costs = costs.unsqueeze(0)
    if costs.ndim != 3 or costs.shape[1:] != (GRID_SIZE, GRID_SIZE):
        raise ValueError("costs must have shape [R,14,14] or [14,14]")
    if not costs.is_floating_point():
        costs = costs.float()
    if not torch.isfinite(costs).all() or torch.any(costs < 0):
        raise ValueError("costs must be finite and non-negative")
    ids = torch.as_tensor(reference_ids, dtype=torch.long, device=costs.device)
    if ids.ndim != 1 or ids.numel() != costs.shape[0] or ids.numel() < 1:
        raise ValueError("reference_ids must contain one ID per cost matrix")
    return costs, ids


def _dtw_one(
    cost: torch.Tensor, *, band: int, non_diagonal_penalty: float
) -> tuple[float, list[tuple[int, int]]]:
    matrix = cost.detach().to(device="cpu", dtype=torch.float64)
    size = matrix.shape[0]
    inf = float("inf")
    accumulated = [[inf] * size for _ in range(size)]
    lengths = [[0] * size for _ in range(size)]
    previous: list[list[Optional[tuple[int, int]]]] = [
        [None] * size for _ in range(size)
    ]
    accumulated[0][0] = float(matrix[0, 0])
    lengths[0][0] = 1
    for i in range(size):
        for j in range(max(0, i - band), min(size, i + band + 1)):
            if i == 0 and j == 0:
                continue
            candidates: list[tuple[float, int, int, int]] = []
            if i > 0 and j > 0 and accumulated[i - 1][j - 1] < inf:
                candidates.append((accumulated[i - 1][j - 1], 0, i - 1, j - 1))
            if i > 0 and accumulated[i - 1][j] < inf:
                candidates.append(
                    (accumulated[i - 1][j] + non_diagonal_penalty, 1, i - 1, j)
                )
            if j > 0 and accumulated[i][j - 1] < inf:
                candidates.append(
                    (accumulated[i][j - 1] + non_diagonal_penalty, 2, i, j - 1)
                )
            if not candidates:
                continue
            best, _, prior_i, prior_j = min(candidates)
            accumulated[i][j] = best + float(matrix[i, j])
            lengths[i][j] = lengths[prior_i][prior_j] + 1
            previous[i][j] = (prior_i, prior_j)
    if accumulated[-1][-1] == inf:
        raise RuntimeError("DTW endpoint is unreachable under the supplied band")
    path = []
    position: Optional[tuple[int, int]] = (size - 1, size - 1)
    while position is not None:
        path.append(position)
        position = previous[position[0]][position[1]]
    path.reverse()
    return accumulated[-1][-1] / lengths[-1][-1], path


@dataclass(frozen=True)
class _ConstrainedMatch:
    scores: torch.Tensor
    reference_id: torch.Tensor
    reference_columns: torch.Tensor


def _constrained_dtw_match(
    costs: torch.Tensor,
    reference_ids: torch.Tensor,
    *,
    band: int = 2,
    non_diagonal_penalty: float = 0.0,
) -> _ConstrainedMatch:
    costs, ids = _validate_costs(costs, reference_ids)
    if not isinstance(band, int) or band < 0 or band >= GRID_SIZE:
        raise ValueError("band must be an integer in [0, 13]")
    penalty = float(torch.as_tensor(non_diagonal_penalty).detach().cpu())
    candidates = [
        _dtw_one(matrix, band=band, non_diagonal_penalty=penalty) for matrix in costs
    ]
    best_position = min(
        range(len(candidates)), key=lambda index: (candidates[index][0], index)
    )
    _, path = candidates[best_position]
    score_lists: list[list[float]] = [[] for _ in range(GRID_SIZE)]
    column_lists: list[list[int]] = [[] for _ in range(GRID_SIZE)]
    selected = costs[best_position].detach().cpu()
    for query_column, reference_column in path:
        score_lists[query_column].append(float(selected[query_column, reference_column]))
        column_lists[query_column].append(reference_column)
    device, dtype = costs.device, costs.dtype
    return _ConstrainedMatch(
        torch.tensor(
            [sum(values) / len(values) for values in score_lists],
            device=device,
            dtype=dtype,
        ),
        ids[best_position],
        torch.tensor(
            [sum(values) / len(values) for values in column_lists],
            device=device,
            dtype=dtype,
        ),
    )


def _torch_positive_median(values: list[torch.Tensor], default: float = 1.0) -> float:
    flat = torch.cat(
        [value.detach().reshape(-1).to("cpu", torch.float32) for value in values]
    )
    flat = flat[torch.isfinite(flat) & (flat > EPS)]
    return float(flat.median()) if flat.numel() else float(default)


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


def match_paths(
    costs: list[torch.Tensor],
    pool: ReferencePool,
    *,
    band: int = 2,
    penalty_scale: float = 0.1,
) -> PathMatch:
    """Run the immutable T3 constrained-DTW selector."""

    ids_all = torch.as_tensor(pool.window_ids, dtype=torch.long, device="cpu")
    valid_all = torch.as_tensor(pool.valid, dtype=torch.bool, device="cpu")
    penalty = float(penalty_scale * _torch_positive_median(costs, default=0.0))
    scores, references, columns = [], [], []
    masks, objectives, lengths, positions = [], [], [], []
    for query, matrices in enumerate(costs):
        ids = ids_all[query, valid_all[query]]
        result = _constrained_dtw_match(
            matrices, ids, band=band, non_diagonal_penalty=penalty
        )
        selected = int(torch.nonzero(ids == result.reference_id, as_tuple=False)[0, 0])
        objective, path = _dtw_one(
            matrices[selected], band=band, non_diagonal_penalty=penalty
        )
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
        torch.stack(scores),
        torch.stack(references),
        torch.stack(columns),
        torch.stack(masks),
        torch.tensor(objectives, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
        penalty,
    )


def column_map(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != GRID_SIZE:
        raise ValueError("temporal scores must have shape [windows, 14]")
    return scores[:, None, :].expand(-1, GRID_SIZE, -1)


def positive_median(values: np.ndarray, default: float = 1.0) -> float:
    flat = np.asarray(values, np.float64).reshape(-1)
    flat = flat[np.isfinite(flat) & (flat > EPS)]
    return float(np.median(flat)) if flat.size else float(default)


def _aggregate_map(anomaly_map: np.ndarray, top_percent: float) -> np.ndarray:
    height, width = anomaly_map.shape
    vector = np.zeros(width, dtype=float)
    count = max(1, int(np.ceil(height * top_percent)))
    for column in range(width):
        vector[column] = np.mean(np.sort(anomaly_map[:, column])[::-1][:count])
    return vector


def _stitch_maps(
    anomaly_maps: np.ndarray, window_step_ratio: float, top_percent: float
) -> np.ndarray:
    windows, _, width = anomaly_maps.shape
    step = int(width / window_step_ratio)
    total = step * (windows - 1) + width
    vectors = np.array([_aggregate_map(item, top_percent) for item in anomaly_maps])
    scores = np.zeros(total, dtype=float)
    counts = np.zeros(total, dtype=int)
    for index in range(windows):
        start = index * step
        scores[start : start + width] += vectors[index]
        counts[start : start + width] += 1
    counts[counts == 0] = 1
    return scores / counts


def _align_scores(values: np.ndarray, full_length: int, windows: int) -> np.ndarray:
    covered = WINDOW_SIZE + (windows - 1) * STEP_SIZE
    old = np.arange(len(values))
    new = np.linspace(0, len(values) - 1, covered)
    interpolated = np.interp(new, old, values)
    if covered < full_length:
        slope = interpolated[-1] - interpolated[-2] if covered > 1 else 0.0
        extra = interpolated[-1] + slope * np.arange(1, full_length - covered + 1)
        return np.concatenate([interpolated, extra])
    return interpolated[:full_length]


def point_scores_from_maps(
    maps: torch.Tensor, *, full_length: int, device: torch.device
) -> np.ndarray:
    maps = torch.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0).to(device)
    upsampled = F.interpolate(
        maps.unsqueeze(1),
        size=(WINDOW_SIZE, WINDOW_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    stitched = _stitch_maps(upsampled.cpu().numpy(), 4.0, 0.25)
    return _align_scores(stitched, full_length, maps.shape[0])
