"""Temporal matching on paired ViT4TS 14 x 14 visual Patch features.

This module changes only how cached two-dimensional Patch features are
matched.  For a query column ``i`` and a reference column ``j`` it computes

``C(i,j) = mean_a min_b max(0, 1 - cosine(Vq[a,i], Vr[b,j]))``.

The same column-distance matrices drive all three variants: independent
column matching (T1), one reference window and one shared shift (T2), and
band-constrained monotone DTW within one reference window (T3).  Candidate
window selection remains outside this module so callers can reuse the exact
same label-free pool as paired visual matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


GRID_SIZE = 14


@dataclass(frozen=True)
class IndependentTemporalMatch:
    """T1 result for one query window."""

    scores: torch.Tensor
    reference_ids: torch.Tensor
    reference_columns: torch.Tensor


@dataclass(frozen=True)
class ShiftConsistentMatch:
    """T2 result for one query window.

    ``valid`` marks the query columns for which ``i + shift`` is inside the
    reference window.  Invalid edge positions are filled with the mean cost
    of the valid alignment in ``scores`` so the downstream 14-column anomaly
    map stays finite without inventing a different temporal correspondence.
    """

    scores: torch.Tensor
    reference_id: torch.Tensor
    shift: torch.Tensor
    valid: torch.Tensor
    objective_cost: torch.Tensor


@dataclass(frozen=True)
class ConstrainedDTWMatch:
    """T3 result for one query window."""

    scores: torch.Tensor
    reference_id: torch.Tensor
    offsets: torch.Tensor
    reference_columns: torch.Tensor
    normalized_path_cost: torch.Tensor
    path_length: torch.Tensor


@dataclass(frozen=True)
class TemporalMatchResult:
    """Batched T1/T2/T3 products for all query windows.

    The three ``*_scores`` fields use ``output_factor`` for compatibility
    with ViT4TS Patch scores.  Objective/path costs and ``dtw_penalty`` retain
    their raw column-distance scale so diagnostics describe the actual
    optimization problem.
    """

    t1_scores: torch.Tensor
    t1_reference_ids: torch.Tensor
    t1_reference_columns: torch.Tensor
    t2_scores: torch.Tensor
    t2_reference_ids: torch.Tensor
    t2_shifts: torch.Tensor
    t2_valid: torch.Tensor
    t2_objective_costs: torch.Tensor
    t3_scores: torch.Tensor
    t3_reference_ids: torch.Tensor
    t3_offsets: torch.Tensor
    t3_reference_columns: torch.Tensor
    t3_path_costs: torch.Tensor
    t3_path_lengths: torch.Tensor
    dtw_penalty: torch.Tensor


def _single_grid(tokens: torch.Tensor, *, name: str) -> torch.Tensor:
    if tokens.ndim == 2 and tokens.shape[0] == GRID_SIZE * GRID_SIZE:
        tokens = tokens.reshape(GRID_SIZE, GRID_SIZE, -1)
    if tokens.ndim != 3 or tokens.shape[:2] != (GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"{name} must have shape [196,D] or [{GRID_SIZE},{GRID_SIZE},D]"
        )
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
        return _single_grid(tokens, name="reference" ).unsqueeze(0), True
    if tokens.ndim == 3 and tokens.shape[1] == GRID_SIZE * GRID_SIZE:
        tokens = tokens.reshape(tokens.shape[0], GRID_SIZE, GRID_SIZE, -1)
    if tokens.ndim != 4 or tokens.shape[1:3] != (GRID_SIZE, GRID_SIZE):
        raise ValueError(
            "reference must have shape [196,D], [14,14,D], "
            "[R,196,D], or [R,14,14,D]"
        )
    if not tokens.is_floating_point():
        tokens = tokens.float()
    if not torch.isfinite(tokens).all():
        raise ValueError("reference contains non-finite values")
    return tokens, False


def column_cost_matrix(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compute exact visual column costs for one or more reference windows.

    Parameters
    ----------
    query:
        ``[196,D]`` or row-major ``[14,14,D]`` query Patch features.
    reference:
        One reference in either query shape, or a batch ``[R,196,D]`` /
        ``[R,14,14,D]``.

    Returns
    -------
    torch.Tensor
        ``[14,14]`` for one reference or ``[R,14,14]`` for a batch.  Rows are
        query time columns and columns are reference time columns.
    """

    query_grid = _single_grid(query, name="query")
    references, single = _reference_grids(reference)
    if references.shape[-1] != query_grid.shape[-1]:
        raise ValueError("query and reference feature dimensions differ")

    # Grid layout is [vertical row, time column, feature].  Reorder to make
    # the temporal dimensions explicit, then compare every vertical Patch
    # pair for each (query column, reference column).
    q = F.normalize(query_grid.permute(1, 0, 2), dim=-1, eps=1e-12)
    r = F.normalize(references.permute(0, 2, 1, 3), dim=-1, eps=1e-12)
    similarities = torch.einsum("iad,rjbd->rijab", q, r)
    distances = (1.0 - similarities).clamp(min=0.0, max=2.0)
    costs = distances.amin(dim=-1).mean(dim=-1)
    return costs[0] if single else costs


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
    if ids.ndim != 1 or ids.numel() != costs.shape[0]:
        raise ValueError("reference_ids must contain one ID per cost matrix")
    if ids.numel() < 1:
        raise ValueError("at least one reference is required")
    return costs, ids


def independent_temporal_match(
    costs: torch.Tensor, reference_ids: torch.Tensor
) -> IndependentTemporalMatch:
    """T1: independently minimize every query column over windows/columns."""

    costs, ids = _validate_costs(costs, reference_ids)
    # [R,I,J] -> [I,R,J], with a stable flattened window-major index.
    flattened = costs.permute(1, 0, 2).reshape(GRID_SIZE, -1)
    scores, indices = flattened.min(dim=1)
    pool_positions = indices // GRID_SIZE
    columns = indices % GRID_SIZE
    return IndependentTemporalMatch(scores, ids[pool_positions], columns)


def shift_consistent_match(
    costs: torch.Tensor,
    reference_ids: torch.Tensor,
    *,
    shift_radius: int = 2,
) -> ShiftConsistentMatch:
    """T2: choose one reference and one common temporal shift.

    A shift ``delta`` means query column ``i`` is matched to reference column
    ``i + delta``.  Selection averages only positions inside the reference
    range, as required by the overlap alignment.
    """

    costs, ids = _validate_costs(costs, reference_ids)
    if not isinstance(shift_radius, int) or shift_radius < 0:
        raise ValueError("shift_radius must be a non-negative integer")
    if shift_radius >= GRID_SIZE:
        raise ValueError("shift_radius must be smaller than the grid size")

    # Zero shift wins exact ties, followed by the smallest absolute shift.
    # This avoids reporting an arbitrary boundary-truncated alignment for
    # constant or identical windows.
    shift_values = sorted(
        range(-shift_radius, shift_radius + 1), key=lambda value: (abs(value), value)
    )
    shifts = torch.tensor(shift_values, device=costs.device, dtype=torch.long)
    positions = torch.arange(GRID_SIZE, device=costs.device)
    objectives = []
    for shift in shifts.tolist():
        valid = (positions + shift >= 0) & (positions + shift < GRID_SIZE)
        query_columns = positions[valid]
        reference_columns = query_columns + shift
        objectives.append(costs[:, query_columns, reference_columns].mean(dim=1))
    objective_grid = torch.stack(objectives, dim=1)
    flat_index = objective_grid.reshape(-1).argmin()
    shift_count = shifts.numel()
    pool_position = flat_index // shift_count
    shift_position = flat_index % shift_count
    shift = shifts[shift_position]
    objective = objective_grid[pool_position, shift_position]

    valid = (positions + shift >= 0) & (positions + shift < GRID_SIZE)
    scores = torch.full(
        (GRID_SIZE,), objective, dtype=costs.dtype, device=costs.device
    )
    scores[valid] = costs[
        pool_position, positions[valid], positions[valid] + shift
    ]
    return ShiftConsistentMatch(
        scores=scores,
        reference_id=ids[pool_position],
        shift=shift,
        valid=valid,
        objective_cost=objective,
    )


def _dtw_one(
    cost: torch.Tensor, *, band: int, non_diagonal_penalty: float
) -> tuple[float, list[tuple[int, int]]]:
    """CPU dynamic program for one 14 x 14 matrix, including backtrace."""

    # The temporal matrices are tiny.  Keeping the dynamic program on CPU
    # avoids thousands of scalar CUDA synchronizations while the expensive
    # Patch cosine computation remains batched on the accelerator.
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
        j_start = max(0, i - band)
        j_stop = min(size, i + band + 1)
        for j in range(j_start, j_stop):
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
            # The move rank makes ties deterministic: diagonal, vertical,
            # horizontal.  This prevents arbitrary warping on identical data.
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


def constrained_dtw_match(
    costs: torch.Tensor,
    reference_ids: torch.Tensor,
    *,
    band: int = 2,
    non_diagonal_penalty: float | torch.Tensor = 0.0,
) -> ConstrainedDTWMatch:
    """T3: select the lowest normalized constrained-DTW reference path."""

    costs, ids = _validate_costs(costs, reference_ids)
    if not isinstance(band, int) or band < 0 or band >= GRID_SIZE:
        raise ValueError("band must be an integer in [0, 13]")
    penalty = float(torch.as_tensor(non_diagonal_penalty).detach().cpu())
    if not torch.isfinite(torch.tensor(penalty)) or penalty < 0:
        raise ValueError("non_diagonal_penalty must be finite and non-negative")

    candidates = [_dtw_one(matrix, band=band, non_diagonal_penalty=penalty) for matrix in costs]
    best_position = min(range(len(candidates)), key=lambda index: (candidates[index][0], index))
    normalized_cost, path = candidates[best_position]

    score_lists: list[list[float]] = [[] for _ in range(GRID_SIZE)]
    column_lists: list[list[int]] = [[] for _ in range(GRID_SIZE)]
    selected_cpu = costs[best_position].detach().cpu()
    for query_column, reference_column in path:
        score_lists[query_column].append(float(selected_cpu[query_column, reference_column]))
        column_lists[query_column].append(reference_column)
    # An end-to-end monotone path necessarily visits every query position.
    if any(not values for values in score_lists):
        raise RuntimeError("DTW path did not cover every query column")
    device, dtype = costs.device, costs.dtype
    scores = torch.tensor(
        [sum(values) / len(values) for values in score_lists],
        device=device,
        dtype=dtype,
    )
    reference_columns = torch.tensor(
        [sum(values) / len(values) for values in column_lists],
        device=device,
        dtype=dtype,
    )
    positions = torch.arange(GRID_SIZE, device=device, dtype=dtype)
    return ConstrainedDTWMatch(
        scores=scores,
        reference_id=ids[best_position],
        offsets=reference_columns - positions,
        reference_columns=reference_columns,
        normalized_path_cost=torch.tensor(normalized_cost, device=device, dtype=dtype),
        path_length=torch.tensor(len(path), device=device, dtype=torch.long),
    )


def _pool_tensors(pool: Any, windows: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(pool, "window_ids") or not hasattr(pool, "valid"):
        raise TypeError("pool must provide window_ids and valid tensors")
    window_ids = torch.as_tensor(pool.window_ids, device=device, dtype=torch.long)
    valid = torch.as_tensor(pool.valid, device=device, dtype=torch.bool)
    if window_ids.ndim != 2 or window_ids.shape[0] != windows or valid.shape != window_ids.shape:
        raise ValueError("pool tensors must have shape [query_windows, candidates]")
    if not torch.equal(valid, window_ids >= 0):
        raise ValueError("pool valid mask and padded IDs disagree")
    selected = window_ids[valid]
    if selected.numel() == 0 or torch.any(selected >= windows):
        raise ValueError("pool contains no valid references or an out-of-range ID")
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("every query window needs at least one reference")
    return window_ids, valid


@torch.inference_mode()
def temporal_match_all(
    tokens: torch.Tensor,
    pool: Any,
    *,
    shift_radius: int = 2,
    dtw_band: int = 2,
    dtw_penalty_scale: float = 0.1,
    output_factor: float = 0.5,
    eps: float = 1e-12,
) -> TemporalMatchResult:
    """Compute shared column matrices once and return batched T1/T2/T3.

    The DTW non-diagonal penalty is one fixed scalar for the complete signal:
    ``dtw_penalty_scale * median(positive candidate column costs)``.  It uses
    cached visual features and the label-free candidate pool only.
    """

    if tokens.ndim == 3 and tokens.shape[1] == GRID_SIZE * GRID_SIZE:
        grids = tokens.reshape(tokens.shape[0], GRID_SIZE, GRID_SIZE, -1)
    elif tokens.ndim == 4 and tokens.shape[1:3] == (GRID_SIZE, GRID_SIZE):
        grids = tokens
    else:
        raise ValueError("tokens must have shape [W,196,D] or [W,14,14,D]")
    if not grids.is_floating_point():
        grids = grids.float()
    if not torch.isfinite(grids).all():
        raise ValueError("tokens contain non-finite values")
    if grids.shape[0] < 1:
        raise ValueError("tokens must contain at least one window")
    if not (torch.isfinite(torch.tensor(dtw_penalty_scale)) and dtw_penalty_scale >= 0):
        raise ValueError("dtw_penalty_scale must be finite and non-negative")
    if not (torch.isfinite(torch.tensor(output_factor)) and output_factor >= 0):
        raise ValueError("output_factor must be finite and non-negative")

    window_ids, valid = _pool_tensors(pool, grids.shape[0], grids.device)
    query_costs: list[torch.Tensor] = []
    query_ids: list[torch.Tensor] = []
    for query_index in range(grids.shape[0]):
        ids = window_ids[query_index, valid[query_index]]
        # Column matrices are compact.  Store them on CPU so the small DTW
        # dynamic programs do not introduce repeated CUDA scalar syncs.
        costs = column_cost_matrix(grids[query_index], grids[ids]).to("cpu")
        query_costs.append(costs)
        query_ids.append(ids.to("cpu"))

    positive = torch.cat([cost.reshape(-1) for cost in query_costs])
    positive = positive[torch.isfinite(positive) & (positive > eps)]
    if positive.numel() == 0:
        penalty_cpu = torch.zeros((), dtype=query_costs[0].dtype)
    else:
        penalty_cpu = positive.median() * dtw_penalty_scale

    t1_results = [
        independent_temporal_match(cost, ids)
        for cost, ids in zip(query_costs, query_ids)
    ]
    t2_results = [
        shift_consistent_match(cost, ids, shift_radius=shift_radius)
        for cost, ids in zip(query_costs, query_ids)
    ]
    t3_results = [
        constrained_dtw_match(
            cost, ids, band=dtw_band, non_diagonal_penalty=penalty_cpu
        )
        for cost, ids in zip(query_costs, query_ids)
    ]
    device = grids.device

    def stack(
        name: str, values: list[Any], *, factor: Optional[float] = None
    ) -> torch.Tensor:
        output = torch.stack([getattr(value, name) for value in values]).to(device)
        return output if factor is None else factor * output

    return TemporalMatchResult(
        t1_scores=stack("scores", t1_results, factor=output_factor),
        t1_reference_ids=stack("reference_ids", t1_results),
        t1_reference_columns=stack("reference_columns", t1_results),
        t2_scores=stack("scores", t2_results, factor=output_factor),
        t2_reference_ids=stack("reference_id", t2_results),
        t2_shifts=stack("shift", t2_results),
        t2_valid=stack("valid", t2_results),
        t2_objective_costs=stack("objective_cost", t2_results),
        t3_scores=stack("scores", t3_results, factor=output_factor),
        t3_reference_ids=stack("reference_id", t3_results),
        t3_offsets=stack("offsets", t3_results),
        t3_reference_columns=stack("reference_columns", t3_results),
        t3_path_costs=stack("normalized_path_cost", t3_results),
        t3_path_lengths=stack("path_length", t3_results),
        dtw_penalty=penalty_cpu.to(device),
    )

