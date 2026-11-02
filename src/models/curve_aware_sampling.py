"""Curve-aware sampling utilities for ViT4TS visual patch features.

The original :func:`src.preprocessing.preprocess.draw_image` fills the complete
Matplotlib canvas with one axes (``subplots_adjust(...=0/1)``), removes the
automatic x margin, and saves the complete figure without a tight bounding
box. The actual ViT4TS call fixes the y limits to ``(0, 1)``. Consequently the
x data limits and fixed y limits map to the outer edges of the saved image.
Matplotlib display coordinates start at the bottom; image and CLIP patch rows
start at the top. This module makes that coordinate change explicit and samples
the row-major 14 x 14 CLIP patch grid at its true patch-centre coordinates.

All functions are label-free and operate on already cached visual features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_GRID_SIZE = 14
DEFAULT_WINDOW_SIZE = 224
DEFAULT_STEP_SIZE = 56


@dataclass(frozen=True)
class CurveCoordinates:
    """Curve locations at the horizontal CLIP patch-centre positions.

    Arrays have shape ``[..., grid_size]``. ``pixel_x`` and ``pixel_y`` use
    image coordinates whose origin is the upper-left image edge. ``patch_x``
    and ``patch_y`` use zero-based feature-grid coordinates, where integer
    values denote patch-token centres. Values of ``-0.5`` and ``grid-0.5`` are
    valid image-edge locations and are border-clamped when sampled.
    """

    pixel_x: np.ndarray
    pixel_y: np.ndarray
    patch_x: np.ndarray
    patch_y: np.ndarray

    def __post_init__(self) -> None:
        shape = self.pixel_y.shape
        if any(value.shape != shape for value in (self.pixel_x, self.patch_x, self.patch_y)):
            raise ValueError("all coordinate arrays must have the same shape")


def _as_window_matrix(values: np.ndarray | Sequence[float], name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape [time] or [windows, time], with time >= 2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array, single


def curve_pixel_coordinates(
    window_values: np.ndarray | Sequence[float],
    time_points: Optional[np.ndarray | Sequence[float]] = None,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    grid_size: int = DEFAULT_GRID_SIZE,
    y_limits: Optional[tuple[float, float]] = (0.0, 1.0),
) -> CurveCoordinates:
    """Map the rendered curve to the horizontal CLIP token centres.

    The curve ordinate is linearly interpolated in the supplied time
    coordinates, exactly as Matplotlib's line segments are. With the default
    ``y_limits=(0,1)`` matches the actual ViT4TS rendering call. If explicitly
    set to ``None``, non-constant windows use their own min/max, matching the
    generic ``draw_image`` default. In that generic mode a constant window maps
    to the vertical canvas centre, matching Matplotlib's symmetric nonsingular
    limit expansion.

    Parameters
    ----------
    window_values:
        ``[T]`` or ``[W,T]`` values passed to the plotter.
    time_points:
        ``[T]`` shared times or ``[W,T]`` per-window times. Omit for ``0..T-1``.
        Times must be strictly increasing because a single curve ordinate must
        exist for every horizontal sampling position.
    image_size:
        Saved image ``(height, width)``.
    grid_size:
        Number of ViT patches along each image dimension.
    y_limits:
        Explicit plotting limits, corresponding to ``plt.ylim``. ViT4TS passes
        ``(0,1)`` after global detrend/min-max preprocessing.
    """

    values, single = _as_window_matrix(window_values, "window_values")
    windows, length = values.shape
    height, width = (int(image_size[0]), int(image_size[1]))
    if height < 1 or width < 1 or grid_size < 1:
        raise ValueError("image dimensions and grid_size must be positive")

    if time_points is None:
        times = np.broadcast_to(np.arange(length, dtype=np.float64), values.shape)
    else:
        raw_times = np.asarray(time_points, dtype=np.float64)
        if raw_times.ndim == 1:
            if raw_times.shape[0] != length:
                raise ValueError("time_points length differs from window_values")
            times = np.broadcast_to(raw_times, values.shape)
        elif raw_times.shape == values.shape:
            times = raw_times
        else:
            raise ValueError("time_points must have shape [time] or [windows, time]")
        if not np.isfinite(times).all():
            raise ValueError("time_points contains non-finite values")
    if np.any(np.diff(times, axis=1) <= 0):
        raise ValueError("time_points must be strictly increasing in every window")

    # Token centres in edge-based image coordinates. The original axes bbox is
    # exactly [0,width] x [0,height], so these are also display x coordinates.
    pixel_x_1d = (np.arange(grid_size, dtype=np.float64) + 0.5) * width / grid_size
    pixel_x = np.broadcast_to(pixel_x_1d, (windows, grid_size)).copy()
    curve_values = np.empty_like(pixel_x)
    for window in range(windows):
        xmin, xmax = times[window, 0], times[window, -1]
        sample_times = xmin + (pixel_x_1d / width) * (xmax - xmin)
        curve_values[window] = np.interp(sample_times, times[window], values[window])

    if y_limits is not None:
        ymin, ymax = float(y_limits[0]), float(y_limits[1])
        if not np.isfinite([ymin, ymax]).all() or ymax <= ymin:
            raise ValueError("y_limits must be finite and strictly increasing")
        ymins = np.full((windows, 1), ymin, dtype=np.float64)
        ymaxs = np.full((windows, 1), ymax, dtype=np.float64)
    else:
        ymins = values.min(axis=1, keepdims=True)
        ymaxs = values.max(axis=1, keepdims=True)

    spans = ymaxs - ymins
    constant = spans[:, 0] == 0
    safe_spans = np.where(spans > 0, spans, 1.0)
    # Matplotlib y display coordinate is height*(y-ymin)/(ymax-ymin).
    # PIL/image coordinates reverse it, hence ymax-y below.
    pixel_y = height * (ymaxs - curve_values) / safe_spans
    pixel_y[constant] = height / 2.0

    patch_width = width / grid_size
    patch_height = height / grid_size
    patch_x = pixel_x / patch_width - 0.5
    patch_y = pixel_y / patch_height - 0.5
    if single:
        return CurveCoordinates(pixel_x[0], pixel_y[0], patch_x[0], patch_y[0])
    return CurveCoordinates(pixel_x, pixel_y, patch_x, patch_y)


def _patch_grid(patch_tokens: torch.Tensor, grid_size: int = DEFAULT_GRID_SIZE) -> torch.Tensor:
    if patch_tokens.ndim == 4 and patch_tokens.shape[-2] == 1:
        patch_tokens = patch_tokens.squeeze(-2)
    if patch_tokens.ndim == 3:
        if patch_tokens.shape[1] != grid_size * grid_size:
            raise ValueError("flattened patch_tokens has an unexpected token count")
        return patch_tokens.reshape(patch_tokens.shape[0], grid_size, grid_size, -1)
    if patch_tokens.ndim == 4 and patch_tokens.shape[1:3] == (grid_size, grid_size):
        return patch_tokens
    raise ValueError("patch_tokens must have shape [W,G*G,D], [W,G*G,1,D], or [W,G,G,D]")


def _coordinate_tensor(value: np.ndarray | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _patch_coordinate_pair(
    coordinates: CurveCoordinates | np.ndarray | torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return grid x/y, accepting a dataclass or cached ``[...,2]`` grid xy."""

    if isinstance(coordinates, CurveCoordinates):
        return (
            _coordinate_tensor(coordinates.patch_x, reference),
            _coordinate_tensor(coordinates.patch_y, reference),
        )
    grid_xy = _coordinate_tensor(coordinates, reference)
    if grid_xy.ndim not in (2, 3) or grid_xy.shape[-1] != 2:
        raise ValueError("coordinates must be CurveCoordinates or have shape [...,time,2]")
    return grid_xy[..., 0], grid_xy[..., 1]


def bilinear_curve_features(
    patch_tokens: torch.Tensor,
    coordinates: CurveCoordinates | np.ndarray | torch.Tensor,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> torch.Tensor:
    """Bilinearly sample ``[W,G,G,D]`` tokens at curve coordinates.

    Sampling uses border padding at the exact outer image edges. This avoids
    introducing artificial zero features when a curve attains the plotted
    minimum or maximum.
    """

    grid = _patch_grid(patch_tokens, grid_size)
    windows = grid.shape[0]
    patch_x, patch_y = _patch_coordinate_pair(coordinates, grid)
    if patch_x.ndim == 1:
        patch_x = patch_x.unsqueeze(0)
        patch_y = patch_y.unsqueeze(0)
    if patch_x.shape != (windows, grid_size) or patch_y.shape != patch_x.shape:
        raise ValueError("coordinates must have shape [windows, grid_size]")

    # align_corners=False maps normalized coordinates back to token index via
    # index=((normal+1)*G-1)/2. Thus index patch_x maps to the expression below.
    x_normalized = 2.0 * (patch_x + 0.5) / grid_size - 1.0
    y_normalized = 2.0 * (patch_y + 0.5) / grid_size - 1.0
    sample_grid = torch.stack((x_normalized, y_normalized), dim=-1).unsqueeze(1)
    channels_first = grid.permute(0, 3, 1, 2)
    sampled = F.grid_sample(
        channels_first,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.squeeze(2).transpose(1, 2)


def gaussian_curve_features(
    patch_tokens: torch.Tensor,
    coordinates: CurveCoordinates | np.ndarray | torch.Tensor,
    *,
    sigma_patches: float = 1.0,
    radius_patches: Optional[float] = 3.0,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> torch.Tensor:
    """Aggregate same-column tokens with a Gaussian around the curve row.

    ``sigma_patches`` is fixed in patch units and requires no labels. Horizontal
    positions are the exact 14 token columns; only vertical neighbours are
    aggregated, preserving temporal order and local vertical context.
    """

    if not np.isfinite(sigma_patches) or sigma_patches <= 0:
        raise ValueError("sigma_patches must be finite and positive")
    if radius_patches is not None and (
        not np.isfinite(radius_patches) or radius_patches <= 0
    ):
        raise ValueError("radius_patches must be finite and positive, or None")
    grid = _patch_grid(patch_tokens, grid_size)
    windows = grid.shape[0]
    _, patch_y = _patch_coordinate_pair(coordinates, grid)
    if patch_y.ndim == 1:
        patch_y = patch_y.unsqueeze(0)
    if patch_y.shape != (windows, grid_size):
        raise ValueError("coordinates must have shape [windows, grid_size]")

    rows = torch.arange(grid_size, device=grid.device, dtype=grid.dtype)
    offsets = rows[None, None, :] - patch_y[:, :, None]
    weights = torch.exp(-0.5 * (offsets / sigma_patches) ** 2)
    if radius_patches is not None:
        weights = weights * (offsets.abs() <= radius_patches)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(grid.dtype).tiny)
    # grid is [W,row,column,D], while weights are [W,column,row].
    return torch.einsum("wcr,wrcd->wcd", weights, grid)


def column_mean_features(
    patch_tokens: torch.Tensor, *, grid_size: int = DEFAULT_GRID_SIZE
) -> torch.Tensor:
    """Average every visual grid column over rows, returning ``[W,G,D]``."""

    return _patch_grid(patch_tokens, grid_size).mean(dim=1)


def paired_time_feature_scores(
    time_features: torch.Tensor,
    candidate_window_ids: torch.Tensor,
    candidate_valid: Optional[torch.Tensor] = None,
    *,
    output_factor: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Nearest-reference scores for ordered visual features.

    Every query time feature is compared with all ordered features belonging to
    the query's supplied candidate windows. The candidate pool is an input so
    column mean, direct sampling, and Gaussian sampling can share exactly the
    same reference windows and overlap exclusions.
    """

    if time_features.ndim != 3:
        raise ValueError("time_features must have shape [windows,time,dim]")
    windows = time_features.shape[0]
    ids = torch.as_tensor(candidate_window_ids, device=time_features.device, dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] != windows:
        raise ValueError("candidate_window_ids must have shape [windows,candidates]")
    if candidate_valid is None:
        valid = ids >= 0
    else:
        valid = torch.as_tensor(candidate_valid, device=ids.device, dtype=torch.bool)
        if valid.shape != ids.shape or not torch.equal(valid, ids >= 0):
            raise ValueError("candidate_valid and padded candidate IDs disagree")

    normalized = F.normalize(time_features, dim=-1, eps=eps)
    scores = []
    for query in range(windows):
        reference_ids = ids[query, valid[query]]
        if reference_ids.numel() == 0:
            raise ValueError(f"query window {query} has no valid reference window")
        if torch.any((reference_ids < 0) | (reference_ids >= windows)):
            raise ValueError("candidate window ID is out of range")
        references = normalized[reference_ids].reshape(-1, normalized.shape[-1])
        distances = (1.0 - normalized[query] @ references.T).clamp_min(0.0)
        scores.append(output_factor * distances.min(dim=-1).values)
    return torch.stack(scores)


def restore_column_scores(
    column_scores: torch.Tensor,
    full_length: int,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> torch.Tensor:
    """Interpolate 14 scores/window and stitch them onto the original timeline.

    This is equivalent to repeating each column score over map rows, applying
    the original bilinear 14->224 upsampling, top-row aggregation, overlap
    averaging, and final linear tail extrapolation.
    """

    if column_scores.ndim != 2 or column_scores.shape[0] < 1:
        raise ValueError("column_scores must have shape [windows,time]")
    if full_length < 1 or window_size < 1 or step_size < 1:
        raise ValueError("lengths must be positive")
    windows = column_scores.shape[0]
    covered = window_size + (windows - 1) * step_size
    if full_length < covered:
        raise ValueError("full_length is shorter than the sliding-window coverage")

    vectors = F.interpolate(
        column_scores.unsqueeze(1),
        size=window_size,
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    total = torch.zeros(covered, dtype=vectors.dtype, device=vectors.device)
    count = torch.zeros(covered, dtype=vectors.dtype, device=vectors.device)
    for window in range(windows):
        start = window * step_size
        total[start : start + window_size] += vectors[window]
        count[start : start + window_size] += 1
    restored = total / count.clamp_min(1)
    if full_length > covered:
        slope = restored[-1] - restored[-2] if covered > 1 else restored.new_zeros(())
        offsets = torch.arange(1, full_length - covered + 1, device=restored.device, dtype=restored.dtype)
        restored = torch.cat((restored, restored[-1] + slope * offsets))
    return restored


__all__ = [
    "CurveCoordinates",
    "bilinear_curve_features",
    "column_mean_features",
    "curve_pixel_coordinates",
    "gaussian_curve_features",
    "paired_time_feature_scores",
    "restore_column_scores",
]
