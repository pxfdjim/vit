"""Shared differentiable correction restoration and unlabeled calibration."""
from __future__ import annotations

import torch
import torch.nn.functional as F


WINDOW = 224
STEP = 56


def restore_point(local: torch.Tensor, full_length: int | None = None) -> torch.Tensor:
    """Average overlapping window corrections and extrapolate an uncovered tail."""
    if local.ndim != 2 or local.shape[1] != 14 or local.shape[0] < 1:
        raise ValueError("local correction must have shape [windows,14]")
    windows = local.shape[0]
    interpolated = F.interpolate(local.double()[:, None], size=WINDOW, mode="linear",
                                 align_corners=False)[:, 0]
    covered = WINDOW + STEP * (windows - 1)
    positions = torch.arange(WINDOW, device=local.device)[None] + \
        STEP * torch.arange(windows, device=local.device)[:, None]
    point = local.new_zeros(covered, dtype=torch.float64)
    counts = local.new_zeros(covered, dtype=torch.float64)
    point.scatter_add_(0, positions.reshape(-1), interpolated.reshape(-1))
    counts.scatter_add_(0, positions.reshape(-1), torch.ones_like(interpolated).reshape(-1))
    point = point / counts.clamp_min(1)
    length = covered if full_length is None else int(full_length)
    if length < 1:
        raise ValueError("full_length must be positive")
    if length > covered:
        slope = point[-1] - point[-2]
        extra = point[-1] + slope * torch.arange(1, length - covered + 1,
                                                 device=local.device, dtype=point.dtype)
        return torch.cat((point, extra))
    return point[:length]


def calibrated_delta(local: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Scale an already bounded signed correction by unlabeled positive medians."""
    point = restore_point(local, base.numel())
    magnitude = point.detach().abs()
    magnitude = magnitude[torch.isfinite(magnitude) & (magnitude > 1e-12)]
    if not magnitude.numel():
        return point, 1.0
    reference = base.detach().reshape(-1)
    reference = reference[torch.isfinite(reference) & (reference > 1e-8)]
    base_median = float(torch.quantile(reference.double(), .5)) if reference.numel() else 1.0
    scale = base_median / max(float(torch.quantile(magnitude, .5)), 1e-12)
    scale = float(min(max(scale, 1e-4), 1e4))
    return point * scale, scale


def final_score(local: torch.Tensor, base: torch.Tensor,
                gamma: float = .05) -> tuple[torch.Tensor, float]:
    delta, scale = calibrated_delta(local, base)
    return base + gamma * delta, scale
