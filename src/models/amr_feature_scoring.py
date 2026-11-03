"""Fixed-path AMR feature evidence and reference-conditioned volatility scaling."""
from __future__ import annotations

import numpy as np
import torch

from src.models.fixed_reference_numeric_evidence import positive_median
from src.models.numeric_temporal_matching import EPS, PathMatch


def reference_volatility_ratio(
    standardized_windows: torch.Tensor,
    match: PathMatch,
    *,
    minimum: float = 0.5,
    maximum: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return one robust reference-volatility ratio per query Patch.

    Only the selected reference window and the already fixed DTW path are read.
    The ratio combines reference-Patch level MAD and first-difference MAD, then
    uses one unlabeled global median and fixed clipping to avoid scale collapse.
    """
    windows = standardized_windows.detach().cpu().float()
    if windows.ndim != 2 or windows.shape[1] != 224:
        raise ValueError("standardized_windows must be [windows,224]")
    patches = windows.reshape(len(windows), 14, 16)
    median = patches.median(-1, keepdim=True).values
    level_mad = 1.4826 * (patches - median).abs().median(-1).values
    differences = patches.diff(dim=-1)
    difference_median = differences.median(-1, keepdim=True).values
    difference_mad = 1.4826 * (differences - difference_median).abs().median(-1).values
    level_scale = max(positive_median(level_mad.numpy()), EPS)
    difference_scale = max(positive_median(difference_mad.numpy()), EPS)
    volatility = 0.5 * (level_mad / level_scale + difference_mad / difference_scale)
    volatility = volatility.clamp(minimum, maximum)

    ratios = []
    for query in range(len(match.reference_ids)):
        reference = int(match.reference_ids[query])
        mask = match.path_masks[query].cpu()
        ratios.append(torch.stack([
            volatility[reference, mask[column]].mean() for column in range(14)
        ]))
    result = torch.stack(ratios)
    if not torch.isfinite(result).all():
        raise RuntimeError("non-finite reference volatility ratio")
    diagnostics = {
        "level_mad_positive_median": float(level_scale),
        "difference_mad_positive_median": float(difference_scale),
        "mean_ratio": float(result.mean()),
        "std_ratio": float(result.std(unbiased=False)),
        "minimum_ratio": float(result.min()),
        "maximum_ratio": float(result.max()),
    }
    return result, diagnostics


def volatility_correct(local_evidence: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    if local_evidence.shape != ratio.shape:
        raise ValueError("local evidence and volatility ratio must have identical shape")
    corrected = local_evidence / ratio.clamp_min(EPS)
    if not torch.isfinite(corrected).all():
        raise RuntimeError("non-finite volatility-corrected evidence")
    return corrected


def fuse_single(visual: np.ndarray, evidence: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0,1]")
    return (1.0 - weight) * np.asarray(visual) + weight * np.asarray(evidence)


def fuse_pair(
    visual: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    total_weight: float,
) -> np.ndarray:
    """Split the fixed new-evidence budget equally between two sources."""
    if not 0.0 <= total_weight <= 1.0:
        raise ValueError("total_weight must be in [0,1]")
    half = 0.5 * total_weight
    return ((1.0 - total_weight) * np.asarray(visual)
            + half * np.asarray(first) + half * np.asarray(second))


__all__ = ["fuse_pair", "fuse_single", "reference_volatility_ratio", "volatility_correct"]
