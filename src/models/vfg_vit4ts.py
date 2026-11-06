"""Core matching utilities for Visual-Frequency Guided ViT4TS.

This module deliberately does not instantiate CLIP or perform dataset I/O.  It
operates on cached ViT-B/16 patch tokens and on the detrended numeric windows
that produced the images.  Keeping this layer independent makes the A/B/C/D
comparisons share exactly the same visual features and reference range.

The original ViT4TS implementation multiplies cosine patch distance by 0.5.
``VisualFrequencyMatcher`` retains that factor at its output so ``lambda=0``
is an exact recovery of the stable visual base-patch score.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FrequencyMode = Literal["shape", "energy", "both"]


def linear_detrend_series(values: torch.Tensor) -> torch.Tensor:
    """Remove one least-squares linear trend from each complete series.

    Detrending is intentionally separate from windowing.  Call this on the
    complete series and then create 224-point windows; no per-window min-max
    normalization is applied to the frequency input.
    """

    x = torch.as_tensor(values)
    if not x.is_floating_point():
        x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.ndim >= 2:
        squeeze = False
    else:
        raise ValueError("values must have at least one dimension")

    length = x.shape[-1]
    if length < 2:
        result = x - x.mean(dim=-1, keepdim=True)
        return result.squeeze(0) if squeeze else result
    time = torch.linspace(-1.0, 1.0, length, dtype=x.dtype, device=x.device)
    centered = x - x.mean(dim=-1, keepdim=True)
    slope = (centered * time).sum(dim=-1, keepdim=True) / time.square().sum()
    result = centered - slope * time
    return result.squeeze(0) if squeeze else result


@dataclass(frozen=True)
class FrequencyEncoding:
    """Spectral representation with shape and amplitude kept separately.

    ``log_magnitude`` is the requested F tensor (32 non-DC bins for local
    STFT). ``shape`` has unit norm. ``energy`` contains three log band-energy
    values plus a constant anchor; the anchor ensures cosine distance remains
    sensitive to a common amplitude change.
    """

    log_magnitude: torch.Tensor
    shape: torch.Tensor
    energy: torch.Tensor

    def to(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FrequencyEncoding":
        """Move every spectral component together to prevent mixed devices."""

        return FrequencyEncoding(
            self.log_magnitude.to(device=device, dtype=dtype),
            self.shape.to(device=device, dtype=dtype),
            self.energy.to(device=device, dtype=dtype),
        )

    def feature(self, mode: FrequencyMode = "both") -> torch.Tensor:
        if mode == "shape":
            return self.shape
        if mode == "energy":
            return F.normalize(self.energy, dim=-1, eps=1e-12)
        if mode == "both":
            # Each block has unit norm, so neither silently dominates merely
            # because it has more coordinates.
            energy = F.normalize(self.energy, dim=-1, eps=1e-12)
            return torch.cat((self.shape, energy), dim=-1)
        raise ValueError(f"unknown frequency mode: {mode!r}")


def _encoding_from_amplitude(amplitude: torch.Tensor) -> FrequencyEncoding:
    if amplitude.ndim < 2:
        raise ValueError("amplitude must end in a frequency-bin dimension")
    log_magnitude = torch.log1p(amplitude.clamp_min(0))
    shape = F.normalize(log_magnitude, dim=-1, eps=1e-12)

    # Fixed, contiguous low/mid/high bands. tensor_split also handles FFT
    # ablations whose number of bins differs from 32.
    bands = torch.tensor_split(amplitude, 3, dim=-1)
    band_energy = torch.stack(
        [band.square().sum(dim=-1) for band in bands], dim=-1
    )
    log_band_energy = torch.log1p(band_energy)
    anchor = torch.ones_like(log_band_energy[..., :1])
    energy = torch.cat((log_band_energy, anchor), dim=-1)
    return FrequencyEncoding(log_magnitude, shape, energy)


def local_stft_encoding(
    windows: torch.Tensor,
    *,
    n_fft: int = 64,
    hop_length: int = 16,
    time_positions: int = 14,
    non_dc_bins: int = 32,
) -> FrequencyEncoding:
    """Encode detrended 224-point windows into 14 aligned local spectra.

    Frame centers are 7.5, 23.5, ..., 215.5, the centers of the 14 visual
    columns.  A 64-point frame therefore needs 24 samples of reflection
    padding on each boundary.  The DC bin is discarded and bins 1..32 are
    retained.  Input is ``[N, 224]`` or ``[224]`` and output keeps the leading
    dimensions, followed by ``[14, 32]``.
    """

    x = torch.as_tensor(windows)
    if not x.is_floating_point():
        x = x.float()
    if x.shape[-1] != time_positions * hop_length:
        raise ValueError(
            f"expected window length {time_positions * hop_length}, got {x.shape[-1]}"
        )
    if non_dc_bins > n_fft // 2:
        raise ValueError("non_dc_bins exceeds the non-DC rFFT bins")
    if n_fft < hop_length or (n_fft - hop_length) % 2:
        raise ValueError("n_fft-hop_length must be a non-negative even number")

    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    boundary = (n_fft - hop_length) // 2
    pad_mode = "reflect" if flat.shape[-1] > boundary else "replicate"
    padded = F.pad(flat, (boundary, boundary), mode=pad_mode)
    frames = padded.unfold(-1, n_fft, hop_length)
    if frames.shape[-2] != time_positions:
        raise RuntimeError(
            f"internal alignment error: produced {frames.shape[-2]} frames"
        )
    hann = torch.hann_window(
        n_fft, periodic=True, dtype=flat.dtype, device=flat.device
    )
    spectrum = torch.fft.rfft(frames * hann, n=n_fft, dim=-1)
    amplitude = spectrum.abs()[..., 1 : non_dc_bins + 1]
    amplitude = amplitude.reshape(*leading, time_positions, non_dc_bins)
    return _encoding_from_amplitude(amplitude)


def full_window_fft_encoding(
    windows: torch.Tensor,
    *,
    time_positions: int = 14,
    non_dc_bins: Optional[int] = None,
) -> FrequencyEncoding:
    """Full-window FFT ablation, repeated over the 14 visual columns.

    By default all 112 non-DC bins of a 224-point window are retained.  The
    repeated time dimension makes the representation usable through the same
    matching interface while intentionally removing within-window localization.
    """

    x = torch.as_tensor(windows)
    if not x.is_floating_point():
        x = x.float()
    window_length = x.shape[-1]
    available = window_length // 2
    bins = available if non_dc_bins is None else non_dc_bins
    if bins < 1 or bins > available:
        raise ValueError(f"non_dc_bins must be in [1, {available}]")
    hann = torch.hann_window(
        window_length, periodic=True, dtype=x.dtype, device=x.device
    )
    amplitude = torch.fft.rfft(x * hann, dim=-1).abs()[..., 1 : bins + 1]
    amplitude = amplitude.unsqueeze(-2).expand(*amplitude.shape[:-1], time_positions, bins)
    return _encoding_from_amplitude(amplitude)


def harmonic_aggregation_original(
    score_size: tuple[int, int, int], similarity: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Unmodified legacy aggregation, retained as an explicit control."""

    batch, height, width = score_size
    similarity = similarity.double()
    groups = mask.T.long()
    score = torch.zeros((batch, height * width), device=similarity.device).double()
    for index in range(height * width):
        patch_index = [
            bool(torch.isin(index + 1, group)) for group in groups
        ]
        patch_index_tensor = torch.tensor(patch_index, device=similarity.device)
        count = patch_index_tensor.sum().item()
        harmonic_sum = torch.sum(
            1.0 / similarity[:, patch_index_tensor], dim=-1
        )
        score[:, index] = count / harmonic_sum
    return score.view(batch, height, width)


def harmonic_aggregation_stable(
    score_size: tuple[int, int, int],
    similarity: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-12,
    legacy_index_offset: int = 1,
) -> torch.Tensor:
    """Numerically stable harmonic aggregation with legacy membership intact.

    Distances are clamped before reciprocal evaluation and uncovered cells are
    assigned zero.  ``legacy_index_offset=1`` preserves the existing ViT4TS
    mask lookup, separating the numerical fix from any indexing experiment.
    """

    batch, height, width = score_size
    if similarity.ndim != 2 or similarity.shape[0] != batch:
        raise ValueError("similarity must have shape [batch, mask_groups]")
    groups = mask.T.long().to(similarity.device)
    if groups.shape[0] != similarity.shape[1]:
        raise ValueError("mask group count does not match similarity")

    targets = (
        torch.arange(height * width, device=similarity.device)
        + legacy_index_offset
    )
    membership = (groups.unsqueeze(-1) == targets).any(dim=1)
    membership_f = membership.to(torch.float64)
    safe = torch.nan_to_num(
        similarity.to(torch.float64), nan=eps, posinf=1.0 / eps, neginf=eps
    ).clamp_min(eps)
    denominator = safe.reciprocal() @ membership_f
    count = membership_f.sum(dim=0).unsqueeze(0)
    score = torch.where(
        count > 0,
        count / denominator.clamp_min(eps),
        torch.zeros_like(denominator),
    )
    return score.view(batch, height, width)


class FrozenCLIPVisualEncoder(nn.Module):
    """Thin frozen interface around the existing ``CLIP_AD`` encoder."""

    def __init__(self, clip_encoder: nn.Module):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.clip_encoder.eval()
        for parameter in self.clip_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenCLIPVisualEncoder":
        # A frozen feature extractor must stay in inference mode even when a
        # surrounding experiment module receives train().
        super().train(False)
        self.clip_encoder.eval()
        return self

    @torch.inference_mode()
    def encode_image(self, images: torch.Tensor, patch_size: int = 16):
        return self.clip_encoder.encode_image(images, patch_size)


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Pairwise ``max(0, 1-cosine)`` for ``[..., D]`` and ``[R, D]``."""

    left_n = F.normalize(left, dim=-1, eps=1e-12)
    right_n = F.normalize(right, dim=-1, eps=1e-12)
    return (1.0 - left_n @ right_n.transpose(-1, -2)).clamp_min(0.0)


@dataclass(frozen=True)
class ReferenceBank:
    """Visual references plus an explicit visual-to-frequency correspondence."""

    visual: torch.Tensor
    frequency: FrequencyEncoding
    visual_columns: torch.Tensor
    frequency_index_for_visual: torch.Tensor
    visual_window_ids: torch.Tensor
    kind: Literal["independent_median", "paired"]
    time_positions: int = 14

    def __post_init__(self) -> None:
        count = self.visual.shape[0]
        if self.visual.ndim != 2:
            raise ValueError("visual reference must have shape [R, D]")
        for name in (
            "visual_columns",
            "frequency_index_for_visual",
            "visual_window_ids",
        ):
            value = getattr(self, name)
            if value.ndim != 1 or value.numel() != count:
                raise ValueError(f"{name} must have one entry per visual reference")
        freq_count = self.frequency.log_magnitude.shape[0]
        indices = self.frequency_index_for_visual
        if indices.numel() and (indices.min() < 0 or indices.max() >= freq_count):
            raise ValueError("frequency correspondence index is out of range")

    def frequency_feature(self, mode: FrequencyMode) -> torch.Tensor:
        return self.frequency.feature(mode)


def _flatten_frequency(encoding: FrequencyEncoding) -> FrequencyEncoding:
    return FrequencyEncoding(
        encoding.log_magnitude.flatten(0, -2),
        encoding.shape.flatten(0, -2),
        encoding.energy.flatten(0, -2),
    )


def _encoding_from_log_magnitude(log_magnitude: torch.Tensor) -> FrequencyEncoding:
    return _encoding_from_amplitude(torch.expm1(log_magnitude).clamp_min(0))


def build_independent_median_reference(
    visual_windows: torch.Tensor,
    frequency_windows: FrequencyEncoding,
    *,
    time_positions: int = 14,
) -> ReferenceBank:
    """Build separate visual and frequency medians over reference windows."""

    if visual_windows.ndim == 4 and visual_windows.shape[-2] == 1:
        visual_windows = visual_windows.squeeze(-2)
    if visual_windows.ndim != 3:
        raise ValueError("visual_windows must have shape [N, patches, D]")
    if frequency_windows.log_magnitude.ndim != 3:
        raise ValueError("frequency_windows must have shape [N, time, bins]")
    if visual_windows.shape[0] != frequency_windows.log_magnitude.shape[0]:
        raise ValueError("visual and frequency window counts differ")
    if frequency_windows.log_magnitude.shape[1] != time_positions:
        raise ValueError("frequency time positions do not match visual columns")
    patches = visual_windows.shape[1]
    if patches % time_positions:
        raise ValueError("visual patch count is not divisible by time positions")

    visual = visual_windows.median(dim=0).values
    log_magnitude = frequency_windows.log_magnitude.median(dim=0).values
    frequency = _encoding_from_log_magnitude(log_magnitude)
    columns = torch.arange(patches, device=visual.device) % time_positions
    frequency_indices = columns.to(log_magnitude.device)
    return ReferenceBank(
        visual=visual,
        frequency=frequency,
        visual_columns=columns,
        frequency_index_for_visual=frequency_indices,
        visual_window_ids=torch.full_like(columns, -1),
        kind="independent_median",
        time_positions=time_positions,
    )


def build_paired_reference(
    visual_windows: torch.Tensor,
    frequency_windows: FrequencyEncoding,
    *,
    time_positions: int = 14,
) -> ReferenceBank:
    """Keep a bounded library of real window pairs for the E4 control."""

    if visual_windows.ndim == 4 and visual_windows.shape[-2] == 1:
        visual_windows = visual_windows.squeeze(-2)
    if visual_windows.ndim != 3:
        raise ValueError("visual_windows must have shape [N, patches, D]")
    windows, patches, dimension = visual_windows.shape
    if frequency_windows.log_magnitude.shape[:2] != (windows, time_positions):
        raise ValueError("paired frequency windows do not align")
    if patches % time_positions:
        raise ValueError("visual patch count is not divisible by time positions")

    device = visual_windows.device
    patch_ids = torch.arange(patches, device=device).repeat(windows)
    window_ids = torch.arange(windows, device=device).repeat_interleave(patches)
    columns = patch_ids % time_positions
    frequency_indices = window_ids.to(columns.device) * time_positions + columns
    return ReferenceBank(
        visual=visual_windows.reshape(windows * patches, dimension),
        frequency=_flatten_frequency(frequency_windows),
        visual_columns=columns,
        frequency_index_for_visual=frequency_indices,
        visual_window_ids=window_ids,
        kind="paired",
        time_positions=time_positions,
    )


@dataclass(frozen=True)
class VisualCandidates:
    visual_distances: torch.Tensor
    reference_indices: torch.Tensor
    reference_columns: torch.Tensor
    reference_window_ids: torch.Tensor
    reference_frequency_indices: torch.Tensor


def robust_scale_calibration(
    visual_distances: torch.Tensor,
    frequency_distances: torch.Tensor,
    *,
    eps: float = 1e-8,
    maximum: float = 1e4,
) -> torch.Tensor:
    """Median-ratio calibration using only unlabeled distance statistics."""

    visual = visual_distances.detach().reshape(-1)
    frequency = frequency_distances.detach().reshape(-1)
    visual = visual[torch.isfinite(visual) & (visual > eps)]
    frequency = frequency[torch.isfinite(frequency) & (frequency > eps)]
    if visual.numel() == 0 or frequency.numel() == 0:
        return torch.ones((), dtype=visual_distances.dtype, device=visual_distances.device)
    factor = visual.median() / frequency.median().clamp_min(eps)
    return factor.clamp(min=1.0 / maximum, max=maximum).to(
        dtype=visual_distances.dtype, device=visual_distances.device
    )


class VisualFrequencyMatcher:
    """A/B/C/D and E3/E4 base-patch matching on cached features."""

    def __init__(
        self,
        reference: ReferenceBank,
        *,
        top_k: int = 5,
        frequency_weight: float = 0.3,
        frequency_mode: FrequencyMode = "both",
        calibration_factor: Optional[float | torch.Tensor] = None,
        vit4ts_output_factor: float = 0.5,
    ):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if frequency_weight < 0:
            raise ValueError("frequency_weight must be non-negative")
        self.reference = reference
        self.top_k = top_k
        self.frequency_weight = frequency_weight
        self.frequency_mode = frequency_mode
        self.calibration_factor = calibration_factor
        self.vit4ts_output_factor = vit4ts_output_factor

    @staticmethod
    def _visual_3d(visual: torch.Tensor) -> torch.Tensor:
        if visual.ndim == 4 and visual.shape[-2] == 1:
            visual = visual.squeeze(-2)
        if visual.ndim == 2:
            visual = visual.unsqueeze(0)
        if visual.ndim != 3:
            raise ValueError("query visual must have shape [B, patches, D]")
        return visual

    def visual_topk_candidates(
        self, query_visual: torch.Tensor, k: Optional[int] = None
    ) -> VisualCandidates:
        query = self._visual_3d(query_visual)
        reference = self.reference.visual.to(query.device, query.dtype)
        distances = cosine_distance(query, reference)
        selected_k = min(self.top_k if k is None else k, reference.shape[0])
        if selected_k < 1:
            raise ValueError("reference bank is empty")
        values, indices = distances.topk(selected_k, dim=-1, largest=False, sorted=True)

        def gather_metadata(value: torch.Tensor) -> torch.Tensor:
            return value.to(indices.device)[indices]

        return VisualCandidates(
            visual_distances=values,
            reference_indices=indices,
            reference_columns=gather_metadata(self.reference.visual_columns),
            reference_window_ids=gather_metadata(self.reference.visual_window_ids),
            reference_frequency_indices=gather_metadata(
                self.reference.frequency_index_for_visual
            ),
        )

    def stable_visual_score(self, query_visual: torch.Tensor) -> torch.Tensor:
        candidates = self.visual_topk_candidates(query_visual, k=1)
        return self.vit4ts_output_factor * candidates.visual_distances[..., 0]

    def _query_frequency_by_patch(
        self, query_visual: torch.Tensor, query_frequency: FrequencyEncoding
    ) -> torch.Tensor:
        query = self._visual_3d(query_visual)
        feature = query_frequency.feature(self.frequency_mode).to(
            query.device, query.dtype
        )
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        if feature.ndim != 3 or feature.shape[0] != query.shape[0]:
            raise ValueError("query frequency must have shape [B, time, features]")
        if feature.shape[1] != self.reference.time_positions:
            raise ValueError("query frequency positions do not match reference")
        columns = torch.arange(query.shape[1], device=feature.device) % feature.shape[1]
        return feature[:, columns, :]

    def _candidate_frequency_distances(
        self,
        query_visual: torch.Tensor,
        query_frequency: FrequencyEncoding,
        candidates: VisualCandidates,
    ) -> torch.Tensor:
        query = self._query_frequency_by_patch(query_visual, query_frequency)
        reference = self.reference.frequency_feature(self.frequency_mode).to(
            query.device, query.dtype
        )
        ref = reference[candidates.reference_frequency_indices.to(reference.device)]
        query_expanded = query.unsqueeze(-2)
        query_n = F.normalize(query_expanded, dim=-1, eps=1e-12)
        ref_n = F.normalize(ref, dim=-1, eps=1e-12)
        return (1.0 - (query_n * ref_n).sum(dim=-1)).clamp_min(0.0)

    def _scale(
        self, visual_distances: torch.Tensor, frequency_distances: torch.Tensor
    ) -> torch.Tensor:
        if self.calibration_factor is None:
            return robust_scale_calibration(visual_distances, frequency_distances)
        return torch.as_tensor(
            self.calibration_factor,
            dtype=visual_distances.dtype,
            device=visual_distances.device,
        )

    def frequency_only_score(
        self, query_visual: torch.Tensor, query_frequency: FrequencyEncoding
    ) -> torch.Tensor:
        """B: match each patch-column spectrum to all frequency references."""

        query = self._query_frequency_by_patch(query_visual, query_frequency)
        reference = self.reference.frequency_feature(self.frequency_mode).to(
            query.device, query.dtype
        )
        distances = cosine_distance(query, reference)
        best = distances.min(dim=-1).values
        # Calibrate B on all unlabeled frequency distances against all visual
        # distances, without consulting labels.
        visual_all = cosine_distance(
            self._visual_3d(query_visual),
            self.reference.visual.to(query.device, query.dtype),
        )
        scale = self._scale(visual_all, distances)
        return self.vit4ts_output_factor * scale * best

    def score_fusion(
        self, query_visual: torch.Tensor, query_frequency: FrequencyEncoding
    ) -> torch.Tensor:
        """C/E3: fixed fusion after independent visual and frequency minima."""

        visual = self.stable_visual_score(query_visual) / self.vit4ts_output_factor
        query = self._query_frequency_by_patch(query_visual, query_frequency)
        reference = self.reference.frequency_feature(self.frequency_mode).to(
            query.device, query.dtype
        )
        frequency_distances = cosine_distance(query, reference)
        frequency = frequency_distances.min(dim=-1).values
        visual_all = cosine_distance(
            self._visual_3d(query_visual),
            self.reference.visual.to(query.device, query.dtype),
        )
        scale = self._scale(visual_all, frequency_distances)
        return self.vit4ts_output_factor * (
            visual + self.frequency_weight * scale * frequency
        )

    def guided_score(
        self,
        query_visual: torch.Tensor,
        query_frequency: FrequencyEncoding,
        *,
        k: Optional[int] = None,
        frequency_weight: Optional[float] = None,
        return_candidates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, VisualCandidates, torch.Tensor, torch.Tensor]:
        """D: rerank visual Top-K using its exact reference correspondence."""

        weight = self.frequency_weight if frequency_weight is None else frequency_weight
        if weight < 0:
            raise ValueError("frequency_weight must be non-negative")
        candidates = self.visual_topk_candidates(query_visual, k=k)
        if weight == 0:
            # This branch intentionally avoids any frequency arithmetic and
            # returns the same first Top-K value used by stable_visual_score.
            score = self.vit4ts_output_factor * candidates.visual_distances[..., 0]
            if return_candidates:
                zeros = torch.zeros_like(candidates.visual_distances)
                one = torch.ones((), dtype=score.dtype, device=score.device)
                return score, candidates, zeros, one
            return score

        frequency = self._candidate_frequency_distances(
            query_visual, query_frequency, candidates
        )
        scale = self._scale(candidates.visual_distances, frequency)
        joint = candidates.visual_distances + weight * scale * frequency
        score = self.vit4ts_output_factor * joint.min(dim=-1).values
        if return_candidates:
            return score, candidates, frequency, scale
        return score

    def direct_distance_fusion(
        self, query_visual: torch.Tensor, query_frequency: FrequencyEncoding
    ) -> torch.Tensor:
        """E3: fuse corresponding distances before an unrestricted minimum."""

        return self.guided_score(
            query_visual,
            query_frequency,
            k=self.reference.visual.shape[0],
        )

    def score(
        self,
        method: Literal[
            "stable_visual", "frequency_only", "score_fusion", "guided", "direct"
        ],
        query_visual: torch.Tensor,
        query_frequency: Optional[FrequencyEncoding] = None,
    ) -> torch.Tensor:
        """Common A/B/C/D/E3 dispatch used by experiment runners."""

        if method == "stable_visual":
            return self.stable_visual_score(query_visual)
        if query_frequency is None:
            raise ValueError(f"{method} requires query_frequency")
        if method == "frequency_only":
            return self.frequency_only_score(query_visual, query_frequency)
        if method == "score_fusion":
            return self.score_fusion(query_visual, query_frequency)
        if method == "guided":
            return self.guided_score(query_visual, query_frequency)
        if method == "direct":
            return self.direct_distance_fusion(query_visual, query_frequency)
        raise ValueError(f"unknown matching method: {method!r}")


def fuse_point_scores(
    visual_scores: torch.Tensor,
    frequency_scores: torch.Tensor,
    *,
    frequency_weight: float = 0.3,
    calibration_factor: Optional[float | torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse complete A and B point-score vectors for variant C.

    Calibration is computed from unlabeled point scores after each branch has
    independently completed map upsampling, overlap stitching, and alignment.
    This keeps C distinct from the E3 patch-distance fusion.
    """

    if visual_scores.shape != frequency_scores.shape:
        raise ValueError("visual and frequency point scores must have the same shape")
    if frequency_weight < 0:
        raise ValueError("frequency_weight must be non-negative")
    if calibration_factor is None:
        scale = robust_scale_calibration(visual_scores, frequency_scores)
    else:
        scale = torch.as_tensor(
            calibration_factor,
            dtype=visual_scores.dtype,
            device=visual_scores.device,
        )
    return visual_scores + frequency_weight * scale * frequency_scores, scale


def _slice_encoding(
    encoding: FrequencyEncoding, index: int | slice | torch.Tensor
) -> FrequencyEncoding:
    return FrequencyEncoding(
        encoding.log_magnitude[index], encoding.shape[index], encoding.energy[index]
    )


def _publish_torch_no_replace(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically publish a tensor cache while refusing to replace a peer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class VFGViT4TSBackend:
    """Stateful end-to-end backend for VFG-ViT4TS inference.

    CLIP is loaded once per worker.  Each signal is rendered and encoded once,
    then all requested variants share the cached base/mid/large tokens.  The
    class never writes outside its caller-provided cache directory.
    """

    CACHE_VERSION = 1

    def __init__(
        self,
        config: Mapping[str, Any],
        device: str | torch.device,
        cache_dir: str | os.PathLike[str],
    ):
        self.config = dict(config)
        self.model_config = dict(config.get("model", {}))
        if str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir)

        self.window_size = int(self.model_config.get("window_size", 224))
        self.step_size = int(self.model_config.get("step_size", 56))
        self.patch_size = int(self.model_config.get("patch_size", 16))
        image_size = self.model_config.get("image_size", [224, 224])
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if (self.window_size, self.step_size, self.patch_size, self.image_size) != (
            224,
            56,
            16,
            (224, 224),
        ):
            raise ValueError("VFG first-stage protocol is fixed to 224/56, patch 16, image 224")

        # The repository's clip_vision module uses the historical ``models``
        # absolute import, so make src importable without changing that module.
        import sys

        src_dir = str(Path(__file__).resolve().parents[1])
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from models.clip_vision import CLIP_AD

        encoder = CLIP_AD(
            model_name="ViT-B-16", device=self.device, image_size=self.image_size
        )
        self.visual_encoder = FrozenCLIPVisualEncoder(encoder).to(self.device)

    @staticmethod
    def _safe_component(value: Any) -> str:
        text = str(value)
        return "".join(character if character.isalnum() or character in "-_." else "_" for character in text)

    def _input_digest(self, values: "Any") -> str:
        import numpy as np

        array = np.ascontiguousarray(values, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(array.view(np.uint8))
        digest.update(b"VFG-ViT4TS-cache-v1-CLIP-ViT-B-16-224-56")
        return digest.hexdigest()

    def _cache_path(self, context: Mapping[str, Any], digest: str) -> Path:
        dataset = self._safe_component(context.get("dataset", "unknown_dataset"))
        signal = self._safe_component(context.get("signal", "unknown_signal"))
        return self.cache_dir / dataset / signal / f"visual_{digest[:20]}.pt"

    def _render_images(self, visual_values: "Any") -> torch.Tensor:
        import numpy as np
        try:
            from src.preprocessing.preprocess import draw_windowed_images
        except ImportError:  # compatibility with the repository's src-on-path entrypoint
            from preprocessing.preprocess import draw_windowed_images

        with tempfile.TemporaryDirectory(prefix="vfg_render_") as directory:
            success = draw_windowed_images(
                base_series_id="series",
                save_path=directory,
                time_series=visual_values,
                time_points=np.arange(len(visual_values)),
                window_size=self.window_size,
                step_size=self.step_size,
                override=True,
                save_image=False,
                image_size=self.image_size,
                dpi=100,
                plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
            )
            if not success:
                raise RuntimeError("ViT4TS image rendering produced no windows")
            images = np.load(Path(directory) / "series_line_img.npy", allow_pickle=False)
            return torch.from_numpy(images).float()

    @torch.inference_mode()
    def _encode_images(self, images: torch.Tensor) -> dict[str, Any]:
        batch_size = int(self.model_config.get("batch_size", 20))
        collected: dict[str, list[torch.Tensor]] = {
            "large": [],
            "mid": [],
            "patch": [],
        }
        large_mask = None
        mid_mask = None
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(self.device)
            large, mid, patch, _, large_mask, mid_mask = self.visual_encoder.encode_image(
                batch, self.patch_size
            )
            collected["large"].append(large.detach().cpu())
            collected["mid"].append(mid.detach().cpu())
            collected["patch"].append(patch.detach().cpu())
        if large_mask is None or mid_mask is None:
            raise RuntimeError("CLIP encoder received an empty image batch")
        return {
            "cache_version": self.CACHE_VERSION,
            "large": torch.cat(collected["large"], dim=0),
            "mid": torch.cat(collected["mid"], dim=0),
            "patch": torch.cat(collected["patch"], dim=0),
            "large_mask": large_mask.detach().cpu(),
            "mid_mask": mid_mask.detach().cpu(),
        }

    def _validate_visual_cache(
        self,
        cached: Mapping[str, Any],
        *,
        digest: str,
        expected_windows: int,
        path: Path,
    ) -> None:
        if int(cached.get("cache_version", -1)) != self.CACHE_VERSION:
            raise RuntimeError(f"unsupported feature cache version: {path}")
        if cached.get("input_digest") != digest:
            raise RuntimeError(f"feature cache input digest mismatch: {path}")
        expected = {
            "large": (expected_windows, 144),
            "mid": (expected_windows, 169),
            "patch": (expected_windows, 196),
        }
        for name, leading_shape in expected.items():
            tensor = cached.get(name)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise RuntimeError(f"invalid {name} tensor in feature cache: {path}")
            actual = tensor.shape
            if name == "patch" and tensor.ndim == 4 and tensor.shape[-2] == 1:
                actual = (tensor.shape[0], tensor.shape[1], tensor.shape[-1])
            if len(actual) != 3 or tuple(actual[:2]) != leading_shape:
                raise RuntimeError(
                    f"invalid {name} shape {tuple(tensor.shape)} in feature cache: {path}"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite {name} tensor in feature cache: {path}")
        masks = {"large_mask": (9, 144), "mid_mask": (4, 169)}
        for name, expected_shape in masks.items():
            tensor = cached.get(name)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                raise RuntimeError(f"invalid {name} in feature cache: {path}")

    def _load_or_extract_visual(
        self, visual_values: "Any", context: Mapping[str, Any], digest: str
    ) -> tuple[dict[str, Any], bool]:
        cache_path = self._cache_path(context, digest)
        expected_windows = 1 + (len(visual_values) - self.window_size) // self.step_size
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            self._validate_visual_cache(
                cached,
                digest=digest,
                expected_windows=expected_windows,
                path=cache_path,
            )
            return cached, True

        images = self._render_images(visual_values)
        features = self._encode_images(images)
        features["input_digest"] = digest
        _publish_torch_no_replace(cache_path, features)
        if cache_path.is_file():
            # When a peer won publication, use its complete artifact.
            stored = torch.load(cache_path, map_location="cpu", weights_only=True)
            self._validate_visual_cache(
                stored,
                digest=digest,
                expected_windows=expected_windows,
                path=cache_path,
            )
            return stored, False
        self._validate_visual_cache(
            features,
            digest=digest,
            expected_windows=expected_windows,
            path=cache_path,
        )
        return features, False

    def _minimum_visual_distance(
        self, tokens: torch.Tensor, *, chunk_size: int = 32
    ) -> torch.Tensor:
        if tokens.ndim == 4 and tokens.shape[-2] == 1:
            tokens = tokens.squeeze(-2)
        reference = tokens.median(dim=0).values.to(self.device)
        pieces = []
        for start in range(0, tokens.shape[0], chunk_size):
            query = tokens[start : start + chunk_size].to(self.device)
            pieces.append((0.5 * cosine_distance(query, reference).min(dim=-1).values).cpu())
        return torch.cat(pieces, dim=0)

    def _stable_visual_components(
        self, features: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        large = self._minimum_visual_distance(features["large"])
        mid = self._minimum_visual_distance(features["mid"])
        patch = self._minimum_visual_distance(features["patch"])
        windows = patch.shape[0]
        large_map = harmonic_aggregation_stable(
            (windows, 14, 14), large, features["large_mask"]
        )
        mid_map = harmonic_aggregation_stable(
            (windows, 14, 14), mid, features["mid_mask"]
        )
        # Legacy ViT4TS harmonic aggregation returns float64.  Preserve that
        # dtype through scale averaging and interpolation; the patch component
        # is promoted exactly as in the original mixed-dtype expression.
        patch_map = patch.reshape(windows, 14, 14).to(torch.float64)
        return large_map, mid_map, patch_map

    def _score_matcher_chunks(
        self,
        matcher: VisualFrequencyMatcher,
        method: Literal["frequency_only", "guided", "direct"],
        visual: torch.Tensor,
        frequency: FrequencyEncoding,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        if matcher.calibration_factor is None:
            # Fit one scale for the complete unlabeled series.  Computing it
            # separately in each inference batch would make scores depend on
            # batch boundaries.  A deterministic bounded subsample controls
            # CPU memory for E3's unrestricted candidate set.
            visual_samples = []
            frequency_samples = []
            chunks = max(1, (visual.shape[0] + chunk_size - 1) // chunk_size)
            samples_per_chunk = max(1, 2_000_000 // chunks)
            candidate_count = (
                matcher.reference.visual.shape[0]
                if method == "direct"
                else matcher.top_k
            )
            for start in range(0, visual.shape[0], chunk_size):
                stop = min(start + chunk_size, visual.shape[0])
                visual_batch = visual[start:stop].to(self.device)
                frequency_batch = _slice_encoding(
                    frequency, slice(start, stop)
                ).to(self.device, visual_batch.dtype)
                candidates = matcher.visual_topk_candidates(
                    visual_batch, k=candidate_count
                )
                frequency_distance = matcher._candidate_frequency_distances(
                    visual_batch, frequency_batch, candidates
                )
                visual_flat = candidates.visual_distances.detach().cpu().reshape(-1)
                frequency_flat = frequency_distance.detach().cpu().reshape(-1)
                stride = max(1, (visual_flat.numel() + samples_per_chunk - 1) // samples_per_chunk)
                visual_samples.append(visual_flat[::stride])
                frequency_samples.append(frequency_flat[::stride])
            matcher.calibration_factor = float(
                robust_scale_calibration(
                    torch.cat(visual_samples), torch.cat(frequency_samples)
                )
            )

        scores = []
        for start in range(0, visual.shape[0], chunk_size):
            stop = min(start + chunk_size, visual.shape[0])
            visual_batch = visual[start:stop].to(self.device)
            frequency_batch = _slice_encoding(
                frequency, slice(start, stop)
            ).to(self.device, visual_batch.dtype)
            scores.append(matcher.score(method, visual_batch, frequency_batch).detach().cpu())
        return torch.cat(scores, dim=0)

    def _point_scores(self, maps: torch.Tensor, full_length: int) -> torch.Tensor:
        import numpy as np
        try:
            from src.models.model_utils import align_anomaly_vector, stitch_anomaly_maps
        except ImportError:  # compatibility with the repository's src-on-path entrypoint
            from models.model_utils import align_anomaly_vector, stitch_anomaly_maps

        maps = torch.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0).to(self.device)
        upsampled = F.interpolate(
            maps.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        stitched = stitch_anomaly_maps(
            upsampled.cpu().numpy(), self.window_size / self.step_size, 0.25
        )
        aligned = align_anomaly_vector(
            stitched,
            full_length,
            self.window_size,
            self.step_size,
            maps.shape[0],
        )
        return torch.from_numpy(np.asarray(aligned, dtype=np.float64))

    @staticmethod
    def _variant_mode(name: str, variant: Mapping[str, Any]) -> FrequencyMode:
        frequency = str(variant.get("frequency", ""))
        if name == "E2_shape_only" or "shape_only" in frequency:
            return "shape"
        if name == "E2_energy_only" or "energy_only" in frequency:
            return "energy"
        return "both"

    def predict_variants(
        self,
        data: "Any",
        variants: Mapping[str, Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Return aligned point scores for all requested variants."""

        import numpy as np
        from scipy.signal import detrend

        started = time.perf_counter()
        raw_values = np.asarray(data["value"], dtype=np.float64)
        timestamps = np.asarray(data["timestamp"])
        if raw_values.ndim != 1 or raw_values.size < self.window_size:
            raise ValueError("input must be one univariate series of at least 224 points")

        # One global detrend is shared.  Only the visual branch receives the
        # original global min-max transform; frequency windows keep amplitude.
        detrended = np.asarray(detrend(raw_values), dtype=np.float64)
        minimum = float(detrended.min())
        span = float(detrended.max() - minimum)
        visual_values = (detrended - minimum) / span if span > 0 else np.zeros_like(detrended)
        digest = self._input_digest(raw_values)
        features, cache_hit = self._load_or_extract_visual(visual_values, context, digest)

        numeric = torch.from_numpy(detrended)
        windows = numeric.unfold(0, self.window_size, self.step_size).contiguous()
        expected_windows = 1 + (len(raw_values) - self.window_size) // self.step_size
        if windows.shape[0] != expected_windows or features["patch"].shape[0] != expected_windows:
            raise RuntimeError("numeric and visual window counts differ")

        local_frequency = local_stft_encoding(windows)
        need_global = any(
            name == "E1_global_fft" or "whole_window_fft" in str(cfg.get("frequency", ""))
            for name, cfg in variants.items()
        )
        global_frequency = full_window_fft_encoding(windows) if need_global else None

        large_map, mid_map, stable_patch_map = self._stable_visual_components(features)
        stable_map = (large_map + mid_map + stable_patch_map) / 3.0
        stable_points = self._point_scores(stable_map, len(raw_values))

        patch_tokens = features["patch"]
        median_local_bank = build_independent_median_reference(
            patch_tokens, local_frequency
        )
        # B is a raw independent frequency branch; C calibrates complete A/B
        # point vectors below, after both have been independently aligned.
        frequency_matcher = VisualFrequencyMatcher(
            median_local_bank,
            frequency_mode="both",
            calibration_factor=1.0,
        )
        frequency_patch = self._score_matcher_chunks(
            frequency_matcher,
            "frequency_only",
            patch_tokens,
            local_frequency,
            chunk_size=32,
        )
        frequency_points = self._point_scores(
            frequency_patch.reshape(expected_windows, 14, 14), len(raw_values)
        )

        results: dict[str, dict[str, Any]] = {}
        computed_points: dict[str, torch.Tensor] = {
            "A_stable_vit4ts": stable_points,
            "B_frequency_only": frequency_points,
        }
        metadata_by_variant: dict[str, dict[str, Any]] = {
            "A_stable_vit4ts": {"matching": "stable_visual_multiscale"},
            "B_frequency_only": {"matching": "independent_frequency_min"},
        }

        for name, variant in variants.items():
            if name in computed_points or name == "C_score_fusion":
                continue
            use_global = name == "E1_global_fft" or "whole_window_fft" in str(
                variant.get("frequency", "")
            )
            encoding = global_frequency if use_global else local_frequency
            if encoding is None:
                raise RuntimeError("global FFT encoding was not prepared")
            paired = (
                name == "E4_paired_reference"
                or variant.get("frequency_reference") == "paired_real_window_library"
            )
            if paired:
                library_size = min(int(variant.get("reference_library_size", 128)), expected_windows)
                indices = torch.linspace(0, expected_windows - 1, library_size).round().long().unique()
                bank = build_paired_reference(
                    patch_tokens[indices], _slice_encoding(encoding, indices)
                )
                scoring_chunk = 1
            else:
                bank = build_independent_median_reference(patch_tokens, encoding)
                scoring_chunk = 32
            matcher = VisualFrequencyMatcher(
                bank,
                top_k=int(variant.get("top_k", 5)),
                frequency_weight=float(variant.get("lambda", 0.3)),
                frequency_mode=self._variant_mode(name, variant),
            )
            direct = (
                name == "E3_direct_distance"
                or variant.get("base_patch_score")
                == "all_reference_visual_frequency_joint_min"
            )
            method: Literal["guided", "direct"] = "direct" if direct else "guided"
            patch_score = self._score_matcher_chunks(
                matcher,
                method,
                patch_tokens,
                encoding,
                chunk_size=scoring_chunk,
            )
            patch_map = patch_score.reshape(expected_windows, 14, 14)
            combined_map = (large_map + mid_map + patch_map) / 3.0
            computed_points[name] = self._point_scores(combined_map, len(raw_values))
            metadata_by_variant[name] = {
                "matching": method,
                "reference": bank.kind,
                "frequency_mode": matcher.frequency_mode,
                "frequency_kind": "global_fft" if use_global else "local_stft",
                "top_k": matcher.top_k,
                "lambda": matcher.frequency_weight,
                "calibration_factor": float(matcher.calibration_factor),
                "paired_reference_windows": int(library_size) if paired else 0,
            }

        if "C_score_fusion" in variants:
            weight = float(variants["C_score_fusion"].get("lambda", 0.3))
            fused, scale = fuse_point_scores(
                stable_points, frequency_points, frequency_weight=weight
            )
            computed_points["C_score_fusion"] = fused
            metadata_by_variant["C_score_fusion"] = {
                "matching": "aligned_point_score_fusion",
                "lambda": weight,
                "calibration_factor": float(scale),
            }

        shared_elapsed = time.perf_counter() - started
        common = {
            "cache_hit": cache_hit,
            "cache_digest": digest[:20],
            "n_windows": expected_windows,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "clip": "OpenAI ViT-B/16 frozen",
            "harmonic": "stable",
            "timing_scope": "shared_backend_call",
        }
        for name in variants:
            if name not in computed_points:
                raise KeyError(f"unsupported VFG variant: {name}")
            metadata = dict(common)
            metadata.update(metadata_by_variant[name])
            results[name] = {
                "scores": computed_points[name].numpy(),
                "timestamps": timestamps,
                "inference_seconds": shared_elapsed,
                "metadata": metadata,
            }
        return results


def create_backend(
    config: Mapping[str, Any],
    device: str | torch.device,
    cache_dir: str | os.PathLike[str],
) -> VFGViT4TSBackend:
    """Runner factory for the frozen first-stage VFG-ViT4TS backend."""

    return VFGViT4TSBackend(config=config, device=device, cache_dir=cache_dir)
