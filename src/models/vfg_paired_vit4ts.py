"""Paired-window reference experiments for the first stage of ViT4TS.

The backend in this module is intentionally cache-only: it reads visual tokens
produced by :mod:`src.models.vfg_vit4ts` and never constructs or invokes CLIP.
Every paired method excludes the query window and all windows whose 224-point
spans overlap it (with stride 56 this is exactly ``abs(q-r) < 4``).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.models.model_utils import align_anomaly_vector, stitch_anomaly_maps
from src.models.vfg_vit4ts import (
    FrequencyEncoding,
    cosine_distance,
    full_window_fft_encoding,
    harmonic_aggregation_stable,
    local_stft_encoding,
    robust_scale_calibration,
)


WINDOW_SIZE = 224
STEP_SIZE = 56
GRID_SIZE = 14
PATCHES = GRID_SIZE * GRID_SIZE
OVERLAP_RADIUS = WINDOW_SIZE // STEP_SIZE
MAX_COMMON_WINDOWS = 32
PATCH_TOP_K = 5
GUIDED_WINDOWS = 5
DEFAULT_LAMBDA = 0.3


def _patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 4 and tokens.shape[-2] == 1:
        tokens = tokens.squeeze(-2)
    if tokens.ndim != 3 or tokens.shape[1] != PATCHES:
        raise ValueError("patch tokens must have shape [windows, 196, dim]")
    return tokens


def eligible_reference_mask(
    n_windows: int,
    *,
    overlap_radius: int = OVERLAP_RADIUS,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Return ``[query, reference]`` mask excluding all overlapping windows."""

    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if overlap_radius < 1:
        raise ValueError("overlap_radius must be positive")
    indices = torch.arange(n_windows, device=device)
    return (indices[:, None] - indices[None, :]).abs() >= overlap_radius


@dataclass(frozen=True)
class CommonWindowPool:
    """Ragged visual candidate windows represented by padded IDs and a mask."""

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

    def ids_for(self, query_window: int) -> torch.Tensor:
        return self.window_ids[query_window, self.valid[query_window]]


def select_common_visual_windows(
    visual_windows: torch.Tensor,
    *,
    max_candidates: int = MAX_COMMON_WINDOWS,
    overlap_radius: int = OVERLAP_RADIUS,
) -> CommonWindowPool:
    """Select at most 32 non-overlapping windows by mean-token cosine distance."""

    tokens = _patch_tokens(visual_windows)
    n_windows = tokens.shape[0]
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    eligible = eligible_reference_mask(
        n_windows, overlap_radius=overlap_radius, device=tokens.device
    )
    counts = eligible.sum(dim=1)
    if torch.any(counts == 0):
        failed = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            "paired reference requires at least one non-overlapping reference "
            f"for every query; none for query windows {failed}"
        )

    means = tokens.mean(dim=1)
    distances = cosine_distance(means.unsqueeze(0), means)[0]
    distances = distances.masked_fill(~eligible, torch.inf)
    width = min(max_candidates, int(counts.max().item()))
    values, ids = distances.topk(width, dim=-1, largest=False, sorted=True)
    valid = torch.isfinite(values)
    ids = ids.masked_fill(~valid, -1)
    return CommonWindowPool(ids, valid, values)


@dataclass(frozen=True)
class PatchCandidates:
    """Top visual Patch matches with their exact paired reference identity."""

    visual_distances: torch.Tensor
    reference_window_ids: torch.Tensor
    reference_patch_ids: torch.Tensor
    reference_columns: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.visual_distances.shape
        for value in (
            self.reference_window_ids,
            self.reference_patch_ids,
            self.reference_columns,
        ):
            if value.shape != shape:
                raise ValueError("candidate metadata shapes differ")
        if torch.any(self.reference_columns != self.reference_patch_ids % GRID_SIZE):
            raise ValueError("reference column is inconsistent with reference patch")


@dataclass(frozen=True)
class PairedVisualMatchCache:
    """Compact products of one full visual distance pass per query window."""

    paired_visual_scores: torch.Tensor
    paired_best_candidates: PatchCandidates
    topk_candidates: list[PatchCandidates]
    guided_visual_scores: Optional[torch.Tensor]
    guided_best_candidates: Optional[PatchCandidates]


def visual_patch_candidates(
    query_patches: torch.Tensor,
    reference_windows: torch.Tensor,
    reference_window_ids: torch.Tensor,
    *,
    top_k: int = PATCH_TOP_K,
) -> PatchCandidates:
    """Find visual Top-K over all patches in the selected real windows."""

    reference_windows = _patch_tokens(reference_windows)
    if query_patches.ndim != 2:
        raise ValueError("query_patches must have shape [196, dim]")
    if reference_window_ids.ndim != 1 or reference_window_ids.numel() < 1:
        raise ValueError("reference_window_ids must be a nonempty vector")
    selected = reference_windows[reference_window_ids]
    flat = selected.reshape(-1, selected.shape[-1])
    distances = cosine_distance(query_patches.unsqueeze(0), flat)[0]
    k = min(top_k, flat.shape[0])
    values, flat_ids = distances.topk(k, dim=-1, largest=False, sorted=True)
    pool_positions = flat_ids // PATCHES
    patch_ids = flat_ids % PATCHES
    window_ids = reference_window_ids[pool_positions]
    return PatchCandidates(values, window_ids, patch_ids, patch_ids % GRID_SIZE)


def paired_visual_patch_scores(
    visual_windows: torch.Tensor,
    pool: CommonWindowPool,
    *,
    output_factor: float = 0.5,
) -> torch.Tensor:
    """Variant B: pure visual nearest Patch in each query's common pool."""

    tokens = _patch_tokens(visual_windows)
    scores = []
    for query in range(tokens.shape[0]):
        candidates = visual_patch_candidates(
            tokens[query], tokens, pool.ids_for(query), top_k=1
        )
        scores.append(candidates.visual_distances[:, 0])
    return output_factor * torch.stack(scores)


def build_paired_visual_match_cache(
    visual_windows: torch.Tensor,
    pool: CommonWindowPool,
    *,
    guided_pool: Optional[GuidedWindowPool] = None,
    top_k: int = PATCH_TOP_K,
    output_factor: float = 0.5,
) -> PairedVisualMatchCache:
    """Compute each query/common-pool visual distance matrix exactly once.

    The full matrix is discarded after extracting B's minimum, C/E's visual
    Top-K identities, and (when requested) D's minimum restricted to its up to
    five FFT-selected windows.
    """

    tokens = _patch_tokens(visual_windows)
    paired_scores = []
    paired_best_windows = []
    paired_best_patches = []
    guided_scores = []
    guided_best_windows = []
    guided_best_patches = []
    topk = []
    for query in range(tokens.shape[0]):
        window_ids = pool.ids_for(query)
        references = tokens[window_ids]
        flat = references.reshape(-1, references.shape[-1])
        distances = cosine_distance(tokens[query].unsqueeze(0), flat)[0]

        paired_values, paired_flat_ids = distances.min(dim=-1)
        paired_scores.append(paired_values)
        paired_pool_positions = paired_flat_ids // PATCHES
        paired_patch_ids = paired_flat_ids % PATCHES
        paired_best_windows.append(window_ids[paired_pool_positions])
        paired_best_patches.append(paired_patch_ids)
        k = min(top_k, distances.shape[-1])
        values, flat_ids = distances.topk(k, dim=-1, largest=False, sorted=True)
        pool_positions = flat_ids // PATCHES
        patch_ids = flat_ids % PATCHES
        reference_ids = window_ids[pool_positions]
        topk.append(
            PatchCandidates(values, reference_ids, patch_ids, patch_ids % GRID_SIZE)
        )

        if guided_pool is not None:
            selected = guided_pool.ids_for(query)
            selected_mask = (window_ids[:, None] == selected[None, :]).any(dim=1)
            if not torch.any(selected_mask):
                raise RuntimeError("guided window selection escaped the common pool")
            guided_distances = distances.reshape(PATCHES, -1, PATCHES)[:, selected_mask]
            guided_values, guided_flat_ids = guided_distances.flatten(1).min(dim=-1)
            guided_scores.append(guided_values)
            guided_reference_windows = window_ids[selected_mask]
            guided_pool_positions = guided_flat_ids // PATCHES
            guided_patch_ids = guided_flat_ids % PATCHES
            guided_best_windows.append(guided_reference_windows[guided_pool_positions])
            guided_best_patches.append(guided_patch_ids)

    paired_values = torch.stack(paired_scores)
    paired_windows = torch.stack(paired_best_windows)
    paired_patches = torch.stack(paired_best_patches)
    paired_best = PatchCandidates(
        paired_values.unsqueeze(-1),
        paired_windows.unsqueeze(-1),
        paired_patches.unsqueeze(-1),
        (paired_patches % GRID_SIZE).unsqueeze(-1),
    )
    if guided_pool is not None:
        guided_values = torch.stack(guided_scores)
        guided_windows = torch.stack(guided_best_windows)
        guided_patches = torch.stack(guided_best_patches)
        guided_best = PatchCandidates(
            guided_values.unsqueeze(-1),
            guided_windows.unsqueeze(-1),
            guided_patches.unsqueeze(-1),
            (guided_patches % GRID_SIZE).unsqueeze(-1),
        )
    else:
        guided_values = None
        guided_best = None

    return PairedVisualMatchCache(
        paired_visual_scores=output_factor * paired_values,
        paired_best_candidates=paired_best,
        topk_candidates=topk,
        guided_visual_scores=(
            output_factor * guided_values
            if guided_values is not None
            else None
        ),
        guided_best_candidates=guided_best,
    )


def _frequency_feature(encoding: FrequencyEncoding) -> torch.Tensor:
    return encoding.feature("both")


def _candidate_frequency_distances(
    query_window: int,
    candidates: PatchCandidates,
    frequency: FrequencyEncoding,
) -> torch.Tensor:
    features = _frequency_feature(frequency)
    query_columns = torch.arange(PATCHES, device=features.device) % GRID_SIZE
    query = features[query_window, query_columns]
    reference = features[
        candidates.reference_window_ids.to(features.device),
        candidates.reference_columns.to(features.device),
    ]
    query_n = F.normalize(query.unsqueeze(1), dim=-1, eps=1e-12)
    reference_n = F.normalize(reference, dim=-1, eps=1e-12)
    return (1.0 - (query_n * reference_n).sum(dim=-1)).clamp_min(0.0)


def collect_joint_candidates(
    visual_windows: torch.Tensor,
    frequency: FrequencyEncoding,
    pool: CommonWindowPool,
    *,
    top_k: int = PATCH_TOP_K,
) -> tuple[list[PatchCandidates], list[torch.Tensor]]:
    """Collect paired visual candidates and exact-reference frequency distances."""

    tokens = _patch_tokens(visual_windows)
    candidates_by_window = []
    frequency_by_window = []
    for query in range(tokens.shape[0]):
        candidates = visual_patch_candidates(
            tokens[query], tokens, pool.ids_for(query), top_k=top_k
        )
        candidates_by_window.append(candidates)
        frequency_by_window.append(
            _candidate_frequency_distances(query, candidates, frequency)
        )
    return candidates_by_window, frequency_by_window


def joint_scores_from_visual_cache(
    cache: PairedVisualMatchCache,
    frequency: FrequencyEncoding,
    *,
    frequency_weight: float = DEFAULT_LAMBDA,
    calibration_factor: Optional[float | torch.Tensor] = None,
    output_factor: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add paired frequency distances to cached visual Top-5 candidates."""

    if frequency_weight < 0:
        raise ValueError("frequency_weight must be non-negative")
    if frequency_weight == 0:
        one = torch.ones(
            (),
            dtype=cache.paired_visual_scores.dtype,
            device=cache.paired_visual_scores.device,
        )
        return cache.paired_visual_scores, one
    spectral = [
        _candidate_frequency_distances(query, candidates, frequency)
        for query, candidates in enumerate(cache.topk_candidates)
    ]
    if calibration_factor is None:
        visual_flat = torch.cat(
            [candidate.visual_distances.reshape(-1) for candidate in cache.topk_candidates]
        )
        frequency_flat = torch.cat([distance.reshape(-1) for distance in spectral])
        scale = robust_scale_calibration(visual_flat, frequency_flat)
    else:
        scale = torch.as_tensor(
            calibration_factor,
            dtype=cache.paired_visual_scores.dtype,
            device=cache.paired_visual_scores.device,
        )
    score = output_factor * torch.stack(
        [
            (candidate.visual_distances + frequency_weight * scale * distance)
            .min(dim=-1)
            .values
            for candidate, distance in zip(cache.topk_candidates, spectral)
        ]
    )
    return score, scale


def paired_joint_patch_scores(
    visual_windows: torch.Tensor,
    frequency: FrequencyEncoding,
    pool: CommonWindowPool,
    *,
    frequency_weight: float = DEFAULT_LAMBDA,
    calibration_factor: Optional[float | torch.Tensor] = None,
    output_factor: float = 0.5,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, list[PatchCandidates]]:
    """Variants C/E: joint distance over paired visual Top-5 candidates.

    One unlabeled sequence-level scale is used for all windows.  The explicit
    zero-weight branch performs the same pure visual minimum as variant B and
    therefore returns it bit-for-bit.
    """

    if frequency_weight < 0:
        raise ValueError("frequency_weight must be non-negative")
    tokens = _patch_tokens(visual_windows)
    if frequency_weight == 0:
        score = paired_visual_patch_scores(tokens, pool, output_factor=output_factor)
        one = torch.ones((), dtype=score.dtype, device=score.device)
        if return_details:
            candidates, _ = collect_joint_candidates(tokens, frequency, pool)
            return score, one, candidates
        return score

    candidates, frequency_distances = collect_joint_candidates(tokens, frequency, pool)
    if calibration_factor is None:
        visual_all = torch.cat([item.visual_distances.reshape(-1) for item in candidates])
        frequency_all = torch.cat([item.reshape(-1) for item in frequency_distances])
        scale = robust_scale_calibration(visual_all, frequency_all)
    else:
        scale = torch.as_tensor(
            calibration_factor,
            dtype=tokens.dtype,
            device=tokens.device,
        )
    scores = []
    for visual, spectral in zip(candidates, frequency_distances):
        joint = visual.visual_distances + frequency_weight * scale * spectral
        scores.append(joint.min(dim=-1).values)
    score = output_factor * torch.stack(scores)
    if return_details:
        return score, scale, candidates
    return score


@dataclass(frozen=True)
class GuidedWindowPool:
    """Ragged frequency-selected windows inside the common visual pool."""

    window_ids: torch.Tensor
    valid: torch.Tensor
    frequency_distances: torch.Tensor

    def __post_init__(self) -> None:
        if self.window_ids.ndim != 2:
            raise ValueError("window_ids must have shape [query, candidates]")
        if self.valid.shape != self.window_ids.shape:
            raise ValueError("valid shape differs from window_ids")
        if self.frequency_distances.shape != self.window_ids.shape:
            raise ValueError("frequency_distances shape differs from window_ids")
        if not torch.equal(self.valid, self.window_ids >= 0):
            raise ValueError("valid mask and padded window IDs disagree")
        if torch.any(self.valid.sum(dim=1) == 0):
            raise ValueError("every query must have a guided reference window")
        if not torch.isinf(self.frequency_distances[~self.valid]).all():
            raise ValueError("padded frequency distances must be infinite")

    def ids_for(self, query_window: int) -> torch.Tensor:
        return self.window_ids[query_window, self.valid[query_window]]


def select_fft_guided_windows(
    frequency: FrequencyEncoding,
    common_pool: CommonWindowPool,
    *,
    selected_windows: int = GUIDED_WINDOWS,
) -> GuidedWindowPool:
    """Select up to five FFT-nearest windows from the common visual pool.

    Short series can have fewer than ``selected_windows`` eligible references
    for central queries. Such rows retain every available non-overlapping
    reference and are padded with ``-1`` IDs, an explicit validity mask, and
    infinite distances.
    """

    if selected_windows < 1:
        raise ValueError("selected_windows must be positive")
    feature = _frequency_feature(frequency)[:, 0]
    rows_ids = []
    rows_values = []
    for query in range(feature.shape[0]):
        ids = common_pool.ids_for(query)
        if ids.numel() < 1:
            raise ValueError(
                "FFT-guided matching requires a non-overlapping candidate; "
                f"query {query} has none"
            )
        distances = cosine_distance(feature[query : query + 1].unsqueeze(0), feature[ids])[0, 0]
        count = min(selected_windows, ids.numel())
        values, positions = distances.topk(
            count, largest=False, sorted=True
        )
        chosen = ids[positions]
        if count < selected_windows:
            padding = selected_windows - count
            chosen = torch.cat([chosen, chosen.new_full((padding,), -1)])
            values = torch.cat([values, values.new_full((padding,), torch.inf)])
        rows_ids.append(chosen)
        rows_values.append(values)
    window_ids = torch.stack(rows_ids)
    values = torch.stack(rows_values)
    return GuidedWindowPool(window_ids, window_ids >= 0, values)


def fft_guided_visual_patch_scores(
    visual_windows: torch.Tensor,
    frequency: FrequencyEncoding,
    common_pool: CommonWindowPool,
    *,
    selected_windows: int = GUIDED_WINDOWS,
    output_factor: float = 0.5,
    return_pool: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, GuidedWindowPool]:
    """Variant D: FFT selects real windows, then pure visual Patch NN scores."""

    tokens = _patch_tokens(visual_windows)
    guided = select_fft_guided_windows(
        frequency, common_pool, selected_windows=selected_windows
    )
    scores = []
    for query in range(tokens.shape[0]):
        candidates = visual_patch_candidates(
            tokens[query], tokens, guided.ids_for(query), top_k=1
        )
        scores.append(candidates.visual_distances[:, 0])
    result = output_factor * torch.stack(scores)
    if return_pool:
        return result, guided
    return result


def stable_median_patch_scores(visual_windows: torch.Tensor) -> torch.Tensor:
    """Variant A base Patch score, identical to stable cached ViT4TS."""

    tokens = _patch_tokens(visual_windows)
    reference = tokens.median(dim=0).values
    return 0.5 * cosine_distance(tokens, reference).min(dim=-1).values


def stable_scale_map(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Stable original scale score using the all-window median reference."""

    tokens = tokens.squeeze(-2) if tokens.ndim == 4 and tokens.shape[-2] == 1 else tokens
    reference = tokens.median(dim=0).values.to(device)
    pieces = []
    for start in range(0, tokens.shape[0], 32):
        query = tokens[start : start + 32].to(device)
        pieces.append((0.5 * cosine_distance(query, reference).min(dim=-1).values).cpu())
    distances = torch.cat(pieces)
    return harmonic_aggregation_stable(
        (tokens.shape[0], GRID_SIZE, GRID_SIZE), distances, mask
    )


def point_scores_from_maps(
    maps: torch.Tensor,
    *,
    full_length: int,
    device: torch.device,
) -> np.ndarray:
    maps = torch.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0).to(device)
    upsampled = F.interpolate(
        maps.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(1)
    stitched = stitch_anomaly_maps(upsampled.cpu().numpy(), 4.0, 0.25)
    return align_anomaly_vector(
        stitched,
        full_length,
        WINDOW_SIZE,
        STEP_SIZE,
        maps.shape[0],
    )


class PairedVFGBackend:
    """Cache-only backend for A/B/C/D/E paired-reference comparisons."""

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

    @staticmethod
    def _safe_component(value: Any) -> str:
        return "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in str(value)
        )

    @staticmethod
    def _input_digest(values: np.ndarray) -> str:
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(array.view(np.uint8))
        digest.update(b"VFG-ViT4TS-cache-v1-CLIP-ViT-B-16-224-56")
        return digest.hexdigest()

    def _cache_path(self, context: Mapping[str, Any], digest: str) -> Path:
        dataset = self._safe_component(context.get("dataset", "unknown_dataset"))
        signal = self._safe_component(context.get("signal", "unknown_signal"))
        return self.cache_dir / dataset / signal / f"visual_{digest[:20]}.pt"

    def _clock(self) -> float:
        """GPU-aware timestamp for honest incremental timing."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _load_visual_cache(
        self,
        raw_values: np.ndarray,
        context: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        digest = self._input_digest(raw_values)
        path = self._cache_path(context, digest)
        if not path.is_file():
            raise FileNotFoundError(
                "paired backend is cache-only; expected existing visual cache "
                f"{path}"
            )
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if int(cached.get("cache_version", -1)) != self.CACHE_VERSION:
            raise RuntimeError(f"unsupported visual cache version: {path}")
        if cached.get("input_digest") != digest:
            raise RuntimeError(f"visual cache digest mismatch: {path}")
        expected_windows = 1 + (raw_values.size - WINDOW_SIZE) // STEP_SIZE
        shapes = {
            "large": (expected_windows, 144),
            "mid": (expected_windows, 169),
            "patch": (expected_windows, 196),
        }
        for name, expected in shapes.items():
            tensor = cached.get(name)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise RuntimeError(f"invalid {name} in visual cache: {path}")
            actual = tensor.squeeze(-2) if name == "patch" and tensor.ndim == 4 else tensor
            if actual.ndim != 3 or tuple(actual.shape[:2]) != expected:
                raise RuntimeError(f"invalid {name} shape in visual cache: {path}")
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite {name} in visual cache: {path}")
        for name, expected in {"large_mask": (9, 144), "mid_mask": (4, 169)}.items():
            if not isinstance(cached.get(name), torch.Tensor) or tuple(cached[name].shape) != expected:
                raise RuntimeError(f"invalid {name} in visual cache: {path}")
        return cached, digest

    @staticmethod
    def _variant_letter(name: str, config: Mapping[str, Any]) -> Literal["A", "B", "C", "D", "E"]:
        explicit = str(config.get("paired_variant", config.get("variant", ""))).upper()
        if explicit in {"A", "B", "C", "D", "E"}:
            return explicit  # type: ignore[return-value]
        letter = name.split("_", 1)[0].upper()
        if letter not in {"A", "B", "C", "D", "E"}:
            raise KeyError(f"cannot infer paired variant A-E from {name!r}")
        return letter  # type: ignore[return-value]

    def predict_variants(
        self,
        data: Any,
        variants: Mapping[str, Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        from scipy.signal import detrend

        raw_values = np.asarray(data["value"], dtype=np.float64)
        timestamps = np.asarray(data["timestamp"])
        if raw_values.ndim != 1 or raw_values.size < WINDOW_SIZE:
            raise ValueError("input must be one univariate series of at least 224 points")
        features, digest = self._load_visual_cache(raw_values, context)
        patch_cpu = _patch_tokens(features["patch"])
        n_windows = patch_cpu.shape[0]
        letters = {
            name: self._variant_letter(name, variant)
            for name, variant in variants.items()
        }
        if any(letter != "A" for letter in letters.values()) and n_windows < 8:
            raise ValueError(
                "paired B-E variants require at least 8 windows (616 points) so every "
                "query has a non-overlapping reference"
            )
        stable_started = self._clock()
        large_map = stable_scale_map(
            features["large"], features["large_mask"], device=self.device
        )
        mid_map = stable_scale_map(
            features["mid"], features["mid_mask"], device=self.device
        )
        stable_patch = stable_median_patch_scores(patch_cpu.to(self.device)).cpu()
        stable_patch_map = stable_patch.reshape(n_windows, GRID_SIZE, GRID_SIZE).to(torch.float64)
        stable_seconds = self._clock() - stable_started

        paired_requested = any(letter != "A" for letter in letters.values())
        if paired_requested:
            patch = patch_cpu.to(self.device)
            pool_started = self._clock()
            common_pool = select_common_visual_windows(patch)
            pool_seconds = self._clock() - pool_started
        else:
            patch = patch_cpu.to(self.device)
            common_pool = None
            pool_seconds = 0.0

        detrended = torch.from_numpy(np.asarray(detrend(raw_values), dtype=np.float64))
        windows = detrended.unfold(0, WINDOW_SIZE, STEP_SIZE).contiguous()
        need_fft = any(letter in {"C", "D"} for letter in letters.values())
        need_stft = any(letter == "E" for letter in letters.values())
        fft = None
        stft = None
        fft_seconds = 0.0
        stft_seconds = 0.0
        if need_fft:
            started = self._clock()
            fft = full_window_fft_encoding(windows).to(self.device, patch.dtype)
            fft_seconds = self._clock() - started
        if need_stft:
            started = self._clock()
            stft = local_stft_encoding(windows).to(self.device, patch.dtype)
            stft_seconds = self._clock() - started

        guided_pool = None
        guided_selection_seconds = 0.0
        guided_min_selected = 0
        guided_max_selected = 0
        if any(letter == "D" for letter in letters.values()):
            assert common_pool is not None and fft is not None
            started = self._clock()
            guided_pool = select_fft_guided_windows(fft, common_pool)
            guided_selection_seconds = self._clock() - started
            guided_counts = guided_pool.valid.sum(dim=1)
            guided_min_selected = int(guided_counts.min().item())
            guided_max_selected = int(guided_counts.max().item())

        visual_cache = None
        visual_matching_seconds = 0.0
        if paired_requested:
            assert common_pool is not None
            started = self._clock()
            visual_cache = build_paired_visual_match_cache(
                patch, common_pool, guided_pool=guided_pool
            )
            visual_matching_seconds = self._clock() - started

        results: dict[str, dict[str, Any]] = {}
        for name, variant in variants.items():
            letter = letters[name]
            incremental_frequency = 0.0
            calibration = None
            if letter == "A":
                base = stable_patch
                incremental_matching = stable_seconds
                semantics = "stable_all_window_median"
            elif letter == "B":
                assert visual_cache is not None
                base = visual_cache.paired_visual_scores
                incremental_matching = pool_seconds + visual_matching_seconds
                semantics = "paired_visual_common_window_pool"
            elif letter == "C":
                assert visual_cache is not None and fft is not None
                started = self._clock()
                weight = float(variant.get("lambda", DEFAULT_LAMBDA))
                base, scale = joint_scores_from_visual_cache(
                    visual_cache,
                    fft,
                    frequency_weight=weight,
                )
                incremental_matching = self._clock() - started
                incremental_frequency = fft_seconds
                calibration = float(scale)
                semantics = "paired_whole_fft_joint_top5"
            elif letter == "D":
                assert visual_cache is not None
                assert visual_cache.guided_visual_scores is not None
                base = visual_cache.guided_visual_scores
                incremental_matching = guided_selection_seconds
                incremental_frequency = fft_seconds
                semantics = "fft_guided_up_to_five_windows_then_visual_nn"
            else:
                assert visual_cache is not None and stft is not None
                started = self._clock()
                weight = float(variant.get("lambda", DEFAULT_LAMBDA))
                base, scale = joint_scores_from_visual_cache(
                    visual_cache,
                    stft,
                    frequency_weight=weight,
                )
                incremental_matching = self._clock() - started
                incremental_frequency = stft_seconds
                calibration = float(scale)
                semantics = "paired_local_stft_joint_top5"

            base_map = base.cpu().reshape(n_windows, GRID_SIZE, GRID_SIZE).to(torch.float64)
            final_map = (large_map + mid_map + base_map) / 3.0
            recovery_started = self._clock()
            scores = point_scores_from_maps(
                final_map, full_length=raw_values.size, device=self.device
            )
            recovery_seconds = self._clock() - recovery_started
            metadata = {
                "paired_variant": letter,
                "matching": semantics,
                "cache_only": True,
                "cache_digest": digest[:20],
                "n_windows": n_windows,
                "overlap_exclusion_window_index_radius": OVERLAP_RADIUS,
                "max_common_visual_windows": MAX_COMMON_WINDOWS,
                "fft_guided_selected_windows": GUIDED_WINDOWS if letter == "D" else 0,
                "fft_guided_target_windows": GUIDED_WINDOWS if letter == "D" else 0,
                "fft_guided_min_selected_windows": (
                    guided_min_selected if letter == "D" else 0
                ),
                "fft_guided_max_selected_windows": (
                    guided_max_selected if letter == "D" else 0
                ),
                "shared_visual_pool_seconds": pool_seconds,
                "shared_visual_patch_distance_seconds": visual_matching_seconds,
                "incremental_matching_seconds": incremental_matching,
                "incremental_frequency_seconds": incremental_frequency,
                "incremental_guided_selection_seconds": (
                    guided_selection_seconds if letter == "D" else 0.0
                ),
                "stable_shared_seconds": stable_seconds,
                "score_recovery_seconds": recovery_seconds,
                "timing_scope": "variant_incremental_cache_only",
            }
            if calibration is not None:
                metadata["calibration_factor"] = calibration
                metadata["lambda"] = float(variant.get("lambda", DEFAULT_LAMBDA))
            results[name] = {
                "scores": np.asarray(scores, dtype=np.float64),
                "timestamps": timestamps,
                "inference_seconds": (
                    incremental_matching
                    + incremental_frequency
                    + recovery_seconds
                ),
                "metadata": metadata,
            }
        return results


def create_backend(
    config: Mapping[str, Any],
    device: str | torch.device,
    cache_dir: str | os.PathLike[str],
) -> PairedVFGBackend:
    """Create a cache-only paired-reference backend; CLIP is never loaded."""

    return PairedVFGBackend(config=config, device=device, cache_dir=cache_dir)


__all__ = [
    "CommonWindowPool",
    "GuidedWindowPool",
    "PATCH_TOP_K",
    "PairedVisualMatchCache",
    "PairedVFGBackend",
    "PatchCandidates",
    "build_paired_visual_match_cache",
    "create_backend",
    "eligible_reference_mask",
    "fft_guided_visual_patch_scores",
    "joint_scores_from_visual_cache",
    "paired_joint_patch_scores",
    "paired_visual_patch_scores",
    "select_common_visual_windows",
    "select_fft_guided_windows",
    "stable_median_patch_scores",
    "visual_patch_candidates",
]
