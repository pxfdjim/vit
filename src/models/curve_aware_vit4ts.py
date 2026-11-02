"""Cache-only curve-aware first-stage ViT4TS comparisons.

The large and mid visual scales always use the stable original final-layer
reference.  Only the 14x14 base Patch branch changes.  Every paired variant
shares one candidate-window pool selected from final-layer window means.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.models.curve_aware_clip import (
    expected_window_count,
    input_digest,
    intermediate_cache_path,
    load_final_tokens,
    validate_intermediate_cache,
)
from src.models.curve_aware_sampling import (
    CurveCoordinates,
    bilinear_curve_features,
    column_mean_features,
    gaussian_curve_features,
    paired_time_feature_scores,
)
from src.models.vfg_paired_vit4ts import (
    GRID_SIZE,
    MAX_COMMON_WINDOWS,
    OVERLAP_RADIUS,
    _patch_tokens,
    paired_visual_patch_scores,
    point_scores_from_maps,
    select_common_visual_windows,
    stable_median_patch_scores,
    stable_scale_map,
)


def _pool_digest(window_ids: torch.Tensor, valid: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(window_ids.cpu().numpy()).view(np.uint8))
    digest.update(np.ascontiguousarray(valid.cpu().numpy()).view(np.uint8))
    return digest.hexdigest()[:20]


def _column_map(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != GRID_SIZE:
        raise ValueError("column scores must have shape [windows, 14]")
    return scores[:, None, :].expand(-1, GRID_SIZE, -1)


class CurveAwareViT4TSBackend:
    """Read immutable final/block-6 caches and score all nine variants."""

    def __init__(
        self,
        config: Mapping[str, Any],
        device: str | torch.device,
        cache_dir: str | Path,
    ) -> None:
        self.config = dict(config)
        self.model_config = dict(config.get("model", {}))
        self.cache_config = dict(config.get("cache", {}))
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir)
        roots = self.cache_config.get("final_cache_roots", [])
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.final_cache_roots = [Path(root) for root in roots]
        if not self.final_cache_roots:
            raise ValueError("cache.final_cache_roots must contain reusable final caches")
        fixed = (
            int(self.model_config.get("window_size", 224)),
            int(self.model_config.get("step_size", 56)),
            tuple(self.model_config.get("image_size", [224, 224])),
            int(self.model_config.get("patch_size", 16)),
        )
        if fixed != (224, 56, (224, 224), 16):
            raise ValueError("curve-aware protocol is fixed to 224/56, image 224, patch 16")

    def _clock(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _load_caches(
        self, raw: np.ndarray, context: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], str, Path, Path]:
        digest = input_digest(raw)
        n_windows = expected_window_count(raw.size)
        _, final_path = load_final_tokens(
            self.final_cache_roots, context, digest, n_windows
        )
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        curve_path = intermediate_cache_path(self.cache_dir, context, digest)
        if not curve_path.is_file():
            raise FileNotFoundError(f"missing block-6 curve cache: {curve_path}")
        curve = torch.load(curve_path, map_location="cpu", weights_only=True)
        validate_intermediate_cache(
            curve, digest=digest, expected_windows=n_windows, path=curve_path
        )
        for name, shape in {
            "large": (n_windows, 144),
            "mid": (n_windows, 169),
            "large_mask": (9, 144),
            "mid_mask": (4, 169),
        }.items():
            tensor = final.get(name)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape[:2]) != shape:
                raise RuntimeError(f"invalid {name} in final cache: {final_path}")
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite {name} in final cache: {final_path}")
        return final, curve, digest, final_path, curve_path

    @staticmethod
    def _coordinates(curve: Mapping[str, Any]) -> CurveCoordinates:
        pixel = curve["curve_pixel_xy"].cpu().numpy()
        grid = curve["curve_grid_xy"].cpu().numpy()
        return CurveCoordinates(
            pixel_x=pixel[..., 0], pixel_y=pixel[..., 1],
            patch_x=grid[..., 0], patch_y=grid[..., 1],
        )

    @staticmethod
    def _variant_spec(name: str, variant: Mapping[str, Any]) -> tuple[str, str, str]:
        family = str(variant.get("family", name.split("_", 1)[0])).upper()
        sampling = str(variant.get("base_sampling", ""))
        layer = str(variant.get("layer", "final"))
        if family not in {"A", "B", "C", "D", "E"}:
            raise KeyError(f"unknown curve-aware family: {family}")
        if layer not in {"final", "mid"}:
            raise KeyError(f"unknown visual layer: {layer}")
        return family, sampling, layer

    def predict_variants(
        self,
        data: Any,
        variants: Mapping[str, Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        raw = np.asarray(data["value"], dtype=np.float64)
        timestamps = np.asarray(data["timestamp"])
        if raw.ndim != 1 or raw.size < 224 or not np.isfinite(raw).all():
            raise ValueError("input must be one finite univariate series of at least 224 points")
        final, curve, digest, final_path, curve_path = self._load_caches(raw, context)
        final_patch_cpu = _patch_tokens(final["patch"])
        mid_patch_cpu = _patch_tokens(curve["intermediate_patch"])
        if final_patch_cpu.shape != mid_patch_cpu.shape:
            raise RuntimeError("final and block-6 Patch token shapes differ")
        n_windows = final_patch_cpu.shape[0]
        if any(self._variant_spec(name, value)[0] != "A" for name, value in variants.items()) and n_windows < 8:
            raise ValueError("paired variants require at least eight windows")

        stable_started = self._clock()
        large_map = stable_scale_map(final["large"], final["large_mask"], device=self.device)
        mid_map = stable_scale_map(final["mid"], final["mid_mask"], device=self.device)
        stable_patch = stable_median_patch_scores(final_patch_cpu.to(self.device))
        stable_patch_map = stable_patch.reshape(n_windows, GRID_SIZE, GRID_SIZE).to(torch.float64)
        stable_seconds = self._clock() - stable_started

        pool_started = self._clock()
        final_patch = final_patch_cpu.to(self.device)
        common_pool = select_common_visual_windows(final_patch)
        pool_seconds = self._clock() - pool_started
        pool_counts = common_pool.valid.sum(dim=1)
        pool_digest = _pool_digest(common_pool.window_ids, common_pool.valid)
        coordinates = self._coordinates(curve)
        tokens_by_layer = {
            "final": final_patch,
            "mid": mid_patch_cpu.to(self.device),
        }

        # Each representation is computed once and reused by its requested variant.
        representation_cache: dict[tuple[str, str], torch.Tensor] = {}
        results: dict[str, dict[str, Any]] = {}
        for name, variant in variants.items():
            family, sampling, layer = self._variant_spec(name, variant)
            matching_started = self._clock()
            if family == "A":
                base_map = stable_patch_map
                semantics = "stable_all_window_median_final_patch"
                layer = "final"
            elif sampling == "full_2d_patch":
                base = paired_visual_patch_scores(tokens_by_layer[layer], common_pool)
                base_map = base.reshape(n_windows, GRID_SIZE, GRID_SIZE).to(torch.float64)
                semantics = "paired_full_2d_patch"
            else:
                key = (layer, sampling)
                if key not in representation_cache:
                    tokens = tokens_by_layer[layer]
                    if sampling == "column_mean":
                        representation_cache[key] = column_mean_features(tokens)
                    elif sampling == "curve_bilinear":
                        representation_cache[key] = bilinear_curve_features(tokens, coordinates)
                    elif sampling == "curve_gaussian":
                        sampling_config = self.model_config.get("curve_sampling", {})
                        sigma = float(sampling_config.get("gaussian_sigma_patch", 1.0))
                        radius = sampling_config.get("gaussian_radius_patch", 3.0)
                        radius = None if radius is None else float(radius)
                        representation_cache[key] = gaussian_curve_features(
                            tokens,
                            coordinates,
                            sigma_patches=sigma,
                            radius_patches=radius,
                        )
                    else:
                        raise KeyError(f"unknown base sampling: {sampling}")
                column_scores = paired_time_feature_scores(
                    representation_cache[key], common_pool.window_ids, common_pool.valid
                )
                base_map = _column_map(column_scores).to(torch.float64)
                semantics = f"paired_{sampling}_ordered_time_features"
            matching_seconds = self._clock() - matching_started

            final_map = (large_map + mid_map + base_map.cpu()) / 3.0
            recovery_started = self._clock()
            scores = point_scores_from_maps(
                final_map, full_length=raw.size, device=self.device
            )
            recovery_seconds = self._clock() - recovery_started
            paired = family != "A"
            metadata = {
                "family": family,
                "visual_layer": layer,
                "base_sampling": sampling,
                "matching": semantics,
                "cache_only": True,
                "cache_digest": digest[:20],
                "final_cache_path": str(final_path),
                "intermediate_cache_path": str(curve_path),
                "intermediate_block_index": int(curve["intermediate_block_index"]),
                "intermediate_block_number": int(curve["intermediate_block_number"]),
                "curve_coordinate_source": self.model_config.get("curve_sampling", {}).get(
                    "coordinate_source", "rendered_polyline_pixel_coordinates"
                ),
                "gaussian_sigma_patch": self.model_config.get("curve_sampling", {}).get(
                    "gaussian_sigma_patch", 1.0
                ),
                "gaussian_radius_patch": self.model_config.get("curve_sampling", {}).get(
                    "gaussian_radius_patch", 3.0
                ),
                "candidate_pool_min_windows": int(pool_counts.min().item()) if paired else 0,
                "candidate_pool_max_windows": int(pool_counts.max().item()) if paired else 0,
                "candidate_pool_digest": pool_digest if paired else "",
                "max_common_visual_windows": MAX_COMMON_WINDOWS,
                "overlap_exclusion_window_index_radius": OVERLAP_RADIUS,
                "n_windows": n_windows,
                "stable_shared_seconds": stable_seconds,
                "candidate_pool_seconds": pool_seconds,
                "matching_seconds": matching_seconds,
                "score_recovery_seconds": recovery_seconds,
                "inference_seconds": matching_seconds + recovery_seconds,
            }
            results[name] = {
                "scores": np.asarray(scores, dtype=np.float64),
                "timestamps": timestamps,
                "metadata": metadata,
                "timing": {
                    "inference_seconds": matching_seconds + recovery_seconds,
                    "matching_seconds": matching_seconds,
                    "score_recovery_seconds": recovery_seconds,
                    "candidate_pool_seconds": pool_seconds,
                    "stable_shared_seconds": stable_seconds,
                },
            }
        return results


def create_backend(
    config: Mapping[str, Any],
    device: str | torch.device,
    cache_dir: str | Path,
) -> CurveAwareViT4TSBackend:
    return CurveAwareViT4TSBackend(config=config, device=device, cache_dir=cache_dir)


__all__ = ["CurveAwareViT4TSBackend", "create_backend"]
