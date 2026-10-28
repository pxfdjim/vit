"""Cache-only temporal matching variants for first-stage ViT4TS.

The implementation deliberately keeps the proven visual representation and
score restoration unchanged.  Final-layer tokens select one common pool of at
most 32 real, non-overlapping windows.  Block-6 tokens are then used by the
paired visual baseline and by all temporal matchers.  The temporal branch only
changes how the fourteen visual columns may correspond in time.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.models.curve_aware_clip import (
    expected_window_count,
    input_digest,
    intermediate_cache_path,
    load_final_tokens,
    validate_intermediate_cache,
)
from src.models.temporal_matching import temporal_match_all
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


_EPS = 1e-8
_MAX_CALIBRATION = 1e4


def _pool_digest(window_ids: torch.Tensor, valid: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(window_ids.cpu().numpy()).view(np.uint8))
    digest.update(np.ascontiguousarray(valid.cpu().numpy()).view(np.uint8))
    return digest.hexdigest()[:20]


def _column_map(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != GRID_SIZE:
        raise ValueError("temporal scores must have shape [windows, 14]")
    return scores[:, None, :].expand(-1, GRID_SIZE, -1)


def _median_ratio(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Return a bounded, label-free median scale for ``value``.

    Only finite positive distances from the current unlabelled signal are
    used.  Empty or degenerate samples receive the neutral factor one.  The
    bound is solely a numerical guard and is fixed for every signal/model.
    """

    ref = reference.detach().reshape(-1)
    val = value.detach().reshape(-1)
    ref = ref[torch.isfinite(ref) & (ref > _EPS)]
    val = val[torch.isfinite(val) & (val > _EPS)]
    if ref.numel() == 0 or val.numel() == 0:
        return torch.ones((), dtype=value.dtype, device=value.device)
    factor = ref.median().to(value.device) / val.median().clamp_min(_EPS)
    return factor.clamp(1.0 / _MAX_CALIBRATION, _MAX_CALIBRATION).to(value.dtype)


def _diagnostic_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


class TemporalViT4TSBackend:
    """Evaluate stable, paired and temporally constrained cached variants."""

    def __init__(
        self,
        config: Mapping[str, Any],
        device: str | torch.device,
        cache_dir: str | Path,
    ) -> None:
        self.config = dict(config)
        self.model_config = dict(config.get("model", {}))
        self.cache_config = dict(config.get("cache", {}))
        if str(device) == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
            raise ValueError("temporal protocol is fixed to 224/56, image 224, patch 16")

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
        block_path = intermediate_cache_path(self.cache_dir, context, digest)
        if not block_path.is_file():
            raise FileNotFoundError(f"missing block-6 cache: {block_path}")
        block = torch.load(block_path, map_location="cpu", weights_only=True)
        validate_intermediate_cache(
            block, digest=digest, expected_windows=n_windows, path=block_path
        )
        for key, shape in {
            "large": (n_windows, 144),
            "mid": (n_windows, 169),
            "large_mask": (9, 144),
            "mid_mask": (4, 169),
        }.items():
            tensor = final.get(key)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape[:2]) != shape:
                raise RuntimeError(f"invalid {key} in final cache: {final_path}")
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite {key} in final cache: {final_path}")
        return final, block, digest, final_path, block_path

    @staticmethod
    def _family(name: str, spec: Mapping[str, Any]) -> str:
        family = str(spec.get("family", name.split("_", 1)[0])).upper()
        aliases = {
            "BMID": "B_MID", "B-MID": "B_MID", "B_MID": "B_MID",
            "BFINAL": "B_FINAL", "B-FINAL": "B_FINAL", "B_FINAL": "B_FINAL",
        }
        family = aliases.get(family, family)
        if family == "B":
            default_layer = "mid" if "mid" in name.lower() else "final"
            family = (
                "B_MID"
                if str(spec.get("layer", default_layer)).lower() in {"mid", "block6", "block_6"}
                else "B_FINAL"
            )
        if family not in {"A", "B_FINAL", "B_MID", "T1", "T2", "T3", "T4", "T5"}:
            raise KeyError(f"unknown temporal family: {family}")
        return family

    @staticmethod
    def _lambda(spec: Mapping[str, Any]) -> float:
        value = float(spec.get("lambda", spec.get("fusion_lambda", 0.25)))
        # Zero is intentionally accepted for the exact-recovery diagnostic.
        if value not in {0.0, 0.25, 0.5}:
            raise ValueError("temporal fusion lambda must be fixed at 0, 0.25, or 0.5")
        return value

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
        final, block, digest, final_path, block_path = self._load_caches(raw, context)
        final_patch_cpu = _patch_tokens(final["patch"])
        block_patch_cpu = _patch_tokens(block["intermediate_patch"])
        if final_patch_cpu.shape != block_patch_cpu.shape:
            raise RuntimeError("final and block-6 Patch token shapes differ")
        n_windows = final_patch_cpu.shape[0]
        families = {self._family(name, spec) for name, spec in variants.items()}
        if any(family != "A" for family in families) and n_windows < 8:
            raise ValueError("paired variants require at least eight windows")

        stable_started = self._clock()
        large_map = stable_scale_map(final["large"], final["large_mask"], device=self.device)
        mid_map = stable_scale_map(final["mid"], final["mid_mask"], device=self.device)
        stable_base = stable_median_patch_scores(final_patch_cpu.to(self.device))
        stable_base_map = stable_base.reshape(n_windows, GRID_SIZE, GRID_SIZE).to(torch.float64)
        stable_seconds = self._clock() - stable_started

        pool_started = self._clock()
        final_patch = final_patch_cpu.to(self.device)
        common_pool = select_common_visual_windows(final_patch)
        pool_seconds = self._clock() - pool_started
        pool_counts = common_pool.valid.sum(dim=1)
        pool_digest = _pool_digest(common_pool.window_ids, common_pool.valid)

        block_patch = block_patch_cpu.to(self.device)
        paired_started = self._clock()
        b_final = (
            paired_visual_patch_scores(final_patch, common_pool).reshape(
                n_windows, GRID_SIZE, GRID_SIZE
            )
            if "B_FINAL" in families
            else None
        )
        need_b_mid = bool(families & {"B_MID", "T1", "T2", "T3", "T4", "T5"})
        b_mid = (
            paired_visual_patch_scores(block_patch, common_pool).reshape(
                n_windows, GRID_SIZE, GRID_SIZE
            )
            if need_b_mid
            else None
        )
        paired_seconds = self._clock() - paired_started

        temporal_config = dict(self.model_config.get("temporal_matching", {}))
        temporal = None
        temporal_maps: dict[str, torch.Tensor] = {}
        calibrations: dict[str, torch.Tensor] = {}
        temporal_seconds = 0.0
        if families & {"T1", "T2", "T3", "T4", "T5"}:
            temporal_started = self._clock()
            temporal = temporal_match_all(
                block_patch,
                common_pool,
                shift_radius=int(temporal_config.get("shift_radius", 2)),
                dtw_band=int(temporal_config.get("dtw_band", 2)),
                dtw_penalty_scale=float(temporal_config.get("dtw_penalty_scale", 0.1)),
                output_factor=0.5,
            )
            temporal_seconds = self._clock() - temporal_started
            temporal_maps = {
                "T1": _column_map(temporal.t1_scores),
                "T2": _column_map(temporal.t2_scores),
                "T3": _column_map(temporal.t3_scores),
            }
            assert b_mid is not None
            calibrations = {
                key: _median_ratio(b_mid, value) for key, value in temporal_maps.items()
            }

        results: dict[str, dict[str, Any]] = {}
        for name, spec in variants.items():
            family = self._family(name, spec)
            variant_started = self._clock()
            fusion_lambda = 0.0
            calibration = torch.ones((), device=self.device, dtype=b_mid.dtype)
            temporal_source = ""
            if family == "A":
                base_map = stable_base_map.to(self.device)
                semantics = "stable_all_window_median_final_patch"
            elif family == "B_FINAL":
                assert b_final is not None
                base_map = b_final
                semantics = "paired_full_2d_patch_final"
            elif family == "B_MID":
                assert b_mid is not None
                base_map = b_mid
                semantics = "paired_full_2d_patch_block6"
            elif family in temporal_maps:
                base_map = temporal_maps[family]
                semantics = f"block6_{family.lower()}_temporal_column_matching"
                temporal_source = family
            else:
                temporal_source = "T2" if family == "T4" else "T3"
                fusion_lambda = self._lambda(spec)
                calibration = calibrations[temporal_source]
                assert b_mid is not None
                if fusion_lambda == 0.0:
                    base_map = b_mid
                else:
                    calibrated = calibration * temporal_maps[temporal_source]
                    base_map = (1.0 - fusion_lambda) * b_mid + fusion_lambda * calibrated
                semantics = f"paired_block6_plus_{temporal_source.lower()}"
            matching_seconds = self._clock() - variant_started

            combined = (large_map + mid_map + base_map.detach().cpu().to(torch.float64)) / 3.0
            recovery_started = self._clock()
            scores = point_scores_from_maps(combined, full_length=raw.size, device=self.device)
            recovery_seconds = self._clock() - recovery_started
            paired = family != "A"
            metadata = {
                "family": family,
                "matching": semantics,
                "visual_layer": "final" if family in {"A", "B_FINAL"} else "block6",
                "cache_only": True,
                "cache_digest": digest[:20],
                "final_cache_path": str(final_path),
                "intermediate_cache_path": str(block_path),
                "candidate_pool_min_windows": int(pool_counts.min().item()) if paired else 0,
                "candidate_pool_max_windows": int(pool_counts.max().item()) if paired else 0,
                "candidate_pool_digest": pool_digest if paired else "",
                "max_common_visual_windows": MAX_COMMON_WINDOWS,
                "overlap_exclusion_window_index_radius": OVERLAP_RADIUS,
                "n_windows": n_windows,
                "temporal_source": temporal_source,
                "fusion_lambda": fusion_lambda,
                "temporal_calibration_factor": float(calibration.detach().cpu()),
                "calibration_formula": "median(positive B-mid base distances) / median(positive temporal base distances)",
                "calibration_epsilon": _EPS,
                "calibration_bound": _MAX_CALIBRATION,
                "shift_radius": int(temporal_config.get("shift_radius", 2)),
                "dtw_band": int(temporal_config.get("dtw_band", 2)),
                "dtw_non_diagonal_penalty": (
                    float(temporal.dtw_penalty.detach().cpu()) if temporal is not None else 0.0
                ),
                "dtw_penalty_scale": float(temporal_config.get("dtw_penalty_scale", 0.1)),
                "stable_shared_seconds": stable_seconds,
                "candidate_pool_seconds": pool_seconds,
                "paired_base_seconds": paired_seconds,
                "temporal_shared_seconds": temporal_seconds,
                "matching_seconds": matching_seconds,
                "score_recovery_seconds": recovery_seconds,
                "inference_seconds": matching_seconds + recovery_seconds,
            }
            diagnostics: dict[str, Any] = {}
            if family == "T1":
                assert temporal is not None
                diagnostics = {
                    "reference_window_ids": _diagnostic_numpy(temporal.t1_reference_ids),
                    "reference_columns": _diagnostic_numpy(temporal.t1_reference_columns),
                    "column_costs": _diagnostic_numpy(temporal.t1_scores),
                }
            elif family in {"T2", "T4"}:
                assert temporal is not None
                diagnostics = {
                    "reference_window_ids": _diagnostic_numpy(temporal.t2_reference_ids),
                    "shift": _diagnostic_numpy(temporal.t2_shifts),
                    "valid_columns": _diagnostic_numpy(temporal.t2_valid),
                    "column_costs": _diagnostic_numpy(temporal.t2_scores),
                    "objective_costs": _diagnostic_numpy(temporal.t2_objective_costs),
                }
            elif family in {"T3", "T5"}:
                assert temporal is not None
                diagnostics = {
                    "reference_window_ids": _diagnostic_numpy(temporal.t3_reference_ids),
                    "path_offsets": _diagnostic_numpy(temporal.t3_offsets),
                    "reference_columns": _diagnostic_numpy(temporal.t3_reference_columns),
                    "local_costs": _diagnostic_numpy(temporal.t3_scores),
                    "normalized_path_costs": _diagnostic_numpy(temporal.t3_path_costs),
                    "path_lengths": _diagnostic_numpy(temporal.t3_path_lengths),
                }
            results[name] = {
                "scores": np.asarray(scores, dtype=np.float64),
                "timestamps": timestamps,
                "metadata": metadata,
                "diagnostics": diagnostics,
                "timing": {
                    "inference_seconds": matching_seconds + recovery_seconds,
                    "matching_seconds": matching_seconds,
                    "score_recovery_seconds": recovery_seconds,
                    "candidate_pool_seconds": pool_seconds,
                    "paired_base_seconds": paired_seconds,
                    "temporal_shared_seconds": temporal_seconds,
                    "stable_shared_seconds": stable_seconds,
                },
            }
        return results


def create_backend(
    config: Mapping[str, Any],
    device: str | torch.device,
    cache_dir: str | Path,
) -> TemporalViT4TSBackend:
    return TemporalViT4TSBackend(config=config, device=device, cache_dir=cache_dir)


__all__ = ["TemporalViT4TSBackend", "create_backend"]
