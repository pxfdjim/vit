"""Cache-only numeric-modality controls around the immutable T3 pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.models.curve_aware_clip import (
    expected_window_count, input_digest, intermediate_cache_path, load_final_tokens,
    validate_intermediate_cache,
)
from src.models.numeric_temporal_matching import (
    build_cost_bundle, match_paths, mix_costs, score_costs_on_paths,
)
from src.models.temporal_matching import temporal_match_all
from src.models.temporal_vit4ts import _column_map, _diagnostic_numpy, _pool_digest
from src.models.vfg_paired_vit4ts import (
    GRID_SIZE, MAX_COMMON_WINDOWS, OVERLAP_RADIUS, _patch_tokens, point_scores_from_maps,
    select_common_visual_windows, stable_scale_map,
)


class NumericTemporalBackend:
    """Evaluate A0/A1/A2/A3 from one shared visual/numeric distance pass."""

    def __init__(self, config: Mapping[str, Any], device: str | torch.device, cache_dir: str | Path):
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
            raise ValueError("cache.final_cache_roots is required")

    def _clock(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _load(self, raw: np.ndarray, context: Mapping[str, Any]):
        digest = input_digest(raw)
        windows = expected_window_count(raw.size)
        _, final_path = load_final_tokens(self.final_cache_roots, context, digest, windows)
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        block_path = intermediate_cache_path(self.cache_dir, context, digest)
        block = torch.load(block_path, map_location="cpu", weights_only=True)
        validate_intermediate_cache(block, digest=digest, expected_windows=windows, path=block_path)
        return final, block, digest, final_path, block_path

    @staticmethod
    def _kind(spec: Mapping[str, Any]) -> str:
        kind = str(spec.get("numeric_variant", "A0")).upper()
        allowed = {"A0", "A1", "A2", "A3_SELECT", "A3_JOINT"}
        if kind not in allowed:
            raise KeyError(f"unknown numeric variant: {kind}")
        return kind

    @staticmethod
    def _weight(spec: Mapping[str, Any]) -> float:
        value = float(spec.get("numeric_weight", 0.0))
        if value not in {0.0, 0.1, 0.25, 0.5}:
            raise ValueError("numeric_weight must be one of 0, 0.1, 0.25, 0.5")
        return value

    @torch.inference_mode()
    def predict_variants(self, data: Any, variants: Mapping[str, Mapping[str, Any]], context: Mapping[str, Any]):
        raw = np.asarray(data["value"], dtype=np.float64)
        timestamps = np.asarray(data["timestamp"])
        if raw.ndim != 1 or raw.size < 224 or not np.isfinite(raw).all():
            raise ValueError("input must be one finite univariate sequence")
        final, block, digest, final_path, block_path = self._load(raw, context)
        final_patch = _patch_tokens(final["patch"]).to(self.device)
        block_patch = _patch_tokens(block["intermediate_patch"]).to(self.device)
        n_windows = block_patch.shape[0]

        scale_started = self._clock()
        large_map = stable_scale_map(final["large"], final["large_mask"], device=self.device)
        mid_map = stable_scale_map(final["mid"], final["mid_mask"], device=self.device)
        scale_seconds = self._clock() - scale_started

        pool_started = self._clock()
        pool = select_common_visual_windows(final_patch)
        pool_seconds = self._clock() - pool_started
        pool_counts = pool.valid.sum(dim=1)
        pool_digest = _pool_digest(pool.window_ids, pool.valid)

        temporal_cfg = dict(self.model_config.get("temporal_matching", {}))
        band = int(temporal_cfg.get("dtw_band", 2))
        penalty_scale = float(temporal_cfg.get("dtw_penalty_scale", 0.1))
        t3_started = self._clock()
        t3 = temporal_match_all(
            block_patch, pool, shift_radius=int(temporal_cfg.get("shift_radius", 2)),
            dtw_band=band, dtw_penalty_scale=penalty_scale, output_factor=0.5,
        )
        t3_seconds = self._clock() - t3_started

        cost_started = self._clock()
        bundle = build_cost_bundle(block_patch, pool, raw)
        visual_match = match_paths(bundle.visual, pool, band=band, penalty_scale=penalty_scale)
        visual_scores = 0.5 * visual_match.scores.to(self.device)
        max_visual_score_error = float((visual_scores - t3.t3_scores).abs().max().cpu())
        references_equal = bool(torch.equal(visual_match.reference_ids.to(self.device), t3.t3_reference_ids))
        if max_visual_score_error > 1e-6 or not references_equal:
            raise RuntimeError(
                f"independent visual-cost audit failed: score={max_visual_score_error}, refs={references_equal}"
            )
        numeric_match = match_paths(bundle.numeric, pool, band=band, penalty_scale=penalty_scale)
        fixed_numeric = 0.5 * score_costs_on_paths(bundle.numeric, visual_match).to(self.device)
        numeric_only = 0.5 * numeric_match.scores.to(self.device)
        requested_weights = sorted({self._weight(spec) for spec in variants.values() if self._kind(spec).startswith("A3")})
        joint_matches = {
            weight: match_paths(mix_costs(bundle.visual, bundle.numeric, weight), pool, band=band, penalty_scale=penalty_scale)
            for weight in requested_weights
        }
        cost_seconds = self._clock() - cost_started

        # Base-only restored evidence is diagnostic; the actual detector still
        # combines its selected base map with unchanged Large/Mid maps below.
        evidence_cache: dict[str, np.ndarray] = {}
        def restore_evidence(key: str, columns: torch.Tensor) -> np.ndarray:
            if key not in evidence_cache:
                evidence_cache[key] = np.asarray(
                    point_scores_from_maps(
                        _column_map(columns).to(torch.float64),
                        full_length=raw.size,
                        device=self.device,
                    ),
                    dtype=np.float64,
                )
            return evidence_cache[key]

        results = {}
        for name, spec in variants.items():
            variant_started = self._clock()
            kind, weight = self._kind(spec), self._weight(spec)
            match = visual_match
            visual_local = visual_scores
            numeric_local = fixed_numeric
            if kind == "A0":
                base_columns = t3.t3_scores
                semantics = "immutable_clip_block6_t3"
            elif kind == "A1":
                match = numeric_match
                numeric_local = numeric_only
                visual_local = 0.5 * score_costs_on_paths(bundle.visual, numeric_match).to(self.device)
                base_columns = numeric_local
                semantics = "numeric_only_candidate_selection_dtw_and_scoring"
            elif kind == "A2":
                base_columns = (1.0 - weight) * visual_scores + weight * fixed_numeric
                semantics = "fixed_t3_reference_path_numeric_score_fusion"
            else:
                match = joint_matches[weight]
                visual_local = 0.5 * score_costs_on_paths(bundle.visual, match).to(self.device)
                numeric_local = 0.5 * score_costs_on_paths(bundle.numeric, match).to(self.device)
                if kind == "A3_SELECT":
                    base_columns = visual_local
                    semantics = "joint_reference_and_path_selection_visual_scoring_only"
                else:
                    base_columns = (1.0 - weight) * visual_local + weight * numeric_local
                    semantics = "joint_reference_path_selection_and_joint_scoring"
            base_map = _column_map(base_columns)
            combined = (large_map + mid_map + base_map.detach().cpu().to(torch.float64)) / 3.0
            matching_seconds = self._clock() - variant_started
            recovery_started = self._clock()
            scores = point_scores_from_maps(
                combined, full_length=raw.size, device=self.device
            )
            recovery_seconds = self._clock() - recovery_started

            reference_changed = match.reference_ids.cpu() != visual_match.reference_ids.cpu()
            path_changed = (match.path_masks.cpu() != visual_match.path_masks.cpu()).reshape(n_windows, -1).any(dim=1)
            diagnostics = {
                "reference_window_ids": _diagnostic_numpy(match.reference_ids),
                "reference_columns": _diagnostic_numpy(match.reference_columns),
                "path_offsets": _diagnostic_numpy(match.reference_columns - torch.arange(GRID_SIZE)),
                "local_costs": _diagnostic_numpy(base_columns),
                "visual_local_costs": _diagnostic_numpy(visual_local),
                "numeric_local_costs": _diagnostic_numpy(numeric_local),
                "normalized_path_costs": _diagnostic_numpy(match.normalized_path_costs),
                "path_lengths": _diagnostic_numpy(match.path_lengths),
                "path_masks": _diagnostic_numpy(match.path_masks.to(torch.uint8)),
                "t3_reference_window_ids": _diagnostic_numpy(visual_match.reference_ids),
                "reference_changed": _diagnostic_numpy(reference_changed.to(torch.uint8)),
                "path_changed": _diagnostic_numpy(path_changed.to(torch.uint8)),
                "visual_evidence_point_scores": restore_evidence("visual_" + kind + f"_{weight}", visual_local),
                "numeric_evidence_point_scores": restore_evidence("numeric_" + kind + f"_{weight}", numeric_local),
            }
            metadata = {
                "family": kind, "matching": semantics, "visual_layer": "block6", "cache_only": True,
                "cache_digest": digest[:20], "final_cache_path": str(final_path),
                "intermediate_cache_path": str(block_path), "candidate_pool_digest": pool_digest,
                "candidate_pool_min_windows": int(pool_counts.min()),
                "candidate_pool_max_windows": int(pool_counts.max()),
                "max_common_visual_windows": MAX_COMMON_WINDOWS,
                "overlap_exclusion_window_index_radius": OVERLAP_RADIUS,
                "n_windows": n_windows, "numeric_weight": weight,
                "numeric_feature_semantics": "sequence_detrend; global robust scale; centered16+diff15+level/amplitude/MAD/std/slope/endpoint",
                "numeric_component_scales": bundle.component_scales,
                "numeric_to_visual_scale": bundle.numeric_to_visual_scale,
                "visual_audit_max_abs_score_error": max_visual_score_error,
                "visual_audit_reference_ids_equal": references_equal,
                "reference_change_fraction": float(reference_changed.float().mean()),
                "path_change_fraction": float(path_changed.float().mean()),
                "dtw_band": band, "dtw_penalty_scale": penalty_scale, "dtw_penalty": match.penalty,
                "shared_scale_seconds": scale_seconds, "candidate_pool_seconds": pool_seconds,
                "original_t3_seconds": t3_seconds, "shared_numeric_visual_cost_seconds": cost_seconds,
                "matching_seconds": matching_seconds, "score_recovery_seconds": recovery_seconds,
                "inference_seconds": matching_seconds + recovery_seconds,
            }
            results[name] = {
                "scores": np.asarray(scores, dtype=np.float64), "timestamps": timestamps,
                "metadata": metadata, "diagnostics": diagnostics,
                "timing": {
                    "matching_seconds": matching_seconds, "score_recovery_seconds": recovery_seconds,
                    "candidate_pool_seconds": pool_seconds, "original_t3_seconds": t3_seconds,
                    "shared_numeric_visual_cost_seconds": cost_seconds,
                    "inference_seconds": matching_seconds + recovery_seconds,
                },
            }
        return results


def create_backend(config: Mapping[str, Any], device: str | torch.device, cache_dir: str | Path):
    return NumericTemporalBackend(config, device, cache_dir)


__all__ = ["NumericTemporalBackend", "create_backend"]
