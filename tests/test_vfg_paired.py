import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.model_utils import patch_scale
from src.models.vfg_paired_vit4ts import (
    PairedVFGBackend,
    build_paired_visual_match_cache,
    create_backend,
    eligible_reference_mask,
    joint_scores_from_visual_cache,
    select_common_visual_windows,
    select_fft_guided_windows,
    stable_scale_map,
    stable_median_patch_scores,
    visual_patch_candidates,
)
from src.models.vfg_vit4ts import (
    VFGViT4TSBackend,
    cosine_distance,
    full_window_fft_encoding,
    local_stft_encoding,
)


def random_visual(windows=8, dimension=16, seed=13):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(windows, 196, dimension, generator=generator)


class PairedMatchingTests(unittest.TestCase):
    def test_a_stable_patch_baseline_is_exact_all_window_median_formula(self):
        visual = random_visual(windows=8)
        expected = 0.5 * cosine_distance(
            visual, visual.median(dim=0).values
        ).min(dim=-1).values
        actual = stable_median_patch_scores(visual)
        self.assertTrue(torch.equal(actual, expected))

    def test_overlap_exclusion_is_exactly_abs_index_difference_at_least_four(self):
        mask = eligible_reference_mask(11)
        indices = torch.arange(11)
        expected = (indices[:, None] - indices[None, :]).abs() >= 4
        self.assertTrue(torch.equal(mask, expected))
        self.assertFalse(torch.diagonal(mask).any())

    def test_common_pool_never_contains_self_or_overlap_and_is_bounded(self):
        visual = random_visual(windows=40)
        pool = select_common_visual_windows(visual)
        self.assertLessEqual(pool.window_ids.shape[1], 32)
        for query in range(40):
            ids = pool.ids_for(query)
            self.assertGreater(ids.numel(), 0)
            self.assertTrue(torch.all((ids - query).abs() >= 4))
            self.assertEqual(ids.unique().numel(), ids.numel())

    def test_patch_candidates_keep_exact_window_patch_and_column(self):
        visual = torch.zeros(8, 196, 12)
        generator = torch.Generator().manual_seed(2)
        visual.normal_(generator=generator)
        visual[6, 31] = torch.tensor([9.0] + [0.0] * 11)
        query = torch.zeros(196, 12)
        query[:, 0] = 1.0
        result = visual_patch_candidates(
            query,
            visual,
            torch.tensor([4, 6]),
            top_k=1,
        )
        self.assertTrue(torch.all(result.reference_window_ids == 6))
        self.assertTrue(torch.all(result.reference_patch_ids == 31))
        self.assertTrue(torch.all(result.reference_columns == 3))

    def test_one_visual_pass_products_preserve_common_pool_correspondence(self):
        visual = random_visual(windows=12)
        pool = select_common_visual_windows(visual)
        windows = torch.arange(12 * 224, dtype=torch.float32).reshape(12, 224)
        fft = full_window_fft_encoding(windows)
        guided = select_fft_guided_windows(fft, pool)
        cache = build_paired_visual_match_cache(visual, pool, guided_pool=guided)
        self.assertEqual(cache.paired_visual_scores.shape, (12, 196))
        self.assertEqual(cache.guided_visual_scores.shape, (12, 196))
        for query, candidates in enumerate(cache.topk_candidates):
            allowed = set(pool.ids_for(query).tolist())
            self.assertTrue(set(candidates.reference_window_ids.flatten().tolist()) <= allowed)
            self.assertTrue(
                torch.equal(candidates.reference_columns, candidates.reference_patch_ids % 14)
            )
            guided_ids = set(guided.ids_for(query).tolist())
            self.assertEqual(len(guided_ids), 5)
            self.assertTrue(guided_ids <= allowed)
            self.assertTrue(all(abs(query - ref) >= 4 for ref in guided_ids))
            self.assertIn(
                int(cache.paired_best_candidates.reference_window_ids[query, 0, 0]),
                allowed,
            )
            self.assertIn(
                int(cache.guided_best_candidates.reference_window_ids[query, 0, 0]),
                guided_ids,
            )

    def test_joint_lambda_zero_is_bit_exact_paired_visual_b(self):
        visual = random_visual(windows=8)
        pool = select_common_visual_windows(visual)
        cache = build_paired_visual_match_cache(visual, pool)
        numeric = torch.randn(8, 224, generator=torch.Generator().manual_seed(8))
        frequency = local_stft_encoding(numeric)
        zero, scale = joint_scores_from_visual_cache(
            cache, frequency, frequency_weight=0.0
        )
        self.assertTrue(torch.equal(zero, cache.paired_visual_scores))
        self.assertEqual(float(scale), 1.0)

    def test_joint_scale_is_one_sequence_scalar_and_outputs_are_finite(self):
        visual = random_visual(windows=8)
        pool = select_common_visual_windows(visual)
        cache = build_paired_visual_match_cache(visual, pool)
        numeric = torch.randn(8, 224, generator=torch.Generator().manual_seed(9))
        frequency = local_stft_encoding(numeric)
        scores, scale = joint_scores_from_visual_cache(cache, frequency)
        self.assertEqual(scale.ndim, 0)
        self.assertTrue(torch.isfinite(scale))
        self.assertTrue(torch.isfinite(scores).all())

    def test_short_window_collection_fails_instead_of_using_overlap(self):
        with self.assertRaisesRegex(ValueError, "non-overlapping reference"):
            select_common_visual_windows(random_visual(windows=7))

    def test_fft_guidance_pads_short_rows_and_keeps_all_valid_references(self):
        visual = random_visual(windows=10)
        pool = select_common_visual_windows(visual)
        frequency = full_window_fft_encoding(torch.randn(10, 224))
        guided = select_fft_guided_windows(frequency, pool)
        counts = guided.valid.sum(dim=1)

        self.assertEqual(guided.window_ids.shape, (10, 5))
        self.assertEqual(int(counts.min()), 3)
        self.assertEqual(int(counts.max()), 5)
        self.assertTrue(torch.all(guided.window_ids[~guided.valid] == -1))
        self.assertTrue(torch.isinf(guided.frequency_distances[~guided.valid]).all())
        for query in range(10):
            ids = guided.ids_for(query)
            allowed = pool.ids_for(query)
            self.assertEqual(ids.numel(), min(5, allowed.numel()))
            self.assertTrue(torch.all((ids - query).abs() >= 4))
            self.assertTrue(all(int(item) in set(allowed.tolist()) for item in ids))

    def test_all_cpu_products_stay_on_cpu(self):
        visual = random_visual(windows=8)
        pool = select_common_visual_windows(visual)
        cache = build_paired_visual_match_cache(visual, pool)
        frequency = full_window_fft_encoding(torch.randn(8, 224))
        scores, scale = joint_scores_from_visual_cache(cache, frequency)
        tensors = [
            pool.window_ids,
            pool.visual_distances,
            cache.paired_visual_scores,
            scores,
            scale,
        ]
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in tensors))


class CacheOnlyBackendTests(unittest.TestCase):
    @staticmethod
    def frame(length=840):
        time = np.arange(length, dtype=np.float64)
        values = np.sin(2 * np.pi * time / 32) + 0.003 * time
        return pd.DataFrame({"timestamp": time, "value": values})

    @staticmethod
    def cache_payload(backend, frame, windows=12, dimension=8):
        generator = torch.Generator().manual_seed(21)
        masks = patch_scale(image_size=(224, 224))
        return {
            "cache_version": 1,
            "input_digest": backend._input_digest(frame["value"].to_numpy()),
            "large": torch.randn(windows, 144, dimension, generator=generator),
            "mid": torch.randn(windows, 169, dimension, generator=generator),
            "patch": torch.randn(windows, 196, 1, dimension, generator=generator),
            "large_mask": masks.make_mask(
                kernel_size=48, patch_size=16, stride_size=16
            ).squeeze(0),
            "mid_mask": masks.make_mask(
                kernel_size=32, patch_size=16, stride_size=16
            ).squeeze(0),
        }

    def test_create_backend_does_not_construct_clip_and_missing_cache_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = create_backend({}, "cpu", directory)
            self.assertIsInstance(backend, PairedVFGBackend)
            self.assertFalse(hasattr(backend, "visual_encoder"))
            with self.assertRaisesRegex(FileNotFoundError, "cache-only"):
                backend.predict_variants(
                    self.frame(), {"A_stable": {}}, {"dataset": "d", "signal": "s"}
                )

    def test_a_maps_and_point_recovery_equal_previous_stable_backend(self):
        frame = self.frame()
        paired = PairedVFGBackend({}, "cpu", "/unused")
        features = self.cache_payload(paired, frame)

        previous = object.__new__(VFGViT4TSBackend)
        previous.device = torch.device("cpu")
        previous.image_size = (224, 224)
        previous.window_size = 224
        previous.step_size = 56
        previous_large, previous_mid, previous_patch = previous._stable_visual_components(
            features
        )
        previous_map = (previous_large + previous_mid + previous_patch) / 3.0
        previous_points = previous._point_scores(previous_map, len(frame)).numpy()

        large = stable_scale_map(
            features["large"], features["large_mask"], device=torch.device("cpu")
        )
        mid = stable_scale_map(
            features["mid"], features["mid_mask"], device=torch.device("cpu")
        )
        patch = stable_median_patch_scores(features["patch"]).reshape(12, 14, 14)
        current_map = (large + mid + patch.to(torch.float64)) / 3.0
        from src.models.vfg_paired_vit4ts import point_scores_from_maps

        current_points = point_scores_from_maps(
            current_map, full_length=len(frame), device=torch.device("cpu")
        )
        torch.testing.assert_close(current_map, previous_map, rtol=0, atol=0)
        np.testing.assert_array_equal(current_points, previous_points)

    def test_backend_reads_cache_without_modifying_it_and_returns_a_to_e(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = create_backend({}, "cpu", directory)
            frame = self.frame()
            context = {"dataset": "dataset", "signal": "signal"}
            digest = backend._input_digest(frame["value"].to_numpy())
            path = backend._cache_path(context, digest)
            path.parent.mkdir(parents=True)
            torch.save(self.cache_payload(backend, frame), path)
            before = path.read_bytes()
            variants = {
                "A_stable_median": {},
                "B_paired_visual": {},
                "C_paired_whole_fft": {"lambda": 0.3},
                "D_fft_guided": {},
                "E_paired_local_stft": {"lambda": 0.3},
            }
            predictions = backend.predict_variants(frame, variants, context)
            after = path.read_bytes()
            self.assertEqual(before, after)
            self.assertEqual(set(predictions), set(variants))
            for name, prediction in predictions.items():
                self.assertEqual(prediction["scores"].shape, (840,), name)
                self.assertTrue(np.isfinite(prediction["scores"]).all(), name)
                metadata = prediction["metadata"]
                self.assertIn("incremental_matching_seconds", metadata)
                self.assertIn("incremental_frequency_seconds", metadata)
                self.assertIn("score_recovery_seconds", metadata)
                self.assertTrue(metadata["cache_only"])
            self.assertEqual(
                predictions["D_fft_guided"]["metadata"]["fft_guided_selected_windows"],
                5,
            )
            guided_metadata = predictions["D_fft_guided"]["metadata"]
            self.assertEqual(guided_metadata["fft_guided_target_windows"], 5)
            self.assertEqual(guided_metadata["fft_guided_min_selected_windows"], 5)
            self.assertEqual(guided_metadata["fft_guided_max_selected_windows"], 5)

    def test_backend_d_accepts_ten_window_series_with_three_to_five_references(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = create_backend({}, "cpu", directory)
            frame = self.frame(length=741)
            context = {"dataset": "Yahoo_A1", "signal": "real_54"}
            digest = backend._input_digest(frame["value"].to_numpy())
            path = backend._cache_path(context, digest)
            path.parent.mkdir(parents=True)
            torch.save(self.cache_payload(backend, frame, windows=10), path)

            prediction = backend.predict_variants(
                frame, {"D_fft_guided": {}}, context
            )["D_fft_guided"]
            metadata = prediction["metadata"]
            self.assertEqual(prediction["scores"].shape, (741,))
            self.assertTrue(np.isfinite(prediction["scores"]).all())
            self.assertEqual(metadata["fft_guided_target_windows"], 5)
            self.assertEqual(metadata["fft_guided_min_selected_windows"], 3)
            self.assertEqual(metadata["fft_guided_max_selected_windows"], 5)

    def test_backend_rejects_short_paired_series_before_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = create_backend({}, "cpu", directory)
            frame = self.frame(length=560)  # seven windows
            context = {"dataset": "d", "signal": "short"}
            digest = backend._input_digest(frame["value"].to_numpy())
            path = backend._cache_path(context, digest)
            path.parent.mkdir(parents=True)
            torch.save(self.cache_payload(backend, frame, windows=7), path)
            with self.assertRaisesRegex(ValueError, "at least 8 windows"):
                backend.predict_variants(frame, {"B_paired": {}}, context)


if __name__ == "__main__":
    unittest.main()
