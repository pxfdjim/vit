import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.curve_aware_clip import (
    input_digest,
    intermediate_cache_path,
    final_cache_path,
)
from src.models.curve_aware_vit4ts import create_backend
from src.models.model_utils import patch_scale
from src.models.vfg_paired_vit4ts import create_backend as create_paired_backend


class CurveAwareBackendTests(unittest.TestCase):
    @staticmethod
    def frame(length=616):
        x = np.arange(length, dtype=np.float64)
        return pd.DataFrame({"timestamp": x, "value": np.sin(x / 17.0) + x * 1e-3})

    @staticmethod
    def payloads(raw, windows=8):
        generator = torch.Generator().manual_seed(20261028)
        digest = input_digest(raw)
        masks = patch_scale(image_size=(224, 224))
        final = {
            "cache_version": 1,
            "input_digest": digest,
            "large": torch.randn(windows, 144, 768, generator=generator),
            "mid": torch.randn(windows, 169, 768, generator=generator),
            "patch": torch.randn(windows, 196, 1, 768, generator=generator),
            "large_mask": masks.make_mask(
                kernel_size=48, patch_size=16, stride_size=16
            ).squeeze(0),
            "mid_mask": masks.make_mask(
                kernel_size=32, patch_size=16, stride_size=16
            ).squeeze(0),
        }
        x = torch.arange(14, dtype=torch.float32) * 16 + 8
        y = torch.linspace(8, 216, 14).repeat(windows, 1)
        pixel = torch.stack((x.repeat(windows, 1), y), dim=-1)
        grid = pixel / 16 - 0.5
        curve = {
            "cache_version": 1,
            "input_digest": digest,
            "intermediate_block_index": 5,
            "intermediate_block_number": 6,
            "token_normalization": "visual.ln_post",
            "token_projection": "none",
            "source_final_cache_name": f"visual_{digest[:20]}.pt",
            "intermediate_patch": torch.randn(windows, 196, 768, generator=generator),
            "curve_pixel_xy": pixel,
            "curve_grid_xy": grid,
        }
        return final, curve, digest

    @staticmethod
    def variants():
        result = {
            "A_stable_final": {"family": "A", "base_sampling": "full_2d_patch", "layer": "final"},
        }
        for family, sampling in [
            ("B", "full_2d_patch"),
            ("C", "column_mean"),
            ("D", "curve_bilinear"),
            ("E", "curve_gaussian"),
        ]:
            for layer in ["final", "mid"]:
                result[f"{family}_{sampling}_{layer}"] = {
                    "family": family, "base_sampling": sampling, "layer": layer,
                }
        return result

    def test_all_variants_are_finite_and_share_exact_candidate_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_root, curve_root = root / "final", root / "curve"
            frame = self.frame(); context = {"dataset": "d", "signal": "s"}
            final, curve, digest = self.payloads(frame.value.to_numpy())
            fp = final_cache_path(final_root, context, digest); fp.parent.mkdir(parents=True)
            cp = intermediate_cache_path(curve_root, context, digest); cp.parent.mkdir(parents=True)
            torch.save(final, fp); torch.save(curve, cp)
            config = {
                "model": {"window_size": 224, "step_size": 56, "image_size": [224, 224], "patch_size": 16, "curve_sampling": {"gaussian_sigma_patch": 1.0}},
                "cache": {"final_cache_roots": [str(final_root)]},
            }
            backend = create_backend(config, "cpu", curve_root)
            predictions = backend.predict_variants(frame, self.variants(), context)
            self.assertEqual(set(predictions), set(self.variants()))
            digests = set()
            for name, prediction in predictions.items():
                self.assertEqual(prediction["scores"].shape, (len(frame),), name)
                self.assertTrue(np.isfinite(prediction["scores"]).all(), name)
                if name.startswith("A_"):
                    self.assertEqual(prediction["metadata"]["candidate_pool_digest"], "")
                else:
                    digests.add(prediction["metadata"]["candidate_pool_digest"])
                    self.assertGreaterEqual(prediction["metadata"]["candidate_pool_min_windows"], 1)
                    self.assertLessEqual(prediction["metadata"]["candidate_pool_max_windows"], 32)
            self.assertEqual(len(digests), 1)

    def test_a_and_final_b_reproduce_paired_backend_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_root, curve_root = root / "final", root / "curve"
            frame = self.frame(); context = {"dataset": "d", "signal": "s"}
            final, curve, digest = self.payloads(frame.value.to_numpy())
            fp = final_cache_path(final_root, context, digest); fp.parent.mkdir(parents=True)
            cp = intermediate_cache_path(curve_root, context, digest); cp.parent.mkdir(parents=True)
            torch.save(final, fp); torch.save(curve, cp)
            config = {"model": {"window_size": 224, "step_size": 56, "image_size": [224, 224], "patch_size": 16}, "cache": {"final_cache_roots": [str(final_root)]}}
            current = create_backend(config, "cpu", curve_root).predict_variants(
                frame,
                {"A": {"family": "A", "base_sampling": "full_2d_patch", "layer": "final"}, "B": {"family": "B", "base_sampling": "full_2d_patch", "layer": "final"}},
                context,
            )
            previous = create_paired_backend({}, "cpu", final_root).predict_variants(
                frame, {"A_stable": {}, "B_paired": {}}, context
            )
            np.testing.assert_array_equal(current["A"]["scores"], previous["A_stable"]["scores"])
            np.testing.assert_array_equal(current["B"]["scores"], previous["B_paired"]["scores"])


if __name__ == "__main__":
    unittest.main()
