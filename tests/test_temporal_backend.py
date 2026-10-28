import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.models.curve_aware_clip import final_cache_path, input_digest, intermediate_cache_path
from src.models.curve_aware_vit4ts import create_backend as create_curve_backend
from src.models.model_utils import patch_scale
from src.models.temporal_vit4ts import create_backend


class TemporalBackendTests(unittest.TestCase):
    @staticmethod
    def frame(length: int = 616) -> pd.DataFrame:
        x = np.arange(length, dtype=np.float64)
        return pd.DataFrame({"timestamp": x, "value": np.sin(x / 19.0) + 8e-4 * x})

    @staticmethod
    def payloads(raw: np.ndarray, windows: int = 8):
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
        block = {
            "cache_version": 1,
            "input_digest": digest,
            "intermediate_block_index": 5,
            "intermediate_block_number": 6,
            "token_normalization": "visual.ln_post",
            "token_projection": "none",
            "source_final_cache_name": f"visual_{digest[:20]}.pt",
            "intermediate_patch": torch.randn(windows, 196, 768, generator=generator),
            "curve_pixel_xy": pixel,
            "curve_grid_xy": pixel / 16 - 0.5,
        }
        return final, block, digest

    @staticmethod
    def variants(include_lambda_zero: bool = False):
        variants = {
            "A_stable": {"family": "A"},
            "B_paired_final": {"family": "B_FINAL"},
            "B_paired_mid": {"family": "B_MID"},
            "T1_independent": {"family": "T1"},
            "T2_shift": {"family": "T2"},
            "T3_dtw": {"family": "T3"},
            "T4_shift_l025": {"family": "T4", "lambda": 0.25},
            "T4_shift_l050": {"family": "T4", "lambda": 0.5},
            "T5_dtw_l025": {"family": "T5", "lambda": 0.25},
            "T5_dtw_l050": {"family": "T5", "lambda": 0.5},
        }
        if include_lambda_zero:
            variants["T4_shift_l000"] = {"family": "T4", "lambda": 0.0}
            variants["T5_dtw_l000"] = {"family": "T5", "lambda": 0.0}
        return variants

    @staticmethod
    def config(final_root: Path):
        return {
            "model": {
                "window_size": 224,
                "step_size": 56,
                "image_size": [224, 224],
                "patch_size": 16,
                "temporal_matching": {"shift_radius": 2, "dtw_band": 2},
            },
            "cache": {"final_cache_roots": [str(final_root)]},
        }

    def prepare(self, root: Path):
        frame = self.frame()
        context = {"dataset": "synthetic", "signal": "s"}
        final, block, digest = self.payloads(frame.value.to_numpy())
        final_root, block_root = root / "final", root / "block"
        fp = final_cache_path(final_root, context, digest)
        bp = intermediate_cache_path(block_root, context, digest)
        fp.parent.mkdir(parents=True)
        bp.parent.mkdir(parents=True)
        torch.save(final, fp)
        torch.save(block, bp)
        return frame, context, final_root, block_root

    def test_outputs_finite_share_pool_and_expose_path_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, context, final_root, block_root = self.prepare(Path(directory))
            predictions = create_backend(
                self.config(final_root), "cpu", block_root
            ).predict_variants(frame, self.variants(), context)
            self.assertEqual(set(predictions), set(self.variants()))
            digests = set()
            for name, prediction in predictions.items():
                self.assertEqual(prediction["scores"].shape, (len(frame),), name)
                self.assertTrue(np.isfinite(prediction["scores"]).all(), name)
                if name.startswith("A_"):
                    self.assertEqual(prediction["metadata"]["candidate_pool_digest"], "")
                else:
                    digests.add(prediction["metadata"]["candidate_pool_digest"])
            self.assertEqual(len(digests), 1)
            self.assertEqual(predictions["T2_shift"]["diagnostics"]["shift"].shape, (8,))
            self.assertEqual(
                predictions["T3_dtw"]["diagnostics"]["path_offsets"].shape, (8, 14)
            )
            self.assertTrue(
                np.isfinite(predictions["T3_dtw"]["diagnostics"]["local_costs"]).all()
            )

    def test_bmid_exactly_reproduces_curve_backend_and_lambda_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, context, final_root, block_root = self.prepare(Path(directory))
            config = self.config(final_root)
            backend = create_backend(config, "cpu", block_root)
            with patch(
                "src.models.temporal_vit4ts.temporal_match_all",
                side_effect=AssertionError("baseline-only run must not compute temporal matching"),
            ):
                baseline = backend.predict_variants(
                    frame, {"B_paired_mid": {"family": "B_MID"}}, context
                )
            previous = create_curve_backend(config, "cpu", block_root).predict_variants(
                frame,
                {
                    "B_mid": {
                        "family": "B",
                        "base_sampling": "full_2d_patch",
                        "layer": "mid",
                    }
                },
                context,
            )
            np.testing.assert_array_equal(
                baseline["B_paired_mid"]["scores"], previous["B_mid"]["scores"]
            )
            current = backend.predict_variants(
                frame,
                {
                    "B_paired_mid": {"family": "B_MID"},
                    "T4_shift_l000": {"family": "T4", "lambda": 0.0},
                    "T5_dtw_l000": {"family": "T5", "lambda": 0.0},
                },
                context,
            )
            np.testing.assert_array_equal(
                current["T4_shift_l000"]["scores"], current["B_paired_mid"]["scores"]
            )
            np.testing.assert_array_equal(
                current["T5_dtw_l000"]["scores"], current["B_paired_mid"]["scores"]
            )


if __name__ == "__main__":
    unittest.main()
