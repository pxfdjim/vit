import json
from pathlib import Path
import unittest

import numpy as np
import torch

from src.models.amr_feature_scoring import (
    fuse_pair, fuse_single, reference_volatility_ratio, volatility_correct,
)
from src.models.numeric_temporal_matching import PathMatch


class TestAMRFeatureScoring(unittest.TestCase):
    def test_fusion_budget(self):
        visual = np.asarray([1.0, 2.0])
        first = np.asarray([3.0, 5.0])
        second = np.asarray([7.0, 11.0])
        np.testing.assert_allclose(fuse_single(visual, first, 0.1), 0.9 * visual + 0.1 * first)
        np.testing.assert_allclose(
            fuse_pair(visual, first, second, 0.1), 0.9 * visual + 0.05 * first + 0.05 * second)

    def test_reference_scale_uses_selected_reference(self):
        windows = torch.zeros(2, 224)
        # Reference window 1 is variable in every Patch.
        windows[1] = torch.arange(224).float().remainder(16)
        masks = torch.eye(14, dtype=torch.bool).repeat(2, 1, 1)
        match = PathMatch(
            scores=torch.zeros(2, 14), reference_ids=torch.ones(2, dtype=torch.long),
            reference_columns=torch.arange(14).repeat(2, 1).float(), path_masks=masks,
            normalized_path_costs=torch.zeros(2), path_lengths=torch.full((2,), 14),
            selected_pool_positions=torch.zeros(2, dtype=torch.long), penalty=0.0,
        )
        ratio, diagnostics = reference_volatility_ratio(windows, match)
        self.assertEqual(tuple(ratio.shape), (2, 14))
        self.assertTrue(torch.isfinite(ratio).all())
        self.assertGreaterEqual(float(ratio.min()), 0.5)
        self.assertLessEqual(float(ratio.max()), 4.0)
        self.assertGreater(diagnostics["mean_ratio"], 0.0)

    def test_correction(self):
        evidence = torch.tensor([[2.0, 4.0]])
        ratio = torch.tensor([[1.0, 2.0]])
        torch.testing.assert_close(volatility_correct(evidence, ratio), torch.tensor([[2.0, 2.0]]))

    def test_preregistered_variant_count_and_controls(self):
        path = Path(__file__).with_name("amr_feature_scoring_config.json")
        config = json.loads(path.read_text())
        variants = config["variants_order"]
        self.assertEqual(len(variants), 47)
        self.assertEqual(len(variants), len(set(variants)))
        self.assertIn("B3_raw_numeric_l010", variants)
        self.assertIn("C4_raw_numeric_l010", variants)
        self.assertEqual(config["fixed_protocol"]["primary_evidence_weight"], 0.1)


if __name__ == "__main__":
    unittest.main()
