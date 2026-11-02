"""Focused regression tests for AMR versus late-fusion selector data flow."""
from __future__ import annotations

import unittest

import torch

from src.models.bimodal_reference import AlignedMultimodalReference
from src.models.bimodal_reference_vit4ts import _learned_cost_kind


class DataflowTest(unittest.TestCase):
    def test_selector_cost_sources_are_independent(self):
        self.assertEqual(_learned_cost_kind("TS_ONLY"), "numeric")
        self.assertEqual(_learned_cost_kind("LATE_JOINT"), "numeric")
        self.assertEqual(_learned_cost_kind("CROSS_MODAL"), "fused")
        with self.assertRaises(ValueError):
            _learned_cost_kind("unknown")

    def test_fused_tokens_depend_on_cross_attention_outputs(self):
        torch.manual_seed(20261102)
        model = AlignedMultimodalReference().eval()
        raw = torch.randn(3, 224)
        visual = torch.randn(3, 196, 768)
        with torch.inference_mode():
            original = model(raw, visual)

            def zero_attention(_module, _inputs, output):
                return torch.zeros_like(output[0]), output[1]

            handles = [
                model.n_to_v.register_forward_hook(zero_attention),
                model.v_to_n.register_forward_hook(zero_attention),
            ]
            without_attention = model(raw, visual)
            for handle in handles:
                handle.remove()

        for name in ("numeric", "visual", "fused"):
            self.assertEqual(tuple(original[name].shape), (3, 14, 192))
            self.assertTrue(torch.isfinite(original[name]).all())
        self.assertTrue(torch.equal(original["numeric"], without_attention["numeric"]))
        self.assertTrue(torch.equal(original["visual"], without_attention["visual"]))
        self.assertGreater(
            float((original["fused"] - without_attention["fused"]).abs().max()),
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
