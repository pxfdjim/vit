from __future__ import annotations

import torch

from src.models.amr_training_ablation import (
    AblationReferenceModel, AlignmentLossConfig, alignment_loss,
    masked_numeric_batch, masked_reconstruction_loss,
)
from src.models.bimodal_reference import AlignedMultimodalReference, patch_alignment_loss


def test_full_model_and_loss_are_exactly_compatible():
    torch.manual_seed(7)
    original = AlignedMultimodalReference()
    ablation = AblationReferenceModel("cross")
    ablation.load_state_dict(original.state_dict(), strict=True)
    original.eval(); ablation.eval()
    raw = torch.randn(2, 224)
    visual = torch.randn(2, 196, 768)
    expected = original(raw, visual)
    actual = ablation(raw, visual)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    expected_loss, _ = patch_alignment_loss(expected)
    actual_loss, _, _ = alignment_loss(actual, AlignmentLossConfig())
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=2e-7)


def test_l5_has_no_active_loss():
    model = AblationReferenceModel("cross").eval()
    output = model(torch.randn(2, 224), torch.randn(2, 196, 768))
    loss, groups, _ = alignment_loss(output, AlignmentLossConfig(False, False, False, False))
    assert loss is None
    assert set(groups) == {"patch", "temporal_relation", "transition_geometry", "reference"}


def test_numeric_masked_reconstruction_is_finite():
    raw = torch.randn(4, 224)
    masked, target, mask = masked_numeric_batch(raw, 0.25)
    assert mask.shape == (4, 14) and mask.any(dim=1).all()
    model = AblationReferenceModel("numeric")
    loss = masked_reconstruction_loss(model(masked), target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.numeric.parameters())

