from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model import Model, gradient_norms, reversible_warp
from src.training.train import final_score
from src.inference.predict import signed_calibration
import numpy as np


def token_batch(batch=3):
    torch.manual_seed(23)
    return [torch.randn(batch, 14, 192) for _ in range(4)]


def test_three_state_path_mass_is_conserved():
    model = Model().eval()
    alignment = model.forward_tokens(*token_batch())["alignment"]
    rows = alignment["soft_match"].sum(-1) + alignment["rejection"]
    columns = alignment["soft_match"].sum(-2) + alignment["reference_skip"]
    assert torch.allclose(rows, torch.ones_like(rows), atol=2e-5)
    assert torch.allclose(columns, torch.ones_like(columns), atol=2e-5)


def test_zero_initialized_head_preserves_b3():
    model = Model().eval()
    delta = model.forward_tokens(*token_batch())["delta"]
    b3 = torch.randn_like(delta)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(b3 + .05 * delta, b3)


def test_final_point_score_matches_inference_and_reaches_alignment():
    model = Model()
    with torch.no_grad():
        model.fusion.out.weight.fill_(1e-3)
    nq, vq, nr, vr = token_batch(4)
    clean = model.forward_tokens(nq, vq, nr, vr)
    synthetic = model.forward_tokens(nq + .3 * torch.randn_like(nq),
                                     vq + .3 * torch.randn_like(vq), nr, vr)
    warped_n, target = reversible_warp(nq, torch.tensor([1., -1., .7, -.7]))
    warped = model.forward_tokens(warped_n, target @ vq, nr, vr)
    length = 224 + 56 * 3
    base = torch.linspace(.2, 1.2, length, dtype=torch.float64)
    score, _ = final_score(synthetic["delta"], base)
    reference, _ = signed_calibration(synthetic["delta"].detach(), base.numpy(),
                                      length, torch.device("cpu"))
    assert np.allclose(score.detach().numpy(), base.numpy() + .05 * reference, atol=1e-12)
    clean_score, _ = final_score(clean["delta"], base)
    warp_score, _ = final_score(warped["delta"], base)
    loss = (score - clean_score).square().mean() + (warp_score - clean_score).square().mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    norms = gradient_norms(model)
    assert norms["alignment"] > 0 and norms["scoring_head"] > 0
