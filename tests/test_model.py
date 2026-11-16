from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model import (AnomalyPreservingReferenceAlignment, Model,
                              gradient_norms, reversible_warp)
from src.training.train import final_score
from src.inference.predict import signed_calibration
import numpy as np


def token_batch(batch=3):
    torch.manual_seed(23)
    return [torch.randn(batch, 14, 192) for _ in range(4)]


def anchor(batch):
    return torch.arange(14).float().expand(batch, -1)


def test_three_state_path_mass_is_conserved():
    model = Model().eval()
    alignment = model.forward_tokens(*token_batch(), anchor(3))["alignment"]
    rows = alignment["soft_match"].sum(-1) + alignment["rejection"]
    columns = alignment["soft_match"].sum(-2) + alignment["reference_skip"]
    assert torch.allclose(rows, torch.ones_like(rows), atol=2e-5)
    assert torch.allclose(columns, torch.ones_like(columns), atol=2e-5)
    assert torch.allclose(alignment["reliability"], 1 - alignment["rejection"], atol=2e-5)
    evidence = alignment["evidence"]
    assert evidence.shape == (3, 14, 6)
    assert torch.all((evidence >= 0) & (evidence <= 1 + 2e-5))
    assert torch.all(evidence[..., 2:4] <= alignment["rejection"][..., None] + 2e-5)


def test_zero_initialized_head_preserves_b3():
    model = Model().eval()
    delta = model.forward_tokens(*token_batch(), anchor(3))["delta"]
    b3 = torch.randn_like(delta)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(b3 + .05 * delta, b3)


def test_final_point_score_matches_inference_and_reaches_alignment():
    model = Model()
    with torch.no_grad():
        model.fusion.out.weight.fill_(1e-3)
    nq, vq, nr, vr = token_batch(4)
    prior = anchor(4)
    clean = model.forward_tokens(nq, vq, nr, vr, prior)
    synthetic = model.forward_tokens(nq + .3 * torch.randn_like(nq),
                                     vq + .3 * torch.randn_like(vq), nr, vr, prior)
    warped_n, target = reversible_warp(nq, torch.tensor([1., -1., .7, -.7]))
    warped = model.forward_tokens(warped_n, target @ vq, nr, vr,
                                  (target @ prior[..., None]).squeeze(-1))
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


def test_dtw_prior_changes_posterior_without_reordering_reference():
    model = Model().eval()
    nq, vq, nr, vr = token_batch(2)
    diagonal = anchor(2)
    delayed = (diagonal + 2).clamp_max(13)
    baseline = model.forward_tokens(nq, vq, nr, vr, diagonal)["alignment"]
    shifted = model.forward_tokens(nq, vq, nr, vr, delayed)["alignment"]
    assert not torch.allclose(baseline["soft_match"], shifted["soft_match"])
    assert baseline["aligned_numeric"].shape == shifted["aligned_numeric"].shape


def test_known_normal_time_shift_follows_reference_coordinates():
    query = torch.eye(14, 192)[None]
    reference = torch.cat((query[:, :1], query[:, :-1]), dim=1)
    prior = (torch.arange(14) + 1).clamp_max(13).float()[None]
    match = AnomalyPreservingReferenceAlignment()(
        query, query, reference, reference, prior)["soft_match"][0]
    shifted = match[1:12, 2:13].diag().mean()
    diagonal = match[1:12, 1:12].diag().mean()
    assert shifted > .8 and shifted > 10 * diagonal


def test_reference_failure_increases_rejection_and_structural_evidence():
    model = Model().eval()
    torch.manual_seed(9)
    nq = torch.nn.functional.normalize(torch.randn(3, 14, 192), dim=-1)
    vq = torch.nn.functional.normalize(torch.randn(3, 14, 192), dim=-1)
    prior = anchor(3)
    matched = model.forward_tokens(nq, vq, nq, vq, prior)["alignment"]
    unrelated = model.forward_tokens(nq, vq, nq.roll(1, 0), vq.roll(1, 0),
                                      prior)["alignment"]
    assert unrelated["rejection"].mean() > matched["rejection"].mean()
    assert unrelated["evidence"][..., 2:4].mean() > matched["evidence"][..., 2:4].mean()
    assert unrelated["evidence"][..., 4:6].mean() > matched["evidence"][..., 4:6].mean()
