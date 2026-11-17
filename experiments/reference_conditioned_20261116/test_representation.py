"""Mechanism checks for the isolated three-way representation experiment."""
import torch

from experiments.reference_conditioned_20261116.model import Model, PosteriorOnlyAlignment
from src.models.model import AnomalyPreservingReferenceAlignment


def inputs(batch=2):
    torch.manual_seed(20261104)
    tokens = [torch.randn(batch, 14, 192) for _ in range(4)]
    anchor = torch.arange(14).float().expand(batch, -1)
    return (*tokens, anchor)


def test_zero_head_and_posterior_mass():
    output = Model("posterior").forward_tokens(*inputs())
    alignment = output["alignment"]
    assert torch.equal(output["delta"], torch.zeros_like(output["delta"]))
    assert torch.allclose(alignment["soft_match"].sum(-1) + alignment["rejection"],
                          torch.ones_like(alignment["rejection"]), atol=2e-5)
    assert torch.allclose(output["prior"].sum(-1),
                          torch.ones_like(output["prior"].sum(-1)), atol=1e-6)


def test_reference_and_posterior_interventions():
    nq, vq, nr, vr, anchor = inputs()
    model = Model("posterior")
    with torch.no_grad():
        model.fusion.out.weight.fill_(0.01)
    before = model.forward_tokens(nq, vq, nr, vr, anchor)["representation"]
    changed = model.forward_tokens(nq, vq, nr.roll(1, 1), vr.roll(1, 1), anchor)["representation"]
    assert not torch.allclose(before["joint"], changed["joint"])
    assert not torch.allclose(before["delta"], changed["delta"])
    direct = Model("t3_direct")
    direct.load_state_dict(model.state_dict(), strict=True)
    direct_rep = direct.forward_tokens(nq, vq, nr, vr, anchor)["representation"]
    assert not torch.allclose(before["numeric_reference_match"],
                              direct_rep["numeric_reference_match"])


def test_no_reference_is_invariant_to_reference_and_prior():
    nq, vq, nr, vr, anchor = inputs()
    model = Model("no_reference")
    with torch.no_grad():
        model.fusion.out.weight.fill_(0.01)
    a = model.forward_tokens(nq, vq, nr, vr, anchor)
    b = model.forward_tokens(nq, vq, nr.roll(1, 1), vr.roll(1, 1),
                             (anchor + 1).clamp_max(13))
    assert torch.allclose(a["delta"], b["delta"], atol=1e-7)
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in Model("posterior").parameters())


def test_detection_gradient_reaches_posterior_and_both_views():
    model = Model("posterior")
    with torch.no_grad():
        model.fusion.out.weight.fill_(0.01)
    nq, vq, nr, vr, anchor = inputs()
    output = model.forward_tokens(nq, vq, nr, vr, anchor)
    output["delta"].square().mean().backward()
    assert model.aligner.anchor_cost.grad.abs().sum() > 0
    assert model.fusion.numeric_projection.weight.grad.abs().sum() > 0
    assert model.fusion.visual_projection.weight.grad.abs().sum() > 0


def test_posterior_conditioned_formula_matches_manual_construction():
    model = Model("posterior")
    nq, vq, nr, vr, anchor = inputs(batch=1)
    posterior = torch.eye(14)[None] * .7
    rejection = torch.full((1, 14), .3)
    prior = torch.softmax(-.7 * (torch.arange(14)[None, None] -
                                  anchor[..., None]).square(), dim=-1)
    output = model.fusion(nq, vq, nr, vr, posterior, rejection, prior, anchor)
    q = model.fusion.numeric_projection(nq)
    r = model.fusion.numeric_projection(nr)
    matched = (posterior @ r) / .7
    nearby = prior @ r
    relation = model.fusion.relation
    def encode(reference):
        return relation(torch.cat((q, reference, q - reference,
                                   (q - reference).abs()), dim=-1))
    expected = .7 * encode(matched) + .3 * encode(nearby)
    assert torch.allclose(output["numeric"], expected, atol=1e-6)
    assert torch.allclose(output["numeric_reference_match"], matched, atol=1e-6)
    assert torch.allclose(output["numeric_reference_prior"], nearby, atol=1e-6)


def test_posterior_only_alignment_is_exactly_equivalent():
    full = AnomalyPreservingReferenceAlignment()
    compact = PosteriorOnlyAlignment()
    compact.load_state_dict(full.state_dict(), strict=True)
    data = inputs()
    a = full(*data)
    b = compact(*data)
    for key in ("soft_match", "rejection", "reference_skip", "reliability"):
        assert torch.equal(a[key], b[key])
    full_loss = a["soft_match"].square().sum() + a["rejection"].square().sum()
    compact_loss = b["soft_match"].square().sum() + b["rejection"].square().sum()
    full_loss.backward()
    compact_loss.backward()
    for name in ("anchor_cost", "gap_cost"):
        assert torch.equal(getattr(full, name).grad, getattr(compact, name).grad)
