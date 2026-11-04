import torch

from src.models.apra_trem import APRATREM, TemporalResidualEvolutionModel


def inputs(batch=3):
    g = torch.Generator().manual_seed(11)
    values = [torch.randn(batch, 14, 192, generator=g) for _ in range(4)]
    base = torch.rand(batch, 14, generator=g)
    return (*values, base)


def test_zero_trem_preserves_apra_exactly_and_is_bounded():
    model = APRATREM()
    nq, vq, nr, vr, base = inputs()
    output = model.forward_tokens(nq, vq, nr, vr, base)
    assert torch.equal(output["delta"], output["apra_delta"])
    assert torch.count_nonzero(output["trem_delta"]) == 0
    assert output["delta"].shape == (3, 14)
    assert output["delta"].abs().max() <= 1


def test_trem_uses_residuals_unmatched_reliability_and_b3_evidence():
    model = APRATREM()
    with torch.no_grad():
        model.trem.output.weight.fill_(.05)
    nq, vq, nr, vr, base = inputs()
    original = model.forward_tokens(nq, vq, nr, vr, base)["trem_delta"]
    changed_residual = model.forward_tokens(nq + .4, vq, nr, vr, base)["trem_delta"]
    changed_base = model.forward_tokens(nq, vq, nr, vr, base + .7)["trem_delta"]
    assert not torch.allclose(original, changed_residual)
    assert not torch.allclose(original, changed_base)


def test_detect_gradient_reaches_apra_and_trem_after_zero_head_step():
    model = APRATREM()
    nq, vq, nr, vr, base = inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = model.forward_tokens(nq, vq, nr, vr, base)["delta"].mean()
    optimizer.zero_grad(); first.backward(); optimizer.step()
    second = model.forward_tokens(nq, vq, nr, vr, base)["delta"].square().mean()
    optimizer.zero_grad(); second.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.aligner.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.trem.parameters())


def test_branch_ablations_remove_only_the_requested_branch():
    full = APRATREM()
    no_local = APRATREM(use_local=False)
    no_context = APRATREM(use_multiscale=False)
    assert full.trem.local is not None and full.trem.context is not None
    assert no_local.trem.local is None and no_local.trem.context is not None
    assert no_context.trem.local is not None and no_context.trem.context is None
    assert sum(p.numel() for p in no_local.parameters()) < sum(p.numel() for p in full.parameters())
    assert sum(p.numel() for p in no_context.parameters()) < sum(p.numel() for p in full.parameters())


def test_reliability_conditions_each_single_branch_ablation():
    for model in (APRATREM(use_local=False), APRATREM(use_multiscale=False)):
        with torch.no_grad():
            model.trem.output.weight.fill_(.05)
        nq, vq, nr, vr, base = inputs()
        output = model.forward_tokens(nq, vq, nr, vr, base)
        alignment = dict(output["alignment"])
        alignment["reliability"] = 1.0 - alignment["reliability"]
        changed = model.trem(alignment, output["apra_delta"], base)["trem_delta"]
        assert not torch.allclose(output["trem_delta"], changed)

