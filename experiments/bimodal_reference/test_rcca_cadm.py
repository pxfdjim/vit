import torch

from src.models.rcca_cadm import (
    LossWeights, RCCACADM, gradient_norms, reference_conditioned_loss,
)


def batch(batch_size=4):
    torch.manual_seed(7)
    raw = torch.randn(batch_size, 224)
    reference = torch.randn(batch_size, 224)
    visual = torch.randn(batch_size, 196, 768)
    reference_visual = torch.randn(batch_size, 196, 768)
    columns = torch.arange(14).repeat(batch_size, 1)
    return raw, reference, visual, reference_visual, columns


def test_rcca_cadm_dimensions_and_attention():
    model = RCCACADM("rcca")
    output = model(*batch())
    assert output["logits"].shape == (4, 14)
    assert output["conditional_difference"].shape == (4, 14, 192)
    assert output["attention"]["query_n_to_v_attention"].shape == (4, 4, 14, 14)
    assert torch.isfinite(output["logits"]).all()


def test_all_new_modules_receive_gradient():
    model = RCCACADM("rcca")
    raw, reference, visual, reference_visual, columns = batch()
    clean = model(raw, reference, visual, reference_visual, columns)
    synthetic = model(raw + 0.7, reference, visual + 0.2, reference_visual, columns)
    mask = torch.zeros(4, 14, dtype=torch.bool); mask[:, 3:7] = True
    loss, _ = reference_conditioned_loss(clean, synthetic, mask, LossWeights())
    loss.backward()
    norms = gradient_norms(model)
    assert norms["numeric"] > 0
    assert norms["rcca"] > 0
    assert norms["cadm"] > 0


def test_real_query_and_reference_change_output():
    model = RCCACADM("rcca").eval()
    raw, reference, visual, reference_visual, columns = batch()
    with torch.no_grad():
        base = model(raw, reference, visual, reference_visual, columns)["logits"]
        changed_query = model(raw + 1.0, reference, visual, reference_visual, columns)["logits"]
        changed_reference = model(raw, reference.roll(1, 0), visual,
                                  reference_visual.roll(1, 0), columns)["logits"]
    assert not torch.allclose(base, changed_query)
    assert not torch.allclose(base, changed_reference)


def test_no_rcca_and_same_window_controls_are_operationally_distinct():
    inputs = batch()
    torch.manual_seed(9); no_rcca = RCCACADM("no_rcca").eval()
    torch.manual_seed(9); same = RCCACADM("same_window").eval()
    with torch.no_grad():
        left = no_rcca(*inputs)["logits"]
        right = same(*inputs)["logits"]
    assert left.shape == right.shape == (4, 14)
    assert not torch.allclose(left, right)
