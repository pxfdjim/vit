import torch

from src.models.apra_acef import (
    APRAACEF, LossWeights, detection_loss, explanation_loss, gradient_norms,
    path_loss, reversible_warp, selective_alignment_loss, selective_time_warp,
)


def token_batch(batch=3):
    torch.manual_seed(23)
    return [torch.randn(batch,14,192) for _ in range(4)]


def test_apra_outputs_are_complete_and_path_mass_is_conserved():
    model=APRAACEF("apra","acef",True).eval(); nq,vq,nr,vr=token_batch()
    output=model.forward_tokens(nq,vq,nr,vr); alignment=output["alignment"]
    assert output["delta"].shape==(3,14)
    assert alignment["soft_match"].shape==(3,14,14)
    assert torch.allclose(alignment["soft_match"].sum(-1)+alignment["rejection"],
                          torch.ones(3,14),atol=2e-4)
    assert ((alignment["reliability"]>=0)&(alignment["reliability"]<=1)).all()
    assert output["shared_evidence"].shape==output["complementary_evidence"].shape==(3,14,192)


def test_zero_correction_preserves_b3_exactly():
    model=APRAACEF("apra","acef",True).eval(); inputs=token_batch()
    output=model.forward_tokens(*inputs)
    assert torch.count_nonzero(output["delta"])==0
    b3=torch.randn(3,14)
    assert torch.equal(b3+.05*output["delta"],b3)


def test_three_losses_reach_expected_modules_after_head_activation():
    model=APRAACEF("apra","acef",True)
    with torch.no_grad(): model.fusion.correction.weight.fill_(1e-3)
    nq,vq,nr,vr=token_batch(4); clean=model.forward_tokens(nq,vq,nr,vr)
    ns=nq+.3*torch.randn_like(nq); vs=vq+.3*torch.randn_like(vq)
    synthetic=model.forward_tokens(ns,vs,nr,vr)
    warped_n,target=reversible_warp(nq,torch.tensor([1.,-1.,.7,-.7]))
    warped=model.forward_tokens(warped_n,target@vq,nr,vr)
    altered=torch.zeros(4,14,dtype=torch.bool); altered[:,4:8]=True
    weights=LossWeights(); clean_base=torch.zeros(4,14); synth_base=torch.zeros(4,14)
    losses=[path_loss(warped,target), explanation_loss(clean,~altered),
            detection_loss(clean,synthetic,warped,clean_base,synth_base,altered,weights)]
    for loss in losses:
        model.zero_grad(set_to_none=True); loss.backward(retain_graph=True); norms=gradient_norms(model)
        assert norms["all"]>0
    model.zero_grad(set_to_none=True); losses[-1].backward(); norms=gradient_norms(model)
    assert norms["apra"]>0 and norms["acef"]>0


def test_selective_alignment_targets_and_gradients():
    model=APRAACEF("apra","apra_head",True)
    nq,vq,nr,vr=token_batch(6); clean=model.forward_tokens(nq,vq,nr,vr)
    ns=nq.clone(); vs=vq.clone(); ns[:,4:8]+=1.; vs[:,4:8]-=.5
    synthetic=model.forward_tokens(ns,vs,nr,vr)
    warped_n,target,modes=selective_time_warp(nq)
    warped=model.forward_tokens(warped_n,target@vq,nr,vr)
    altered=torch.zeros(6,14,dtype=torch.bool); altered[:,4:8]=True
    structural=altered.clone(); structural[1::2]=False
    loss,parts=selective_alignment_loss(clean,synthetic,warped,target,altered,structural)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts)=={"clean_anchor","synthetic_anchor","warp_match","normal_reject",
                        "structural_reject","skip_prior"}
    assert set(modes.tolist())=={0,1,2}
    assert sum(float(p.grad.square().sum()) for p in model.aligner.parameters()
               if p.grad is not None)>0
    coordinates=(target*torch.arange(14,dtype=target.dtype)[None,None]).sum(-1)
    assert torch.all(coordinates[:,1:]>=coordinates[:,:-1]-1e-6)


def test_controls_are_distinct_and_reference_changes_output():
    nq,vq,nr,vr=token_batch(); outputs=[]
    for aligner,fusion in (("apra","acef"),("fixed_dtw","acef"),
                           ("cross_attention","acef"),("apra","gated")):
        model=APRAACEF(aligner,fusion,True).eval()
        with torch.no_grad():
            if hasattr(model.fusion,"correction"): model.fusion.correction.weight.fill_(.01)
            elif hasattr(model.fusion,"out"): model.fusion.out.weight.fill_(.01)
            outputs.append(model.forward_tokens(nq,vq,nr,vr)["delta"])
    assert all(out.shape==(3,14) for out in outputs)
    assert not torch.allclose(outputs[0],outputs[1])
    model=APRAACEF("apra","acef",True).eval()
    with torch.no_grad():
        model.fusion.correction.weight.fill_(.01)
        left=model.forward_tokens(nq,vq,nr,vr)["delta"]
        right=model.forward_tokens(nq,vq,nr.roll(1,0).flip(1),vr.roll(1,0).flip(1))["delta"]
    assert not torch.allclose(left,right)
