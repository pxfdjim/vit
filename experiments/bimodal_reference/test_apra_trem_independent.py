import numpy as np
import torch
from src.models.apra_trem_independent import FrozenAPRAIndependentTREM, robust_signed_calibration

def tensors():
    g=torch.Generator().manual_seed(7)
    return [torch.randn(2,14,192,generator=g) for _ in range(4)] + [torch.rand(2,14,generator=g)]

def test_zero_head_preserves_apra_and_freezes_base():
    model=FrozenAPRAIndependentTREM(); model.freeze_base()
    nq,vq,nr,vr,b=tensors(); out=model.forward_tokens(nq,vq,nr,vr,b)
    assert torch.equal(out["delta"],out["apra_delta"])
    assert all(not p.requires_grad for name,p in model.named_parameters() if not name.startswith("trem."))

def test_independent_output_not_limited_by_apra_capacity():
    model=FrozenAPRAIndependentTREM(); model.freeze_base()
    with torch.no_grad(): model.trem.output.weight.fill_(.05)
    nq,vq,nr,vr,b=tensors(); out=model.forward_tokens(nq,vq,nr,vr,b)
    assert out["trem_raw"].abs().max()>0
    assert torch.equal(out["trem_raw"],out["trem_delta"])

def test_ablations_and_numeric_input():
    assert FrozenAPRAIndependentTREM(use_local=False).trem.local is None
    assert FrozenAPRAIndependentTREM(use_multiscale=False).trem.context is None
    assert FrozenAPRAIndependentTREM(use_reliability=False).trem.reliability_fusion is None
    assert FrozenAPRAIndependentTREM(input_mode="numeric").trem.input_mode=="numeric"

def test_calibration_is_centered_bounded_and_has_floor():
    out,stats=robust_signed_calibration(np.array([1.,1.,1.,2.]))
    assert np.max(np.abs(out))<=1 and stats["scale"]>=.05
