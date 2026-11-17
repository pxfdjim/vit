# Reference-Conditioned Anomaly Representation Network

This is an isolated experiment. The released model and its checkpoints remain
unchanged. Only the source files in this directory are tracked; generated
checkpoints, scores, figures, and reports live in the Git-ignored `run/` tree.

## Method

The frozen T3 reference selector supplies reference window IDs and the local
position centers `a_i`. The existing three-state forward/backward aligner
returns Match edge marginals `P_ij`, Query Reject mass `u_i`, and Reference Skip
mass. For each modality `m`:

The experimental aligner computes exactly the same posterior as the released
aligner but skips its obsolete six-score statistics. A unit test checks
posterior values and gradients elementwise; an actual 2,380-point signal score
is bitwise identical before and after this compute-only simplification.

```
p_i = Σ_j P_ij
π_ij = softmax_j(-softplus(anchor_cost) (j-a_i)^2)
R_match_i^m = Σ_j P_ij R_j^m / max(p_i,1e-8)
R_prior_i^m = Σ_j π_ij R_j^m
Fθ(Q,R) = GELU(W₂ GELU(W₁ [Q,R,Q-R,|Q-R|]))
H_i^m = p_i Fθ(Q_i^m,R_match_i^m) + u_i Fθ(Q_i^m,R_prior_i^m)
H_i = GELU(W₄ GELU(W₃ [H_i^n,H_i^v,|H_i^n-H_i^v|]))
δ_i = tanh(wᵀ H_i + b)
```

Numeric and visual tokens each have a separate 192→64 linear projection; the
two modalities share `Fθ`. The joint 192→64→64 network has its own parameters.
The final linear head is zero initialized. The final point score is the
released `B3 + 0.05 × calibrated(δ)`, with the same differentiable point
restoration and positive-median unlabeled calibration in training and inference.

`posterior` is the proposed model. `no_reference` sets reference tokens to
zero and `(p,u)=(1,0)` within the added representation network, while keeping
exactly the same parameter tensors and frozen B3 score. `t3_direct` takes the
reference token at the rounded T3 center instead of the posterior-weighted
reference token, while retaining `p` and `u` to isolate the reference
aggregation mechanism. All three compute the same frozen T3 selector, APRA
posterior, and no-label Detect-only training objective.

## Protocol

Run `launch.py` on four GPUs, then `complete.py` for inference, immutable
OrionBench evaluation, unlabeled mechanism diagnostics, integrity validation,
and the Markdown report. The baseline A scores are reused from
`data/runs/model_refactor_v2_20261116_eval/`. No test label is read during
training, inference, or the normal-warp/local-injection mechanism analysis.
The evaluator alone opens labels, after the final-epoch checkpoints are fixed.

This is a single-seed comparison. Differences are observations and should not
be used to select a new default architecture without independent replication.
