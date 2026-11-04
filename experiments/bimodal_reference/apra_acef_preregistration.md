# APRA + ACEF preregistration

Date: 2026-11-04 (Asia/Shanghai)

This experiment tests anomaly preserving reference alignment (APRA) and
conditional explanation driven complementary fusion (ACEF) under the existing
OrionBench native protocol. No evaluation label is available to training,
calibration, checkpoint selection, loss weighting, or model selection.

## Fixed protocol

- Eleven datasets and 492 evaluation signals.
- MSL/SMAP use official Train statistics and Test evaluation. Yahoo and NAB use
  complete sequence unlabeled transductive training and evaluation.
- Window 224, step 56, 16 fixed epochs, final epoch checkpoint.
- Frozen T3 visual retrieval, T3 constrained DTW anchor, frozen B3 branch.
- Final score: `S_B3 + 0.05 * DeltaS`; `DeltaS` is bounded and calibrated with
  unlabeled robust statistics. Its final linear correction layer is zero
  initialized.
- Aff-F1 uses alpha 0.01 and complete misses score zero. VUS-PR and VUS-ROC use
  continuous scores. The same four all-normal signals are excluded for every
  method.

## Model definition

APRA uses a differentiable monotone path distribution anchored to the fixed T3
DTW path. Its path lattice contains match, query rejection, and reference skip
transitions. It returns match marginals, query rejection probability, alignment
reliability, aligned reference features, unmatched query features, and local
structural residuals. Reliability is predicted from path entropy, path coverage,
and contextual features; raw residual magnitude is not an input to that head.

ACEF uses two real cross-attention explanation branches. Visual reference tokens
predict numeric query dynamics and numeric reference tokens predict visual query
structure. Prediction residuals form shared and complementary evidence, which is
modeled by a temporal Transformer together with every APRA output. The bounded
signed correction may increase or decrease B3.

## Only three optimization losses

- `L_path`: known reversible monotone feature warp mapping supervision, inverse
  mapping consistency, and non-degenerate rejection supervision.
- `L_explain`: bidirectional feature prediction on reliable, unaltered regions.
- `L_detect`: hard synthetic trend/period/morphology detection, unchanged-region
  consistency, and structure-preserving warp consistency on the final score.

The registered weights are path 1.0, explain 0.5, detect 1.0. Synthetic hard
examples receive a fixed weight based only on insufficient frozen B3 proxy
response. `L_detect` must have nonzero gradients to APRA and ACEF after the
zero-initialized correction head's first update.

## Registered configurations

- `APRA_only`: APRA plus a temporal APRA evidence correction head. This is also
  the condition-explanation branch ablation.
- `ACEF_only_fixed_dtw`: identity T3 alignment plus ACEF. This is also the fixed
  DTW control.
- `APRA_ACEF`: complete model.
- `no_reject`: complete model with query rejection disabled.
- `no_Lpath`, `no_Lexplain`, `no_Ldetect`: one registered loss weight is zero.
- `ordinary_cross_attention`: APRA replaced by band-limited ordinary attention.
- `gated_fusion`: ACEF replaced by a conventional gated residual fusion head.
- `random_numeric`: complete model with a frozen random numeric encoder.

No post-evaluation variant or score weight will replace a preregistered primary
configuration. The primary comparison is `APRA_ACEF` against B3 and the completed
RCCA-CADM M1 result. All other configurations are mechanism controls.
