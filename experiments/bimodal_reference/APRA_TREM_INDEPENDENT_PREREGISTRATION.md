# Frozen APRA and independent TREM preregistration

Date: 2026-11-04. No evaluation label is used in training, calibration, weighting, checkpoint selection, or model selection.

## Fixed components

- B3, numeric encoder, visual projection, APRA aligner and APRA evidence head are loaded from the validated per-dataset APRA-only checkpoint and frozen byte-for-byte.
- The APRA-only point score is copied as the base score. Its 0.05 correction coefficient is unchanged.
- T3 reference ids and constrained-DTW columns are reused exactly.

## Independent TREM evidence

TREM emits a zero-initialized bounded raw score `tanh(logit)` without multiplying by APRA remaining capacity. At inference, its restored point score is centered by its unlabeled median and divided by `3 * max(1.4826*MAD, 0.05)`, then clipped to `[-1, 1]`. The final score is `S_APRA-only + beta * calibrated_TREM`.

The primary coefficient is `beta=0.05`. Coefficients `0, 0.025, 0.075, 0.10` are prespecified sensitivity analyses and will not replace the primary result based on evaluation labels.

## Training and ablations

- Only TREM parameters are optimized for 16 epochs using the existing synthetic detection/ranking/stability objective and existing perturbation cache.
- Primary-seed variants: full, no Local Residual Dynamics, no Multi-scale Residual Context, no Reliability-aware Fusion, and ordinary numeric-token input without APRA residual inputs.
- Full/no-local/no-multiscale are repeated with seeds 20261104, 20261105 and 20261106. Synthetic examples stay fixed; seeds vary initialization and training order.
- The previously trained full TREM checkpoint is also evaluated after removing only its old capacity multiplier, before retraining.

The report will present the primary beta first, all fixed beta sensitivities, seed means/standard deviations, per-dataset and scope metrics, event recovery and false positives, parameter/runtime costs, frozen-parameter equality, gradient norms, stage amplitudes and input replacement effects.
