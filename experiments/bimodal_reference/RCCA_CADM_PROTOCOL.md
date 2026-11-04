# Preregistered RCCA/CADM experiment on OrionBench

Date: 2026-11-04

This protocol is fixed before RCCA/CADM model training or evaluation. Evaluation
labels are not read by cache preparation, training, checkpoint selection, score
calibration, or hyperparameter selection.

## Immutable data and score protocol

- Eleven OrionBench datasets and all 492 indexed signals are used. MSL and SMAP
  use official NASA Train for fitting/training and official Test for evaluation.
  Yahoo and NAB use the complete unlabeled sequence transductively.
- Window length is 224, stride is 56, and training lasts 16 epochs. Every model
  is trained independently per dataset and the final epoch is used.
- T3 visual references and constrained-DTW paths are immutable. Training pairs
  use T3 retrieval within each signal and exclude overlapping/neighboring query
  windows through the existing common reference pool.
- Primary score weights are fixed to visual/raw/numeric/cross =
  `0.90/0.05/0.025/0.025`. M6 reallocates the removed cross budget to numeric,
  giving `0.90/0.05/0.05/0.00`. Total added evidence is always 0.10.
- Each evidence channel is calibrated without labels by positive-median matching
  to the immutable T3 score before fixed fusion.

## Model

- The numeric encoder is initialized from the immutable Full AMR checkpoint.
  The frozen visual column projection and frozen OpenAI CLIP ViT-B/16 Block-6
  features are unchanged.
- RCCA contains numeric-query/reference-visual and visual-query/reference-numeric
  attention, learned 14-position embeddings, residual connections, LayerNorm,
  and a 192-dimensional conditional fusion. Query and reference-self conditional
  tokens are both constructed and differenced.
- CADM receives signed/absolute visual and numeric query-reference differences,
  RCCA conditional differences, and cross-modal reference matching products. A
  192-dimensional projection, one temporal Transformer layer, and an MLP output
  one logit per Patch.

## Synthetic reference-conditioned objective

- One deterministic synthetic counterpart is generated per training window and
  seed. Perturbation types are uniformly sampled from local amplitude jump,
  trend shift, frequency change, volatility burst, and subsequence morphology.
  A contiguous span of 2--5 Patches is altered; the exact Patch mask is stored.
- The perturbed numeric window is transformed through the same frozen affine
  image normalization, rendered again, and encoded by frozen CLIP. Numeric and
  visual branches therefore receive the same perturbed signal.
- Fixed loss: `L = L_patch + 1.0 L_synthetic + 0.5 L_ranking + 0.25 L_consistency`.
  Synthetic loss is balanced BCE over stored Patch masks. Ranking uses margin
  0.5 between perturbed and clean logits on altered Patches. Consistency is
  Smooth-L1 between clean and perturbed logits on untouched Patches.
- Original windows are never assigned all-normal labels. `L_patch` is the AMR
  direct numeric/visual Patch consistency on query and reference windows.

## Variants

- M1: RCCA + CADM + all fixed losses.
- M2: no RCCA; the same CADM receives zero conditional differences and retains
  visual/numeric/reference matching inputs.
- M3: query same-window Cross-Attention replaces reference-conditioned query
  Cross-Attention; reference-self construction is retained.
- M4: RCCA retained; synthetic BCE and ranking weights are zero. Patch and
  untouched-position consistency remain.
- M5: inference-only M1 control with deterministically shuffled reference Token
  identities; all other score channels and T3 routing remain fixed.
- M6: inference-only M1 control with `S_cross` removed and its 0.025 budget
  returned to `S_num`.

## Seeds and decision rules

- Primary seed: 20261104. M1 stability seeds: 20261104, 20261105, 20261106.
- M1--M4 primary models and M1 stability models run on all eleven datasets.
- No variant, loss weight, perturbation severity, epoch, or score weight is
  selected from evaluation labels. All results are reported even if worse.
- Metrics are Aff-F1 (`alpha=0.01`, complete miss with events is zero), VUS-PR,
  and VUS-ROC, using the existing common-valid-signal rule.

## Required integrity gates

- Unit tests must verify dimensions, finite outputs, nonzero RCCA/CADM gradients,
  reference replacement sensitivity, and actual query/reference influence.
- Inference must reproduce immutable T3 references and DTW paths exactly.
- M5 must alter RCCA/CADM outputs while leaving visual/raw/numeric channels fixed.
- M6 must have exact zero cross contribution. All checkpoints, caches, scores,
  diagnostics, manifests, and reports are written under a new run directory.
