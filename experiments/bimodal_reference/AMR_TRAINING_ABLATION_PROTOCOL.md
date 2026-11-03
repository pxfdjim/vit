# AMR training ablation protocol

This protocol is fixed before any new checkpoint is trained or scored.

## Scope

The experiment contains numerical Transformer efficacy, four cross-modal loss
ablations plus the all-loss removal, and Cross-Attention necessity. Experiment
4 and multi-seed stability are excluded by the latest instruction.

Full and C1 share the eleven immutable dataset-native Full checkpoints. T2 and
C4 share one independently trained numeric-only checkpoint per dataset because
their architecture, objective, input, and scoring are identical. Every other
trained configuration receives an independent dataset checkpoint.

## Fixed protocol

- MSL and SMAP train on official NASA Train and evaluate official Test.
- Yahoo and NAB train without labels on the complete sequence and evaluate the
  same complete sequence.
- Window 224, stride 56, 16 epochs, batch size 32, AdamW learning rate 3e-4 and
  weight decay 0.02 are fixed.
- B3 is always 0.90 T3 visual score + 0.05 raw residual score + 0.05 learned
  numeric-token score. T3 references and constrained-DTW paths are immutable.
- No anomaly label is opened by training, calibration, weighting, checkpoint
  selection, or feature diagnostics.

## Objectives and architecture adjustments

The four named loss groups exactly partition the Full objective. Patch
Consistency contains direct numeric/visual Patch alignment and the original
0.25-weight fused-token Patch alignment when a fusion module exists. The other
groups are the 0.5-weight full temporal relation matrix, 0.25-weight adjacent
transition geometry, and 0.5-weight reference geometry.

C2 has no interaction or fusion module, so fused-token alignment is
inapplicable; its direct Patch, temporal relation, transition, and reference
terms remain. C3 replaces bidirectional attention with an MLP Late-joint head,
so its fused Patch term trains that head. C4/T2 contains only the unchanged
numeric Transformer and a training-only linear decoder. It uses masked
numeric-Patch reconstruction with a fixed 25% mask. The decoder is absent from
B3 inference.

L1 removes both direct and fused Patch consistency. L5 removes every group.
The published Full objective has no independent numeric self-supervised term,
so L5 has no valid gradient and receives zero optimizer updates. No replacement
objective is introduced. With the common seed its numeric encoder is expected
to be identical to T1; this equality is audited rather than assumed.

## Staging and decisions

T1, T2/ C4, Full/C1, C2, and C3 are completed first. L1-L5 are then executed
under this already fixed protocol. No configuration, epoch, weight, or dataset
is selected from evaluation labels.
