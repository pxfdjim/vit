# APRA mechanism ablation preregistration

This protocol is fixed before E6--E10 full training or evaluation.

## Reuse and immutable settings

- E0/B3 and E1--E5 scores are reused byte-for-byte from the validated selective
  APRA run. Source/config/seed hashes are checked before aggregation.
- E6--E10 use the same Full-AMR numeric and visual initialization, unlabeled
  synthetic cache, T3 reference IDs/columns, 16 epochs, optimizer, batch order,
  three seeds, fixed final epoch and `gamma=0.05` as E2--E5.
- All new variants use only `L_detect`. No evaluation label, validation split,
  score weight search or checkpoint selection is introduced.

## New variants

- E6 Detection-only: full Match/Query-Reject/Reference-Skip APRA and full APRA
  residual head.
- E7 No Query Reject: Query-Reject log potential is disabled; Match and
  Reference Skip code remains.
- E8 No Reference Skip: Reference-Skip log potential is disabled; Match and
  Query Reject code remains.
- E9 No structural residual: full three-state alignment; unmatched and structural
  residual inputs to the score head are zeroed.
- E10 Match-only: Query Reject and Reference Skip are disabled and unmatched/
  structural score inputs are zeroed.

Disabled modules remain instantiated to preserve the same initialization stream
and nominal capacity, but their transition logits or head inputs are constants
and they receive zero functional gradient. Active parameter and gradient counts
are reported separately.

## Required audits

- Exact row/column marginal conservation and disabled-state mass are checked for
  every checkpoint.
- `L_detect` must update the score head after the zero-head first step. It must
  update the aligner when both gap states are enabled.
- T3 reference/path arrays, B3, and reused E1--E5 point scores must reproduce
  exactly.
- Evaluation covers 11 datasets, 492 signals, common 488 valid signals, three
  seeds, Aff-F1, VUS-PR, VUS-ROC, events, path behavior, real anomaly versus
  volatile normal states, parameters and runtime.

