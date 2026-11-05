# APRA selective-alignment preregistration

This protocol is fixed before any E2--E5 final-epoch checkpoint is trained or
any corresponding OrionBench label is opened.

## Immutable protocol

- E0 is the immutable B3 score. E1 reuses the three validated APRA-only
  `L_path + L_detect` seeds. E2--E5 use the same Full-AMR numeric initialization,
  frozen visual projection, cached paired synthetic views, T3 references, 16
  epochs, optimizer, batch size and final-epoch rule.
- Final scoring remains `S_B3 + 0.05 * calibrated_delta_APRA`. No extra evidence
  budget, validation split, label-selected coefficient or checkpoint selection is
  introduced.
- E2 uses `L_select + L_detect`; E3 removes only structural Query-Reject
  supervision; E4 removes only the known normal-warp correspondence term; E5
  replaces learned APRA alignment with fixed T3-aligned identity correspondence
  and therefore trains with `L_detect` only.

## Selective target construction

- The fixed T3 reference columns are gathered before APRA, so the clean target is
  the 14-token identity correspondence. Known smooth shift, local stretch and
  smooth deformation maps are composed with this target exactly at token level.
- Frequency change, volatility burst and morphology are preregistered as
  correspondence-breaking perturbations and supervise Query Reject only inside
  the cached Patch mask. Amplitude jump and trend shift retain a reasonable time
  correspondence, so they supervise Match and expose their magnitude/trend
  residual to the scoring head.
- Unchanged synthetic tokens keep the clean T3 anchor. Reference Skip receives
  no anomaly target and retains only a weak mask-free clean anti-collapse prior.

## Loss balance

Before training, one fixed, label-free initial batch from every dataset is used to
measure `L_select` and `L_detect`. `lambda_select` is set once to
`round_0.05(clip(median(L_detect)/median(L_select), 0.25, 4.0))`. The resulting
number and raw measurements are committed before launching E2--E5. Evaluation
labels are absent from this calculation.

## Evaluation

All 11 datasets and 492 signals use the existing OrionBench-native protocol.
Aff-F1, VUS-PR and VUS-ROC are reported by dataset and for All-11, Real-6 and
Yahoo-4. Three seeds are aggregated with population standard deviation. Event
hits, misses and false-positive intervals, path state statistics, known-warp
target mass, synthetic reject separation, real-anomaly rejection and top-decile
normal-volatility behavior are retained. Statistical tests are descriptive and
do not select models.
