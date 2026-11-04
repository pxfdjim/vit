# APRA-only loss ablation and score-weight sensitivity preregistration

- E0 uses `L_path + L_detect`; seed 20261104 is reused byte-for-byte from the immutable APRA-only run, while seeds 20261105 and 20261106 are newly trained.
- E1 removes `L_path`; E2 removes `L_detect`. No replacement loss or supervision is introduced. All new checkpoints use the same model, numeric/visual initialization procedure, immutable synthetic cache, 16 epochs, optimizer, batch size, and fixed final epoch.
- The three seeds change only random module initialization and sample order. Within a seed, all loss variants start from identical weights and order.
- Score sensitivity reuses the existing seed-20261104 APRA correction and evaluates gamma in `{0, 0.025, 0.05, 0.075, 0.10, 0.15}`. Gamma 0.05 is the primary historical setting; the scan is descriptive and cannot select a replacement.
- Training and score calibration do not read evaluation labels. Evaluation labels are opened only after all point scores are fixed.
- Primary metrics are Aff-F1, VUS-PR, and VUS-ROC on All-11, Real-6, Yahoo-4, and every dataset. Mechanism diagnostics are path entropy, rejection, reliability, alignment displacement, event hits, and false-positive intervals.
