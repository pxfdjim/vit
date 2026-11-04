# APRA + TREM preregistration

- E0 reuses immutable B3 point scores. E1 reuses immutable APRA-only checkpoints and point scores.
- E2–E4 initialize the numeric encoder, frozen visual projection, APRA aligner and APRA evidence head from the corresponding validated E1 checkpoint.
- APRA is unchanged. TREM consumes only APRA structural residuals, unmatched features, reliability/rejection, the APRA score and a frozen 0.90/0.05/0.05 B3 token proxy.
- TREM width is 128. Its zero initialized output is bounded by `0.25 * (1 - |S_APRA_token|)`. The final token evidence stays in the existing APRA 0.05 correction budget.
- E2 uses local residual dynamics and dilation 1/2/4 context. E3 removes the local branch. E4 removes the multi-scale branch.
- Training uses the existing APRA-only `L_path + L_detect`, 16 epochs, seed 20261104 and the immutable synthetic cache. No new loss, perturbation family, evaluation label, validation split or checkpoint selection is introduced.
- Primary comparison is E2 against E1 on All-11 Aff-F1/VUS-PR/VUS-ROC. Real-6, Yahoo-4, five focus datasets, event hits, false-positive intervals, parameters and inference time are reported without post-hoc selection.
