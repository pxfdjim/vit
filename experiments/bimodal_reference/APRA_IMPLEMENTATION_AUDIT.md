# APRA implementation and gradient audit

This audit was completed before E6--E10 training. It uses no OrionBench labels.

## Differentiable alignment

For query and T3-aligned reference tokens, APRA projects concatenated numeric and
visual tokens and their first differences. The Match log potential is the sum of
content cosine similarity, structural cosine similarity, and a learned positive
quadratic distance penalty. Matches outside a three-token band receive `-1e4`.

`_monotone_marginals` performs an exact log-space forward/backward dynamic
program over an edit lattice. A complete path moves from `(0,0)` to `(Q,R)`:

- Match consumes one query and one reference token.
- Query Reject consumes one query token and preserves its unmatched evidence.
- Reference Skip consumes one reference token without consuming a query token.

The returned values are exact transition marginals. They satisfy
`sum_j Match[i,j] + Reject[i] = 1` and
`sum_i Match[i,j] + Skip[j] = 1`. Aligned reference features are the
row-normalized Match marginals multiplied by reference features. Reliability is
learned from path entropy, Match mass, reference coverage and local path
smoothness; it is not a thresholded residual.

Because the experiment uses equal 14-token query/reference sequences and exact
endpoints, every complete path has the same number of Reject and Skip moves.
Consequently, disabling either state makes the remaining gap state unreachable.
E7 and E8 retain the requested named state but test the same feasible Match-only
path family. This mathematical coupling is reported rather than hidden by a new
boundary transition.

## Residual and score construction

`unmatched = query - MatchMass * aligned_reference`. Structural residuals are
the difference between query and aligned-reference first differences. The APRA
head receives unmatched numeric/visual features, structural numeric/visual
features, aligned reference features, Reject probability and reliability. A
temporal Transformer produces a bounded `tanh` correction. Final point evidence
is restored and calibrated without labels, then scored as
`S_B3 + 0.05 * calibrated_delta`.

E9/E10 replace unmatched and structural inputs to the head with exact zeros.
They keep the same head architecture and parameter initialization, so differences
are attributable to input mechanisms rather than extra capacity.

## Loss definitions

- `L_path`: target-path cross entropy on known reversible warps, plus
  `0.25 * mean(Reject)` and `0.25 * SmoothL1(Match @ inverse_target, I)`.
- `L_detect`: hard-example-weighted softplus ranking on synthetic changed tokens,
  plus `0.5` stability loss on unchanged tokens and `0.5` stability loss under
  structure-preserving warps. B3 proxy scores are detached; APRA corrections are
  differentiable.
- `L_select`: `0.20` clean Match anchor, `0.20` synthetic normal/matchable anchor,
  `0.25` known normal-warp Match anchor, `0.10` normal Reject suppression,
  `0.20` structural Query-Reject target and `0.05` mask-free Skip anti-collapse
  prior. Its external fixed coefficient is `2.15`.

## Gradient path and detach audit

The differentiable path is
`final loss -> delta -> temporal score head -> residual/aligned/state features ->
exact Match/Reject/Skip marginals -> Match/Reject/Skip logits -> APRA and numeric
parameters`. There is no `argmax`, Viterbi path, top-k or detached alignment in
training. `no_grad` is limited to frozen visual projection and post-training
diagnostics. Detaches in `L_detect` intentionally freeze B3 proxy evidence and
hard-example weights; they do not detach APRA delta.

The correction layer is exactly zero initialized. At initialization, this makes
the first `L_detect` backward update only the correction layer. Upstream gradient
appears after that layer becomes nonzero. The previously reported `99,267x`
ratio compared `L_select` to this one-step, nearly zero-head transient.

On the same final E2 checkpoints, same first batch, and same parameter vectors
across all 11 datasets:

| loss | group | median gradient norm | parameter coverage | element coverage |
|---|---|---:|---:|---:|
| L_detect | aligner | 0.046579 | 1.000000 | 1.000000 |
| L_select | aligner | 0.171493 | 0.826087 | 0.999798 |
| L_path | aligner | 0.390868 | 0.826087 | 0.999798 |
| L_detect | residual module | 0.296986 | 1.000000 | 0.999820 |
| L_detect | scoring head | 0.008699 | 1.000000 | 1.000000 |

Mean aligner gradient cosine is `0.171238` for L_select versus L_detect and
`-0.136752` for L_path versus L_detect. The implementation therefore has no
detection-gradient break at the final checkpoint. The remaining issue is weak,
dataset-dependent magnitude and conflicting directions, not a detached graph.

