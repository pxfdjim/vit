# Temporal Matching ViT4TS

Cache-only evaluation of temporal constraints on the existing paired Block-6 baseline. It preserves frozen OpenAI CLIP ViT-B/16, 224/56 windows, full 14x14 Patch features, the shared 32-window reference pool, stable large/mid scales, and original score recovery.

## Variants

- `A_stable_final`, `B_paired_2d_final`, and `B_paired_2d_mid` reproduce existing controls.
- `T1_independent` independently selects a reference column for every query column.
- `T2_shift_consistent` selects one reference window and one shift in `[-2,2]`.
- `T3_constrained_dtw` uses monotone endpoint-constrained DTW with band 2 and a label-free fixed-scale non-diagonal penalty.
- `T4_*` and `T5_*` preserve B-mid and fuse calibrated T2/T3 scores at separately reported lambda 0.25 and 0.5. Labels never select lambda.

## Safe three-worker run

Use a new output directory. `--final-cache-root` is repeatable and read-only; `--cache-root` points to the existing Block-6 cache. Run dry-run for all three shards first.

```bash
python experiments/temporal_matching/runner.py --dry-run --worker-id 0 --num-workers 3 \
  --cache-root /home/pengxf/jim/VLM4TS/data/runs/curve_aware_cache_20261028_ac7e288/curve_cache \
  --final-cache-root /tmp/VLM4TS-vfg-paired-d593dfb/data/runs/vfg_paired_20261027/feature_cache \
  --final-cache-root /tmp/VLM4TS-vfg-paired-d593dfb/data/runs/vfg_paired_yahoo_20261028/feature_cache
```

First run only `--variants B_paired_2d_mid` on all three workers, then validate before starting the remaining variants:

```bash
python experiments/temporal_matching/validate_results.py --baseline-only --output-root "$OUT" \
  --previous-root /tmp/VLM4TS-curve-aware-c433a60/data/runs/curve_aware_20261028_01
```

A successful baseline check requires all 492 scores to differ by at most `1e-6` and the all-11 macro to equal `0.657370` within `1e-6`. Continue the same output with the other comma-separated variants. Existing artifacts are validated and kept; publication uses atomic hard links and never replaces files.

After all variants finish:

```bash
MPLCONFIGDIR=/tmp/temporal-mpl python experiments/temporal_matching/summarize.py --output-root "$OUT"
python experiments/temporal_matching/validate_results.py --output-root "$OUT" \
  --previous-root /tmp/VLM4TS-curve-aware-c433a60/data/runs/curve_aware_20261028_01
```

Outputs include point scores, contextual metrics at all three alphas, macro scopes, event changes, runtimes, candidate-pool audit, T2/T3 path arrays, path summaries, plots, and `experiment_report.md`.
