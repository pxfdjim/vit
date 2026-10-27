# Paired visual-frequency experiment

This directory owns orchestration and reporting only. The backend is
`src.models.vfg_paired_vit4ts` and is imported only for a real run.

- Seven datasets contain 125 complete single-variable signals.
- Three or more workers receive disjoint deterministic slices.
- `--cache-root` explicitly selects the existing visual cache. The expected
  cache payload contains `cache_version`, `input_digest`, `large`, `mid`,
  `patch`, `large_mask`, and `mid_mask`.
- Point scores, metrics, summaries, logs, and plots are atomically published or
  retained when already present. Nothing is replaced.
- Dataset F1-max first averages signal contextual F1 at each alpha, then chooses
  among 0.1, 0.01, and 0.001. Both six-real-dataset and seven-dataset macros are
  reported.
- Runtime is cache-only. It excludes CLIP rendering and extraction. The backend
  `inference_seconds` is reported as variant incremental runtime (matching +
  frequency + score recovery), while shared visual pool, shared Patch distance,
  Stable shared work, and score recovery remain separate columns.

CPU-only input validation:

```bash
/home/pengxf/anaconda3/envs/vit/bin/python experiments/vfg_paired/runner.py \
  --dry-run --num-workers 3 --worker-id 0
```

After checking resources, use at least three distinct worker labels. CPU labels
are supported when fewer than three GPUs are fully free:

```bash
DEVICES=cuda:5,cpu:0,cpu:1 LIMIT_PER_WORKER=1 SUMMARIZE=0 \
  experiments/vfg_paired/run_parallel.sh
```

Generate the controlled B comparison and validate every saved score:

```bash
/home/pengxf/anaconda3/envs/vit/bin/python experiments/vfg_paired/finalize_analysis.py \
  --output-root data/runs/vfg_paired_20261027
/home/pengxf/anaconda3/envs/vit/bin/python experiments/vfg_paired/validate_results.py \
  --output-root data/runs/vfg_paired_20261027 \
  --previous-root /path/to/previous/vfg_run
```
