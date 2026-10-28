# Paired visual-frequency experiment

This directory owns orchestration and reporting only. The backend is
`src.models.vfg_paired_vit4ts` and is imported only for a real run.

- `config.json` contains the original seven datasets and 125 complete
  single-variable signals. `config_yahoo.json` contains the 367 indexed Yahoo
  A1-A4 signals.
- Three or more workers receive disjoint deterministic slices.
- `--cache-root` explicitly selects the existing visual cache. The expected
  cache payload contains `cache_version`, `input_digest`, `large`, `mid`,
  `patch`, `large_mask`, and `mid_mask`.
- Point scores, metrics, summaries, logs, and plots are atomically published or
  retained when already present. Nothing is replaced.
- Dataset F1-max first averages signal contextual F1 at each alpha, then chooses
  among 0.1, 0.01, and 0.001. Macro scopes come from the selected configuration:
  `real_6`/`all_7` for the original run and `yahoo_4` for Yahoo.
- Runtime is cache-only. It excludes CLIP rendering and extraction. The backend
  `inference_seconds` is reported as variant incremental runtime (matching +
  frequency + score recovery), while shared visual pool, shared Patch distance,
  Stable shared work, and score recovery remain separate columns.

CPU-only input validation:

```bash
/home/pengxf/anaconda3/envs/vit/bin/python experiments/vfg_paired/runner.py \
  --dry-run --num-workers 3 --worker-id 0
```

Yahoo task discovery reads the exact signal names in `data/datasets.csv`, maps
them into `data/Yahoo_S5/A{1,2,3,4}Benchmark`, and therefore excludes the A3/A4
aggregate `*_all.csv` files. Yahoo labels in `anomalies.csv` use one-based row
positions, so the runner replaces both the singular `timestamp` used by A1/A2
and the plural `timestamps` used by A3/A4 with `1..N`. Source `is_anomaly` and
`anomaly` columns are not model inputs.

Validate the Yahoo task list without loading CLIP (the cache root must already
contain one validated cache for each selected signal):

```bash
/home/pengxf/anaconda3/envs/vit/bin/python experiments/vfg_paired/runner.py \
  --config experiments/vfg_paired/config_yahoo.json \
  --output-root data/runs/vfg_paired_yahoo_20261028 \
  --cache-root data/runs/vfg_paired_yahoo_20261028/feature_cache \
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

Extract missing Yahoo visual caches with the dedicated entrypoint. It uses the
same VFG backend digest, 224/56 rendering, frozen OpenAI CLIP encoder, cache
validation, and atomic no-replace publication. Dry-run validates source files
without importing CLIP or writing artifacts:

```bash
/home/pengxf/anaconda3/envs/vit/bin/python experiments/vfg_paired/extract_visual_cache.py \
  --config experiments/vfg_paired/config_yahoo.json --dry-run \
  --worker-index 0 --num-workers 3
```

For real extraction, pass the same isolated `--cache-root` to each worker and
assign an available device. Per-worker manifests are published beside the
cache root under `cache_manifests` and never replace existing manifests.
