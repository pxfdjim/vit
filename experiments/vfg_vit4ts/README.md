# Visual-Frequency Guided ViT4TS experiments

This directory contains the isolated single-variable orchestration layer. Model
code remains in `src/models/vfg_vit4ts.py`; the runner imports it only when a
real inference run begins.

## Safety and resumption

- The seven configured datasets contain 125 complete signals.
- `--dry-run` validates all inputs without importing CLIP, touching CUDA, or
  writing outputs.
- Point scores and metric JSON files are atomically published and never
  replaced. Existing artifacts are validated before reuse.
- Deterministic slicing (`tasks[worker_id::num_workers]`) gives workers disjoint
  signal sets.
- `run_parallel.sh` refuses to start with fewer than three devices and requires
  the caller to provide devices after checking GPU occupancy.

CPU validation:

```bash
/home/pengxf/anaconda3/envs/vit/bin/python experiments/vfg_vit4ts/runner.py \
  --dry-run --suite all --device cpu
```

GPU execution after selecting free devices:

```bash
DEVICES=cuda:1,cuda:2,cuda:3 \
  experiments/vfg_vit4ts/run_parallel.sh
```

For a three-signal interface check on selected free GPUs, set
`LIMIT_PER_WORKER=1 SUMMARIZE=0`. Each launch receives a fresh log attempt
directory, while model artifacts and earlier logs remain untouched.

The strict aggregation order is: for each dataset/model/alpha, equally average
the contextual F1 of all signals; choose the alpha with the highest mean; then
macro average the seven dataset F1-max values. Incomplete groups are rejected by
default.

Local STFT frames are centered on the 14 visual columns. The 224-point window is
reflect-padded by 24 points on both sides so the 64-point Hann frames at hop 16
have a deterministic, symmetric boundary convention.
