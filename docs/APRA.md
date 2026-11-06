# APRA production layout

The validated Detect-only APRA implementation is split by responsibility:

- `src/models/apra.py`: model, differentiable three-state alignment, evidence head, and detection loss.
- `src/training/apra.py`: production training entry point.
- `src/inference/apra.py`: single-worker inference and score generation.
- `src/inference/apra_workers.py`: multi-GPU inference launcher.
- `src/evaluation/apra.py`: OrionBench metric and event evaluation.
- `src/evaluation/apra_workers.py`: parallel evaluation launcher.
- `src/evaluation/orionbench_protocol.py`: dataset split and signal protocol.
- `src/evaluation/mindts_metrics.py`: Affiliation and VUS metric wrappers.
- `configs/apra.json`: default validated configuration.
- `tests/test_apra.py`: core model and gradient tests.
- `tools/apra/`: immutable-history validation and report generation.

The `experiments/` directory is ignored by Git and is reserved for temporary
ablations, diagnostics, and local exploratory scripts.

## Commands

Train one dataset cache:

```bash
python -m src.training.apra \
  --config configs/apra.json \
  --cache /path/to/dataset-cache.pt \
  --output /path/to/checkpoint.pt \
  --device cuda:0
```

Run inference across GPUs:

```bash
python -m src.inference.apra_workers \
  --config configs/apra.json \
  --output-root /path/to/run \
  --gpus 0 1 2 3
```

Evaluate generated scores:

```bash
python -m src.evaluation.apra_workers \
  --config configs/apra.json \
  --output-root /path/to/run \
  --workers 8
```

Run the core tests:

```bash
python -m pytest tests/test_apra.py -q
```
