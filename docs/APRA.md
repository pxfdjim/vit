# APRA production layout

The validated Detect-only APRA implementation is split by responsibility:

- `src/models/apra.py`: model, differentiable three-state alignment, evidence head, and detection loss.
- `src/apra/cache.py`: immutable final-layer and Block-6 visual-cache contract.
- `src/apra/reference.py`: T3 candidate retrieval, constrained DTW, normalization, and point-score restoration.
- `src/training/apra.py`: production training entry point.
- `src/inference/apra.py`: single-worker inference and score generation.
- `src/inference/apra_workers.py`: multi-GPU inference launcher.
- `src/evaluation/apra.py`: OrionBench metric and event evaluation.
- `src/evaluation/apra_workers.py`: parallel evaluation launcher.
- `src/evaluation/orionbench_protocol.py`: dataset split and signal protocol.
- `src/evaluation/mindts_metrics.py`: Affiliation and VUS metric wrappers.
- `configs/apra.json`: default validated configuration.
- `tests/test_apra.py`: core model and gradient tests.
- `tests/test_apra_reference.py`: T3 reference and score-restoration contract tests.

The `experiments/` directory is ignored by Git and is reserved for temporary
ablations, diagnostics, and local exploratory scripts.

The production package reads validated B3 point scores, T3 visual caches,
normalization statistics, and APRA checkpoints from `configs/apra.json`;
training receives its synthetic cache through `--cache`. It does not import
experimental backends. Historical source removed from the dependency graph is
kept locally under `experiments/legacy_source_20261106/` and remains available
in Git history.

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
python -m pytest tests -q
```

## Validated invariants

- APRA parameters: `2,059,973`
- Checkpoint state tensors: `73`
- Historical E6 Checkpoint strict-load error: `0.0`
- Model output, detection loss, and gradient error: `0.0`
- Historical point-score error over 4,045,296 values: `0.0`
