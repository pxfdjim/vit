# Curve-aware ViT4TS experiment

This experiment evaluates curve-aware sampling at only the base 14×14 CLIP
Patch scale. The existing large and middle visual scales, Harmonic
stabilization, 224/56 windowing, candidate-window selection, score recovery,
and contextual F1-max protocol remain shared by every variant.

## Variants

- `A_stable_final`: Stable ViT4TS median-reference baseline, final layer.
- `B_paired_2d_{final,mid}`: paired real-window 2D Patch matching.
- `C_column_mean_{final,mid}`: one feature per time column by vertical mean.
- `D_curve_bilinear_{final,mid}`: bilinear feature sample at the rendered
  curve position in every time column.
- `E_curve_gaussian_{final,mid}`: Gaussian-weighted vertical neighborhood
  around the rendered curve position.

A is intentionally unique. Repeating A with a middle-layer base feature would
change the published baseline and would not add a controlled layer comparison.
B--E use one shared eligible reference-window pool for every signal.

## Backend contract

`src.models.curve_aware_vit4ts` must export:

```python
create_backend(config, device, cache_dir)
backend.predict_variants(data, variants, context)
```

The prediction mapping has one entry per requested variant. Each entry contains
`scores`, optional `timestamps`, `timing`, and `metadata`. Paired variants must
report `candidate_pool_min_windows`, `candidate_pool_max_windows`, and a common
`candidate_pool_digest`. The validator uses that digest to verify that sampling
methods and CLIP layers did not silently change the reference candidates.

## Safe execution

Use a new output directory. The runner publishes scores, metrics, config, and
worker manifests atomically and never replaces an existing file. It shards the
492 indexed signals deterministically, so three workers can run concurrently:

```bash
python experiments/curve_aware/runner.py --dry-run \
  --config experiments/curve_aware/config.json --final-cache-root /READONLY/FINAL1 \
  --final-cache-root /READONLY/FINAL2 \
  --cache-root /NEW/CACHE --output-root /NEW/OUTPUT

python experiments/curve_aware/runner.py --worker-id 0 --num-workers 3 --device cuda:1 \
  --final-cache-root /READONLY/FINAL1 --final-cache-root /READONLY/FINAL2 \
  --cache-root /NEW/CACHE --output-root /NEW/OUTPUT &
python experiments/curve_aware/runner.py --worker-id 1 --num-workers 3 --device cuda:2 \
  --final-cache-root /READONLY/FINAL1 --final-cache-root /READONLY/FINAL2 \
  --cache-root /NEW/CACHE --output-root /NEW/OUTPUT &
python experiments/curve_aware/runner.py --worker-id 2 --num-workers 3 --device cuda:7 \
  --final-cache-root /READONLY/FINAL1 --final-cache-root /READONLY/FINAL2 \
  --cache-root /NEW/CACHE --output-root /NEW/OUTPUT &
wait
```

GPU identifiers are examples only; recheck processes and free memory before
launching. Point `--cache-root` at the dedicated curve-aware cache prepared by
the cache extractor. Compatible existing CLIP tensors are reused by the model
backend, while missing layer or curve-coordinate fields are added only in the
new cache directory.

## Summarize and validate

```bash
python experiments/curve_aware/summarize.py --output-root /NEW/OUTPUT
python experiments/curve_aware/validate_results.py --output-root /NEW/OUTPUT \
  --previous-root /PREVIOUS/ORIGINAL7/RUN \
  --previous-root /PREVIOUS/YAHOO4/RUN
```

The summary saves all α rows, selected dataset F1-max, four macro scopes,
contextual TP/FP/FN, event recovery/loss, per-variant runtime, candidate-pool
audit, plots, and `experiment_report.md`. Yahoo signals are taken strictly from
`datasets.csv`; their timestamps are one-based row positions to match
`anomalies.csv`.
