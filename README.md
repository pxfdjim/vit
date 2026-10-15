<div align="center">

<img src="asset/PSU.png" height="80px" alt="Penn State">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="asset/MIT.svg" height="80px" alt="MIT">

---

# [AAAI 2026 Oral] Harnessing Vision-Language Models for Time Series Anomaly Detection

[Paper](https://arxiv.org/abs/2506.06836) | [Quick Start](#quick-start) | [Reproduce Results](#reproduce-paper-results) | [Citation](#citation)

</div>

<div align="center">
  <img src="Workflow.gif" alt="VLM4TS workflow" width="900">
</div>

## Overview

VLM4TS is a two-stage, zero-shot anomaly detection pipeline for time series:
1. **ViT4TS (screening):** converts a time series into short sliding-window plots and scores anomalies via pre-trained vision encoders.
2. **VLM4TS (verification):** prompts a vision-language model with the long time horizons and ViT4TS proposals to refine anomalies.

## Quick Start

### Environment Setup

```bash
conda activate vit
python -m pip install -r requirements.txt
```

On the current server, `vit` is at `/home/pengxf/anaconda3/envs/vit`.
The default verification backend is the deployed Qwen vision model at
`http://192.168.110.81:11434/v1`, model `qwen3.8:27b-q4_K_M`.
The endpoint uses an OpenAI-compatible protocol but does not need an external
OpenAI API key. Override it when needed:

```bash
export VLM_BASE_URL=http://192.168.110.81:11434/v1
export VLM_MODEL=qwen3.8:27b-q4_K_M
```

The client bypasses the server's HTTP proxy for this LAN endpoint.
The first-stage OpenAI ViT-B/16 checkpoint is cached at
`data/model_cache/ViT-B-16.pt` on this server and its official SHA-256 is
verified before use; this avoids a second model download during inference.

### Run on Your Own Time Series

Input format (Orion-compatible):
- A CSV or `pandas.DataFrame` with two columns: `timestamp` and `value`.
- `timestamp` can be numeric or datetime; rows are sorted by `timestamp` internally.

Output format:
- A `pandas.DataFrame` with columns `start`, `end`, `severity`, where `start` and `end` are anomaly interval timestamps.

```python
import pandas as pd
from src.models.vit4ts import ViT4TS
from src.models.vlm4ts import VLM4TS

df = pd.read_csv("your_series.csv")  # columns: timestamp, value
df = df[["timestamp", "value"]]

# Stage 1: ViT4TS screening
vit4ts = ViT4TS()
vit_intervals = vit4ts.detect(df)
print(vit_intervals.head())  # columns: start, end, severity

# Stage 2: VLM4TS verification via the deployed Qwen endpoint
vlm4ts = VLM4TS()
vlm_intervals = vlm4ts.detect(df)
print(vlm_intervals.head())
```

Notes:
- `ViT4TS.predict_scores(df)` returns `(aligned_scores, timestamps)` if you want raw anomaly scores.
- By default, `ViT4TS` uses `alpha=0.01` and `window_size=224`, with `alpha` being the uppper quantile for thresholding and `window_size` being the rolling window size for vision screening. One should tune these hyperparameters for the best performance.

### Reproduce Paper Results

```bash
PYTHON_BIN=/home/pengxf/anaconda3/envs/vit/bin/python N_WORKERS=1 bash run_experiments.sh
```

This script checks all eleven benchmark datasets (492 series), runs VLM4TS
across multiple alphas, and aggregates results. Seven Orion datasets (125 series)
are downloaded and size-checked from the public bucket. Yahoo S5 A1-A4
(367 series) are read from the locally supplied `data/Yahoo_S5/A*Benchmark/`
folders; their labels are checked against `data/anomalies.csv`. The Yahoo
archive is not downloaded by this script because its Orion raw-series URLs
return HTTP 404. Yahoo's raw timestamps are converted to one-based row
positions during inference to align with the Orion ground truth. The two
`*Benchmark_all.csv` files are excluded from the series list.
The worker count defaults to one to avoid competing model loads and
simultaneous Qwen requests. Results with the local Qwen model are a
model-substitution experiment, rather than the paper's GPT-4o result.

## Repository Structure

```
├── data
│   └── raw                # Raw benchmark data
├── results                # Generated detection results and metrics
├── src
│   ├── evaluation         # Scoring, visualization, and utilities
│   ├── models             # ViT4TS and VLM4TS code
│   └── preprocessing      # Data conversion and image rendering
├── run_experiments.sh     # Full benchmark sweep
└── README.md              # You are here
```

## Citation

If this work is beneficial, please kindly cite:

```bibtex
@article{he2025harnessing,
  title={Harnessing Vision-Language Models for Time Series Anomaly Detection},
  author={He, Zelin and Alnegheimish, Sarah and Reimherr, Matthew},
  journal={arXiv preprint arXiv:2506.06836},
  year={2025}
}
```

If you find this work useful, please also consider starring the repo.

## Updates

Follow-up updates are coming soon.

## License

This project is released under the MIT License.
# vit
