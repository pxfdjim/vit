"""Differential check against the old 224/56 quantile and Top score vectors."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from worker import ROOT, OLD, infer, e

assert os.environ['CLIP_PRETRAINED'] == 'openai'
ds = 'realAdExchange'
signal = 'exchange-2_cpc_results'
frame = pd.read_csv(ROOT / 'data' / ds / (signal + '.csv')).sort_values('timestamp')
model = e.vit.ViT4TS(window_size=224, window_step_ratio=4.0,
                     image_size=(224, 224), verbose=False)
official, top, q25, timestamps, vectors = infer(model, frame, 224)
with np.load(OLD / 'scores' / ds / signal / 'paper_window_224.npz') as old:
    top_diff = float(np.max(np.abs(top - old['corrected'])))
    official_diff = float(np.max(np.abs(official - old['official'])))
with np.load(OLD / 'quantile_diagnostic/scores' / ds / (signal + '.npz')) as old:
    q_diff = float(np.max(np.abs(q25 - old['quantile'])))
print({'top_max_abs_diff': top_diff, 'official_max_abs_diff': official_diff,
       'q25_max_abs_diff': q_diff}, flush=True)
if max(top_diff, official_diff, q_diff) > 1e-6:
    raise RuntimeError('Differential score check failed')
