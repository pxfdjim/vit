"""Threshold and interval utilities used by the APRA evaluation protocol."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_detection_intervals(
    score_vector,
    alpha,
    method="mean",
    smoothing=True,
    sliding=False,
    anomaly_padding=0,
):
    """Convert point scores into contiguous anomaly intervals."""

    scores = np.array(score_vector, dtype=float)
    length = len(scores)
    if smoothing:
        span = max(1, int(length * 0.01))
        scores = pd.Series(scores).ewm(span=span).mean().values
    z = norm.ppf(1 - alpha)
    anomaly_flags = np.zeros(length, dtype=bool)
    if not sliding:
        if method == "mean":
            central, spread = scores.mean(), scores.std()
        elif method == "median":
            central = np.median(scores)
            spread = np.median(np.abs(scores - central))
        else:
            raise ValueError("method must be 'mean' or 'median'")
        threshold = central + z * spread
        anomaly_flags = scores > threshold
    else:
        threshold = 0
        window = max(1, length // 3)
        step = max(1, length // 10)
        for start in range(0, length, step):
            end = min(start + window, length)
            segment = scores[start:end]
            if method == "mean":
                central, spread = segment.mean(), segment.std()
            else:
                central = np.median(segment)
                spread = np.median(np.abs(segment - central))
            anomaly_flags[start:end] |= segment > central + z * spread
    intervals = []
    active = False
    for index, flag in enumerate(anomaly_flags):
        if flag and not active:
            active, start = True, index
        elif not flag and active:
            active = False
            intervals.append((start, index - 1))
    if active:
        intervals.append((start, length - 1))
    if anomaly_padding > 0 and intervals:
        padded = [
            (max(0, start - anomaly_padding), min(length - 1, end + anomaly_padding))
            for start, end in intervals
        ]
        merged = []
        current_start, current_end = padded[0]
        for next_start, next_end in padded[1:]:
            if next_start <= current_end + 1:
                current_end = max(current_end, next_end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged.append((current_start, current_end))
        intervals = merged
    return intervals, threshold, scores
