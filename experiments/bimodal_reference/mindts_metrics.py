"""Audited MindTS Affiliation-F and VUS wrappers used by the AMR rerun."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


MINDTS_COMMIT = "7299cbb5fcf5d3c59ffc6d3552e57048d1e0ab1f"


def _events(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels) > 0
    delta = np.diff(np.r_[False, labels, False].astype(np.int8))
    starts = np.flatnonzero(delta == 1)
    stops = np.flatnonzero(delta == -1) - 1
    return list(zip(starts.tolist(), stops.tolist()))


def _event_lengths(labels: np.ndarray) -> np.ndarray:
    delta = np.diff(np.r_[0, np.asarray(labels, np.int8), 0])
    return np.flatnonzero(delta == -1) - np.flatnonzero(delta == 1)


def _extended_labels(labels: np.ndarray, window: int) -> np.ndarray:
    out = np.asarray(labels, dtype=float).copy()
    n = len(out)
    if window <= 1:
        return out
    for start, end in _events(labels):
        right = np.arange(end, min(end + window // 2, n))
        if len(right):
            out[right] += np.sqrt(1 - (right - end) / window)
        left = np.arange(max(start - window // 2, 0), start)
        if len(left):
            out[left] += np.sqrt(1 - (start - left) / window)
    return np.minimum(1.0, out)


@dataclass(frozen=True)
class VUSResult:
    vus_roc: float
    vus_pr: float
    median_event_length: int
    max_window: int


def fast_vus(labels: np.ndarray, scores: np.ndarray) -> VUSResult:
    """Algebraically equivalent to MindTS RangeAUC_volume's 250 thresholds."""
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be aligned 1-D arrays")
    if labels.min() == labels.max():
        raise ValueError("VUS is undefined for single-class ground truth")
    median_length = int(np.median(_event_lengths(labels)))
    max_window = 2 * median_length
    order = np.argsort(-scores, kind="stable")
    thresholds = scores[order][np.linspace(0, len(scores) - 1, 250).astype(int)]
    ascending = np.sort(scores)
    predicted_counts = len(scores) - np.searchsorted(ascending, thresholds, side="left")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    original_positive = float(labels.sum())
    auc_values, ap_values = [], []
    for window in range(max_window + 1):
        extended = _extended_labels(labels, window)
        tp = np.cumsum(extended[order])[predicted_counts - 1]
        p_new = (original_positive + extended.sum()) / 2.0
        recall = np.minimum(tp / p_new, 1.0)
        first = np.array([ranks[a:b + 1].min() for a, b in _events(extended)])
        first.sort()
        existence = np.searchsorted(first, predicted_counts - 1, side="right") / len(first)
        tpr_mid = recall * existence
        fpr_mid = (predicted_counts - tp) / (len(labels) - p_new)
        precision_mid = tp / predicted_counts
        tpr = np.r_[0.0, tpr_mid, 1.0]
        fpr = np.r_[0.0, fpr_mid, 1.0]
        precision = np.r_[1.0, precision_mid]
        auc_values.append(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
        ap_values.append(np.sum(np.diff(tpr[:-1]) * (precision[1:] + precision[:-1]) / 2.0))
    return VUSResult(float(np.mean(auc_values)), float(np.mean(ap_values)), median_length, max_window)


def affiliation_scores(
    labels: np.ndarray,
    predictions: np.ndarray,
    mindts_root: Path,
    *,
    empty_prediction_f1_zero: bool = False,
):
    """Return MindTS Affiliation precision, recall and F1.

    MindTS returns ``precision=NaN`` and ``recall=0`` when ground truth has
    events but the prediction event list is empty.  Its direct harmonic mean
    is consequently NaN.  ``empty_prediction_f1_zero`` provides an explicit
    evaluation policy for this one boundary case: precision remains undefined,
    recall remains zero, and F1 is recorded as zero so a complete miss is not
    removed from a macro average.  The default preserves historical outputs.
    """
    labels, predictions = np.asarray(labels, np.uint8), np.asarray(predictions, np.uint8)
    if labels.min() == labels.max():
        return np.nan, np.nan, np.nan
    root = str(mindts_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ts_benchmark.evaluation.metrics.affiliation.generics import convert_vector_to_events
    from ts_benchmark.evaluation.metrics.affiliation.metrics import pr_from_events
    result = pr_from_events(convert_vector_to_events(predictions), convert_vector_to_events(labels), (0, len(labels)))
    precision, recall = float(result["precision"]), float(result["recall"])
    if empty_prediction_f1_zero and not predictions.any() and labels.any():
        if not np.isnan(precision) or recall != 0.0:
            raise RuntimeError("unexpected MindTS empty-prediction affiliation result")
        return precision, recall, 0.0
    denom = precision + recall
    return precision, recall, 2 * precision * recall / denom if denom > 0 else np.nan
