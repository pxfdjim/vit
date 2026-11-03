from pathlib import Path

import numpy as np

from experiments.bimodal_reference.mindts_metrics import affiliation_scores


MINDTS_ROOT = Path(
    "/home/pengxf/jim/VLM4TS/.worktrees/unified-aff-vus-1e38ecf/.references/MindTS"
)


def test_mindts_raw_empty_prediction_is_nan_but_policy_is_zero():
    labels = np.array([0, 0, 1, 1, 0, 0], np.uint8)
    predictions = np.zeros_like(labels)
    precision, recall, raw_f1 = affiliation_scores(labels, predictions, MINDTS_ROOT)
    assert np.isnan(precision)
    assert recall == 0.0
    assert np.isnan(raw_f1)
    precision, recall, corrected_f1 = affiliation_scores(
        labels, predictions, MINDTS_ROOT, empty_prediction_f1_zero=True,
    )
    assert np.isnan(precision)
    assert recall == 0.0
    assert corrected_f1 == 0.0


def test_nonempty_prediction_is_unchanged_by_policy():
    labels = np.array([0, 0, 1, 1, 0, 0], np.uint8)
    predictions = np.array([0, 1, 0, 0, 0, 0], np.uint8)
    raw = affiliation_scores(labels, predictions, MINDTS_ROOT)
    corrected = affiliation_scores(
        labels, predictions, MINDTS_ROOT, empty_prediction_f1_zero=True,
    )
    np.testing.assert_allclose(raw, corrected, rtol=0, atol=0)


def test_single_class_ground_truth_remains_undefined():
    labels = np.zeros(6, np.uint8)
    predictions = np.zeros_like(labels)
    result = affiliation_scores(
        labels, predictions, MINDTS_ROOT, empty_prediction_f1_zero=True,
    )
    assert all(np.isnan(value) for value in result)
