import numpy as np
import torch

from src.models.fixed_reference_numeric_evidence import (
    align_point_score_to_t3, margin_confidence, rank_agreement,
    representation_reliability, statistical_reliability,
)


def test_point_score_calibration_matches_positive_median():
    numeric = np.array([0.0, 1.0, 2.0, 3.0])
    visual = np.array([0.0, 2.0, 4.0, 6.0])
    aligned, scale = align_point_score_to_t3(numeric, visual)
    assert scale == 2.0
    assert np.array_equal(aligned, visual)


def test_reliability_is_bounded_and_rewards_rank_agreement():
    visual = np.array([0.1, 0.2, 0.4, 0.7, 1.0])
    agreeing = np.array([0.2, 0.3, 0.5, 0.8, 1.2])
    reversed_cost = agreeing[::-1]
    good = statistical_reliability(visual, agreeing)
    bad = statistical_reliability(visual, reversed_cost)
    assert 0.0 <= bad <= good <= 1.0
    assert rank_agreement(visual, agreeing) == 1.0
    assert np.isclose(rank_agreement(visual, reversed_cost), 0.0)
    assert 0.0 <= margin_confidence(visual) <= 1.0


def test_representation_reliability_is_bounded():
    visual = np.array([0.1, 0.2, 0.4, 0.7, 1.0])
    numeric = np.array([0.2, 0.3, 0.5, 0.8, 1.2])
    assert 0.0 <= representation_reliability(visual, numeric, numeric) <= 1.0
