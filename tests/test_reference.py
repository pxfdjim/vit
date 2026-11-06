import numpy as np
import torch

from src.core.visual_features import expected_window_count, input_digest
from src.core.reference import (
    column_map,
    match_paths,
    point_scores_from_maps,
    select_reference_windows,
    visual_costs,
)


def test_t3_reference_contract_is_deterministic_and_non_overlapping():
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randn(12, 196, 24, generator=generator)
    pool = select_reference_windows(tokens)
    query = torch.arange(12)[:, None]
    assert torch.all((pool.window_ids - query).abs()[pool.valid] >= 4)

    costs = visual_costs(tokens.reshape(12, 14, 14, 24), pool)
    first = match_paths(costs, pool)
    second = match_paths(costs, pool)
    assert torch.equal(first.reference_ids, second.reference_ids)
    assert torch.equal(first.reference_columns, second.reference_columns)
    assert torch.equal(first.path_masks, second.path_masks)
    assert torch.all(first.reference_columns[:, 1:] >= first.reference_columns[:, :-1])


def test_score_restoration_returns_complete_finite_series():
    local = torch.linspace(-1, 1, 12 * 14, dtype=torch.float64).reshape(12, 14)
    restored = point_scores_from_maps(
        column_map(local), full_length=901, device=torch.device("cpu")
    )
    assert restored.shape == (901,)
    assert np.isfinite(restored).all()


def test_cache_identity_contract():
    values = np.arange(901, dtype=np.float64)
    assert expected_window_count(len(values)) == 13
    assert input_digest(values) == input_digest(values.copy())
    changed = values.copy()
    changed[-1] += 1
    assert input_digest(values) != input_digest(changed)
