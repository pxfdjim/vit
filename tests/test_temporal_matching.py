from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.models.temporal_matching import (
    GRID_SIZE,
    column_cost_matrix,
    constrained_dtw_match,
    independent_temporal_match,
    shift_consistent_match,
    temporal_match_all,
)


def _manual_column_cost(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    query = query.reshape(GRID_SIZE, GRID_SIZE, -1)
    reference = reference.reshape(GRID_SIZE, GRID_SIZE, -1)
    output = torch.empty(GRID_SIZE, GRID_SIZE, dtype=query.dtype)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            vertical = []
            for a in range(GRID_SIZE):
                q = F.normalize(query[a, i], dim=0, eps=1e-12)
                distances = []
                for b in range(GRID_SIZE):
                    r = F.normalize(reference[b, j], dim=0, eps=1e-12)
                    distances.append(max(0.0, 1.0 - float(q @ r)))
                vertical.append(min(distances))
            output[i, j] = sum(vertical) / GRID_SIZE
    return output


def _column_basis(shift: int = 0) -> torch.Tensor:
    """Every row in column i is basis vector e_(i+shift mod 14)."""

    basis = torch.eye(GRID_SIZE, dtype=torch.float64)
    columns = torch.stack(
        [basis[(column + shift) % GRID_SIZE] for column in range(GRID_SIZE)], dim=0
    )
    return columns.unsqueeze(0).expand(GRID_SIZE, -1, -1).contiguous()


class ColumnCostTests(unittest.TestCase):
    def test_matches_hand_computation_and_batch(self) -> None:
        generator = torch.Generator().manual_seed(123)
        query = torch.randn(GRID_SIZE, GRID_SIZE, 4, generator=generator, dtype=torch.float64)
        references = torch.randn(
            2, GRID_SIZE, GRID_SIZE, 4, generator=generator, dtype=torch.float64
        )
        actual = column_cost_matrix(query, references)
        expected = torch.stack(
            [_manual_column_cost(query, reference) for reference in references]
        )
        self.assertEqual(tuple(actual.shape), (2, GRID_SIZE, GRID_SIZE))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

        flattened = column_cost_matrix(query.reshape(196, 4), references[0].reshape(196, 4))
        torch.testing.assert_close(flattened, expected[0], atol=1e-12, rtol=1e-12)

    def test_identical_features_and_zero_vectors_are_stable(self) -> None:
        features = _column_basis()
        identical = column_cost_matrix(features, features)
        torch.testing.assert_close(
            identical.diagonal(), torch.zeros(GRID_SIZE, dtype=features.dtype)
        )
        off_diagonal = identical[~torch.eye(GRID_SIZE, dtype=torch.bool)]
        torch.testing.assert_close(off_diagonal, torch.ones_like(off_diagonal))

        zeros = torch.zeros(GRID_SIZE, GRID_SIZE, 8)
        zero_cost = column_cost_matrix(zeros, zeros)
        self.assertTrue(torch.isfinite(zero_cost).all())
        torch.testing.assert_close(zero_cost, torch.ones_like(zero_cost))

    def test_rejects_non_finite_features(self) -> None:
        query = _column_basis()
        query[0, 0, 0] = math.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            column_cost_matrix(query, _column_basis())


class TemporalRuleTests(unittest.TestCase):
    def test_t1_independently_changes_reference_window_and_column(self) -> None:
        costs = torch.full((2, GRID_SIZE, GRID_SIZE), 2.0)
        for i in range(GRID_SIZE):
            reference = i % 2
            column = (i + 3) % GRID_SIZE
            costs[reference, i, column] = i / 100.0
        result = independent_temporal_match(costs, torch.tensor([10, 20]))
        torch.testing.assert_close(
            result.scores, torch.arange(GRID_SIZE, dtype=costs.dtype) / 100.0
        )
        self.assertEqual(result.reference_ids.tolist(), [10, 20] * 7)
        self.assertEqual(
            result.reference_columns.tolist(),
            [(i + 3) % GRID_SIZE for i in range(GRID_SIZE)],
        )

    def test_t2_uses_one_window_and_one_shift(self) -> None:
        costs = torch.ones(2, GRID_SIZE, GRID_SIZE)
        # Reference 17 is best only under one common +2 shift.
        positions = torch.arange(GRID_SIZE - 2)
        costs[1, positions, positions + 2] = torch.linspace(0.01, 0.12, GRID_SIZE - 2)
        # A tempting per-column match in reference 4 must not let T2 mix refs.
        costs[0, torch.arange(GRID_SIZE), torch.arange(GRID_SIZE)] = 0.3
        result = shift_consistent_match(
            costs, torch.tensor([4, 17]), shift_radius=2
        )
        self.assertEqual(int(result.reference_id), 17)
        self.assertEqual(int(result.shift), 2)
        self.assertEqual(result.valid.tolist(), [True] * 12 + [False, False])
        torch.testing.assert_close(result.scores[:12], costs[1, positions, positions + 2])
        torch.testing.assert_close(
            result.scores[12:], result.objective_cost.expand(2)
        )

    def test_t2_tie_break_is_deterministic(self) -> None:
        costs = torch.zeros(2, GRID_SIZE, GRID_SIZE)
        result = shift_consistent_match(costs, torch.tensor([7, 8]), shift_radius=2)
        self.assertEqual(int(result.reference_id), 7)
        self.assertEqual(int(result.shift), 0)

    def test_t3_diagonal_path_has_manual_normalized_cost(self) -> None:
        costs = torch.ones(2, GRID_SIZE, GRID_SIZE)
        diagonal = torch.linspace(0.0, 0.13, GRID_SIZE)
        costs[0].diagonal().copy_(diagonal)
        costs[1].diagonal().fill_(0.5)
        result = constrained_dtw_match(
            costs, torch.tensor([30, 31]), band=0, non_diagonal_penalty=0.2
        )
        self.assertEqual(int(result.reference_id), 30)
        self.assertEqual(int(result.path_length), GRID_SIZE)
        torch.testing.assert_close(result.scores, diagonal)
        torch.testing.assert_close(result.offsets, torch.zeros_like(result.offsets))
        torch.testing.assert_close(result.normalized_path_cost, diagonal.mean())

    def test_t3_recovers_local_warp_but_respects_monotonic_order(self) -> None:
        cost = torch.ones(GRID_SIZE, GRID_SIZE)
        desired_path = [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
            (6, 5), (7, 6), (8, 7), (9, 8), (10, 9), (11, 10),
            (12, 11), (12, 12), (13, 13),
        ]
        for i, j in desired_path:
            cost[i, j] = 0.0
        result = constrained_dtw_match(
            cost, torch.tensor([91]), band=2, non_diagonal_penalty=0.05
        )
        self.assertEqual(int(result.reference_id), 91)
        self.assertGreater(int(result.path_length), GRID_SIZE)
        # Two non-diagonal moves each add 0.05 to a 15-cell zero-local-cost
        # path.  The normalized diagnostic includes that penalty while the
        # 14 mapped local anomaly scores deliberately do not.
        self.assertAlmostEqual(float(result.normalized_path_cost), 0.1 / 15.0, places=7)
        torch.testing.assert_close(result.scores, torch.zeros_like(result.scores))
        self.assertTrue(torch.any(result.offsets != 0))
        self.assertTrue(torch.all(result.offsets.abs() <= 2))
        self.assertTrue(torch.all(result.reference_columns[1:] >= result.reference_columns[:-1]))

        # A fully reversed order cannot be followed by a monotone path in a
        # narrow band even though each query column has one zero-cost match.
        reversed_cost = torch.ones(GRID_SIZE, GRID_SIZE)
        reversed_cost[torch.arange(GRID_SIZE), torch.arange(GRID_SIZE - 1, -1, -1)] = 0
        reversed_result = constrained_dtw_match(
            reversed_cost, torch.tensor([92]), band=2, non_diagonal_penalty=0.05
        )
        self.assertGreater(float(reversed_result.normalized_path_cost), 0.5)

    def test_non_diagonal_penalty_prevents_excess_warp(self) -> None:
        cost = torch.full((GRID_SIZE, GRID_SIZE), 1.0)
        cost.diagonal().fill_(0.1)
        warp = [
            (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5),
            (7, 6), (8, 7), (9, 8), (10, 9), (11, 10), (12, 11),
            (13, 12), (13, 13),
        ]
        for i, j in warp:
            cost[i, j] = 0.0
        unpenalized = constrained_dtw_match(
            cost, torch.tensor([1]), band=2, non_diagonal_penalty=0.0
        )
        penalized = constrained_dtw_match(
            cost, torch.tensor([1]), band=2, non_diagonal_penalty=2.0
        )
        self.assertGreater(int(unpenalized.path_length), GRID_SIZE)
        self.assertEqual(int(penalized.path_length), GRID_SIZE)
        torch.testing.assert_close(penalized.offsets, torch.zeros_like(penalized.offsets))


class BatchedTemporalTests(unittest.TestCase):
    def test_all_variants_share_exact_candidate_pool(self) -> None:
        tokens = torch.stack(
            [_column_basis(0), _column_basis(0), _column_basis(1)], dim=0
        ).float()
        pool = SimpleNamespace(
            window_ids=torch.tensor([[1, 2], [0, 2], [0, 1]]),
            valid=torch.ones(3, 2, dtype=torch.bool),
        )
        result = temporal_match_all(
            tokens,
            pool,
            shift_radius=2,
            dtw_band=2,
            dtw_penalty_scale=0.1,
            output_factor=1.0,
        )
        for scores in (result.t1_scores, result.t2_scores, result.t3_scores):
            self.assertEqual(tuple(scores.shape), (3, GRID_SIZE))
            self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(result.t1_reference_ids[0].unique().tolist(), [1])
        self.assertEqual(int(result.t2_reference_ids[0]), 1)
        self.assertEqual(int(result.t3_reference_ids[0]), 1)
        self.assertEqual(result.t1_reference_ids.dtype, torch.long)
        self.assertEqual(result.t2_shifts.dtype, torch.long)
        self.assertEqual(result.t2_valid.dtype, torch.bool)
        self.assertEqual(result.t3_path_lengths.dtype, torch.long)
        self.assertEqual(tuple(result.t3_offsets.shape), (3, GRID_SIZE))
        self.assertGreaterEqual(float(result.dtw_penalty), 0.0)

    def test_zero_features_stay_finite_and_factor_is_applied(self) -> None:
        tokens = torch.zeros(2, 196, 3)
        pool = SimpleNamespace(
            window_ids=torch.tensor([[1], [0]]),
            valid=torch.ones(2, 1, dtype=torch.bool),
        )
        result = temporal_match_all(tokens, pool, output_factor=0.5)
        for scores in (result.t1_scores, result.t2_scores, result.t3_scores):
            torch.testing.assert_close(scores, torch.full_like(scores, 0.5))
        torch.testing.assert_close(result.t2_objective_costs, torch.ones(2))
        torch.testing.assert_close(result.t3_path_costs, torch.ones(2))
        torch.testing.assert_close(result.dtw_penalty, torch.tensor(0.1))
        self.assertTrue(torch.isfinite(result.t3_offsets).all())

    def test_rejects_inconsistent_candidate_mask(self) -> None:
        tokens = torch.zeros(2, 196, 3)
        pool = SimpleNamespace(
            window_ids=torch.tensor([[1], [-1]]),
            valid=torch.ones(2, 1, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "mask"):
            temporal_match_all(tokens, pool)


if __name__ == "__main__":
    unittest.main()

