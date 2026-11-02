import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.curve_aware_sampling import (
    CurveCoordinates,
    bilinear_curve_features,
    column_mean_features,
    curve_pixel_coordinates,
    gaussian_curve_features,
    paired_time_feature_scores,
    restore_column_scores,
)


class CurveCoordinateTests(unittest.TestCase):
    def test_coordinates_match_actual_matplotlib_transform(self):
        times = np.linspace(7.0, 19.0, 224) ** 1.01
        values = np.sin(np.linspace(0.0, 3.0, 224)) + np.linspace(-0.3, 0.7, 224)
        coordinates = curve_pixel_coordinates(values, times)

        figure = plt.figure(figsize=(2.24, 2.24), dpi=100)
        axes = figure.add_subplot(111)
        axes.plot(times, values, linestyle="-", linewidth=1, marker="*", markersize=2, color="black")
        axes.set_ylim((0, 1))
        axes.set_xticks([])
        axes.set_yticks([])
        figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        axes.margins(0, 0)
        figure.canvas.draw()

        sample_times = times[0] + coordinates.pixel_x / 224.0 * (times[-1] - times[0])
        sample_values = np.interp(sample_times, times, values)
        display = axes.transData.transform(np.column_stack((sample_times, sample_values)))
        expected_top_origin = 224.0 - display[:, 1]
        np.testing.assert_allclose(coordinates.pixel_x, display[:, 0], atol=1e-10)
        np.testing.assert_allclose(coordinates.pixel_y, expected_top_origin, atol=1e-10)
        self.assertEqual(tuple(int(v) for v in axes.bbox.bounds), (0, 0, 224, 224))
        plt.close(figure)

    def test_vertical_origin_boundary_and_constant_series(self):
        increasing = curve_pixel_coordinates(np.linspace(0.0, 1.0, 224))
        self.assertTrue(np.all(np.diff(increasing.pixel_y) < 0))
        self.assertGreater(increasing.pixel_y[0], increasing.pixel_y[-1])
        self.assertGreaterEqual(increasing.pixel_y.min(), 0.0)
        self.assertLessEqual(increasing.pixel_y.max(), 224.0)

        constant = curve_pixel_coordinates(np.zeros(224))
        np.testing.assert_allclose(constant.pixel_y, 224.0)
        np.testing.assert_allclose(constant.patch_y, 13.5)

        generic_constant = curve_pixel_coordinates(np.full(224, 3.0), y_limits=None)
        np.testing.assert_allclose(generic_constant.pixel_y, 112.0)
        np.testing.assert_allclose(generic_constant.patch_y, 6.5)

        # Explicit points at plot limits map to image edges, not pixel centres.
        edges = curve_pixel_coordinates(np.array([1.0, 0.0]), grid_size=2)
        self.assertTrue(np.all((edges.pixel_y >= 0.0) & (edges.pixel_y <= 224.0)))

    def test_batched_time_coordinates(self):
        values = np.stack((np.linspace(0, 1, 224), np.linspace(1, 0, 224)))
        times = np.stack((np.arange(224), np.arange(224) * 3.0))
        coordinates = curve_pixel_coordinates(values, times)
        self.assertEqual(coordinates.pixel_y.shape, (2, 14))
        np.testing.assert_allclose(coordinates.pixel_y[0], coordinates.pixel_y[1][::-1])

    def test_bad_time_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            curve_pixel_coordinates(np.arange(4), np.array([0, 2, 1, 3]))


class CurveFeatureSamplingTests(unittest.TestCase):
    @staticmethod
    def _coordinates(patch_y, windows=1):
        patch_y = np.broadcast_to(np.asarray(patch_y, dtype=np.float64), (windows, 14)).copy()
        patch_x = np.broadcast_to(np.arange(14, dtype=np.float64), (windows, 14)).copy()
        return CurveCoordinates(
            pixel_x=(patch_x + 0.5) * 16,
            pixel_y=(patch_y + 0.5) * 16,
            patch_x=patch_x,
            patch_y=patch_y,
        )

    def test_row_major_grid_and_column_mean(self):
        grid = torch.arange(196, dtype=torch.float64).reshape(1, 196, 1)
        mean = column_mean_features(grid)
        expected = torch.arange(14, dtype=torch.float64) + 14.0 * 6.5
        torch.testing.assert_close(mean[0, :, 0], expected)

    def test_bilinear_interpolates_one_hot_grid(self):
        grid = torch.zeros(1, 14, 14, 1, dtype=torch.float64)
        grid[0, 5, 3, 0] = 1.0
        # Halfway in both dimensions from the one-hot centre gives weight 1/4.
        coordinates = self._coordinates(np.full(14, 5.5))
        coordinates = CurveCoordinates(
            pixel_x=coordinates.pixel_x,
            pixel_y=coordinates.pixel_y,
            patch_x=np.full((1, 14), 3.5),
            patch_y=coordinates.patch_y,
        )
        sampled = bilinear_curve_features(grid, coordinates)
        torch.testing.assert_close(sampled[0, :, 0], torch.full((14,), 0.25, dtype=torch.float64))

    def test_bilinear_uses_top_row_for_high_curve_and_clamps_edges(self):
        grid = torch.arange(14, dtype=torch.float64).reshape(1, 14, 1, 1).expand(1, 14, 14, 1)
        top_edge = self._coordinates(np.full(14, -0.5))
        bottom_edge = self._coordinates(np.full(14, 13.5))
        torch.testing.assert_close(
            bilinear_curve_features(grid, top_edge)[0, :, 0], torch.zeros(14, dtype=torch.float64)
        )
        torch.testing.assert_close(
            bilinear_curve_features(grid, bottom_edge)[0, :, 0], torch.full((14,), 13.0, dtype=torch.float64)
        )

    def test_sampling_accepts_cached_grid_xy_tensor(self):
        grid = torch.arange(196, dtype=torch.float64).reshape(1, 196, 1)
        grid_xy = torch.stack(
            (torch.arange(14, dtype=torch.float64), torch.full((14,), 4.0)), dim=-1
        ).unsqueeze(0)
        sampled = bilinear_curve_features(grid, grid_xy)
        expected = 4 * 14 + torch.arange(14, dtype=torch.float64)
        torch.testing.assert_close(sampled[0, :, 0], expected)

    def test_gaussian_is_symmetric_normalized_and_same_column(self):
        grid = torch.zeros(1, 14, 14, 2, dtype=torch.float64)
        rows = torch.arange(14, dtype=torch.float64)
        columns = torch.arange(14, dtype=torch.float64)
        grid[0, :, :, 0] = rows[:, None]
        grid[0, :, :, 1] = columns[None, :]
        coordinates = self._coordinates(np.full(14, 6.5))
        sampled = gaussian_curve_features(grid, coordinates, sigma_patches=1.0)
        torch.testing.assert_close(sampled[0, :, 0], torch.full((14,), 6.5, dtype=torch.float64))
        torch.testing.assert_close(sampled[0, :, 1], columns)

        one_hot = torch.zeros(1, 14, 14, 1, dtype=torch.float64)
        one_hot[0, 6, :, 0] = 1.0
        one_hot[0, 7, :, 0] = 1.0
        response = gaussian_curve_features(one_hot, coordinates, sigma_patches=1.0)
        self.assertTrue(torch.all(response > 0))
        self.assertTrue(torch.all(response < 1))

    def test_invalid_gaussian_sigma_is_rejected(self):
        tokens = torch.zeros(1, 196, 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            gaussian_curve_features(tokens, self._coordinates(np.zeros(14)), sigma_patches=0)

    def test_gaussian_radius_excludes_distant_rows(self):
        tokens = torch.zeros(1, 14, 14, 1)
        tokens[:, 0, :, 0] = 1.0
        sampled = gaussian_curve_features(
            tokens,
            self._coordinates(np.full(14, 7.0)),
            sigma_patches=10.0,
            radius_patches=3.0,
        )
        self.assertTrue(torch.equal(sampled, torch.zeros_like(sampled)))


class CurveMatchingAndRestorationTests(unittest.TestCase):
    def test_paired_scores_use_only_supplied_candidate_pool(self):
        features = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
                [[-1.0, 0.0], [0.0, -1.0]],
            ],
            dtype=torch.float64,
        )
        ids = torch.tensor([[1, -1], [0, -1], [0, 1]])
        valid = ids >= 0
        scores = paired_time_feature_scores(features, ids, valid)
        torch.testing.assert_close(scores[0], torch.tensor([0.0, 0.5], dtype=torch.float64))
        torch.testing.assert_close(scores[1], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(scores[2], torch.full((2,), 0.5, dtype=torch.float64))

    def test_zero_features_are_finite(self):
        features = torch.zeros(2, 14, 4)
        ids = torch.tensor([[1], [0]])
        scores = paired_time_feature_scores(features, ids)
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(scores, torch.full_like(scores, 0.5))

    def test_restore_constant_scores_and_tail(self):
        columns = torch.full((3, 14), 2.5, dtype=torch.float64)
        covered = 224 + 2 * 56
        restored = restore_column_scores(columns, covered + 17)
        self.assertEqual(restored.shape, (covered + 17,))
        torch.testing.assert_close(restored, torch.full_like(restored, 2.5))

    def test_restore_rejects_impossible_full_length(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            restore_column_scores(torch.zeros(2, 14), 250)


if __name__ == "__main__":
    unittest.main()
