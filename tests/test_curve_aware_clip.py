from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from experiments.curve_aware.extract_visual_cache import (
    CacheTask,
    discover_tasks,
    shard_tasks,
)
from src.models.curve_aware_clip import (
    FrozenCLIPBlock6Encoder,
    build_intermediate_payload,
    curve_coordinates_from_plot_transform,
    final_cache_path,
    input_digest,
    intermediate_cache_path,
    load_final_tokens,
    publish_torch_no_replace,
    validate_intermediate_cache,
)
from src.models.curve_aware_sampling import curve_pixel_coordinates


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(resblocks=nn.ModuleList([nn.Identity() for _ in range(12)]))
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward_intermediates(self, images, **kwargs):
        self.calls.append(kwargs)
        return {"image_intermediates": [torch.ones(images.shape[0], 196, 768)]}


class FakeClipAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(visual=FakeVisual())


class CurveAwareClipTests(unittest.TestCase):
    def test_block6_openclip_api_semantics_and_shape(self):
        clip = FakeClipAD()
        encoder = FrozenCLIPBlock6Encoder(clip)
        result = encoder(torch.zeros(2, 3, 224, 224))
        self.assertEqual(result.shape, (2, 196, 768))
        call = clip.model.visual.calls[0]
        self.assertEqual(call["indices"], [5])
        self.assertTrue(call["stop_early"])
        self.assertTrue(call["normalize_intermediates"])
        self.assertTrue(call["intermediates_only"])
        self.assertEqual(call["output_fmt"], "NLC")
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))

    def test_curve_transform_matches_original_axes_and_orientation(self):
        increasing = np.linspace(0, 1, 224)
        constant = np.full(224, 0.5)
        coordinates = curve_coordinates_from_plot_transform(
            np.stack((increasing, constant))
        )
        pixel = coordinates["curve_pixel_xy"].numpy()
        grid = coordinates["curve_grid_xy"].numpy()
        np.testing.assert_allclose(
            pixel[:, :, 0],
            np.broadcast_to(np.arange(8, 224, 16), (2, 14)),
            atol=1e-5,
        )
        np.testing.assert_allclose(pixel[1, :, 1], 112.0, atol=1e-5)
        # For y=x/223 and x display centres 8..216, the filled 224px axes
        # yields top-origin y=224-x exactly.
        np.testing.assert_allclose(
            pixel[0, :, 1], 224.0 - np.arange(8, 224, 16), atol=1e-5
        )
        self.assertGreater(pixel[0, 0, 1], pixel[0, -1, 1])
        np.testing.assert_allclose(
            grid[:, :, 0], np.broadcast_to(np.arange(14), (2, 14)), atol=1e-5
        )
        np.testing.assert_allclose(grid[1, :, 1], 6.5, atol=1e-5)

    def test_plot_transform_matches_independent_analytic_coordinates(self):
        rng = np.random.default_rng(104)
        windows = rng.uniform(0.0, 1.0, size=(3, 224))
        rendered = curve_coordinates_from_plot_transform(windows)
        analytic = curve_pixel_coordinates(windows, y_limits=(0.0, 1.0))
        np.testing.assert_allclose(
            rendered["curve_pixel_xy"][..., 0].numpy(), analytic.pixel_x, atol=1e-10
        )
        np.testing.assert_allclose(
            rendered["curve_pixel_xy"][..., 1].numpy(), analytic.pixel_y, atol=2e-5
        )
        np.testing.assert_allclose(
            rendered["curve_grid_xy"][..., 1].numpy(), analytic.patch_y, atol=2e-6
        )

    def test_coordinates_follow_pixels_from_original_draw_image(self):
        from src.preprocessing.preprocess import draw_image

        values = np.linspace(0.0, 1.0, 224)
        expected = curve_coordinates_from_plot_transform(values[None])["curve_pixel_xy"][0]
        with tempfile.TemporaryDirectory() as directory:
            image = draw_image(
                "linear",
                directory,
                values,
                np.arange(224),
                image_size=(224, 224),
                dpi=100,
                plot_params=("-", 1, "*", 0.1, "black", (0, 1)),
            )
        darkness = 1.0 - image.mean(axis=0)
        for x_edge, y_edge in expected.numpy():
            left = max(0, int(np.floor(x_edge)) - 1)
            right = min(224, int(np.ceil(x_edge)) + 2)
            # Ignore the default axes spines at the image's outermost rows;
            # they are part of the original raster but not the curve.
            local = darkness[2:222, left:right]
            row = int(np.unravel_index(np.argmax(local), local.shape)[0]) + 2
            # Raster rows are represented by centre coordinate row+0.5.  A
            # one-pixel tolerance accounts for the one-pixel antialiased line.
            self.assertLessEqual(abs((row + 0.5) - float(y_edge)), 1.5)

    def test_final_cache_is_reused_and_intermediate_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = {"dataset": "D", "signal": "S"}
            digest = input_digest(np.arange(224, dtype=np.float64))
            final_path = final_cache_path(root / "final", context, digest)
            final_path.parent.mkdir(parents=True)
            torch.save({
                "cache_version": 1,
                "input_digest": digest,
                "patch": torch.ones(1, 196, 1, 768),
            }, final_path)
            tokens, found = load_final_tokens([root / "missing", root / "final"], context, digest, 1)
            self.assertEqual(tokens.shape, (1, 196, 768))
            self.assertEqual(found, final_path)

            target = intermediate_cache_path(root / "curve", context, digest)
            coords = curve_coordinates_from_plot_transform(np.zeros((1, 224)))
            payload = build_intermediate_payload(
                torch.ones(1, 196, 768), coords, digest=digest, source_final_cache=found
            )
            self.assertTrue(publish_torch_no_replace(target, payload))
            original = target.read_bytes()
            changed = dict(payload)
            changed["intermediate_patch"] = torch.zeros(1, 196, 768)
            self.assertFalse(publish_torch_no_replace(target, changed))
            self.assertEqual(target.read_bytes(), original)
            stored = torch.load(target, map_location="cpu", weights_only=True)
            validate_intermediate_cache(stored, digest=digest, expected_windows=1, path=target)

    def test_all_eleven_datasets_use_strict_index_and_three_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = [f"D{i}" for i in range(11)]
            counts = {dataset: i % 3 + 1 for i, dataset in enumerate(datasets)}
            with (root / "datasets.csv").open("w", encoding="utf-8") as handle:
                for dataset in datasets:
                    names = tuple(f"s{j}" for j in range(counts[dataset]))
                    handle.write(f'{dataset},"{names}"\n')
                    folder = root / dataset
                    folder.mkdir()
                    for name in names:
                        (folder / f"{name}.csv").write_text("timestamp,value\n1,0\n", encoding="utf-8")
                    (folder / "extra.csv").write_text("timestamp,value\n1,0\n", encoding="utf-8")
            config = {"datasets": datasets, "expected_signal_counts": counts}
            tasks, ignored = discover_tasks([config], root)
            self.assertEqual(len(tasks), sum(counts.values()))
            self.assertTrue(all(value == ["extra.csv"] for value in ignored.values()))
            shards = [shard_tasks(tasks, index, 3) for index in range(3)]
            flattened = [task for shard in shards for task in shard]
            self.assertEqual(set(flattened), set(tasks))
            self.assertEqual(sum(map(len, shards)), len(tasks))


if __name__ == "__main__":
    unittest.main()
