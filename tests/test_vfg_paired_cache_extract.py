import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.vfg_paired.extract_visual_cache import (
    CacheTask,
    discover_tasks,
    extract_one,
    guided_reference_summary,
    load_yahoo_signal,
    manifest_path,
    run,
    shard_tasks,
    window_count,
)
from experiments.vfg_paired.runner import (
    discover_tasks as runner_discover_tasks,
    load_signal as runner_load_signal,
)


def write_index(root: Path, signals: tuple[str, ...]) -> None:
    with (root / "datasets.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["YAHOOA1", repr(signals)])


def write_signal(path: Path, length: int = 840) -> None:
    pd.DataFrame(
        {
            "timestamp": np.arange(1000, 1000 + length),
            "value": np.linspace(-2.0, 3.0, length),
            "is_anomaly": np.zeros(length, dtype=int),
        }
    ).to_csv(path, index=False)


class YahooVisualCacheEntryTests(unittest.TestCase):
    def make_tree(self, root: Path) -> tuple[dict, Path]:
        directory = root / "Yahoo_S5" / "A1Benchmark"
        directory.mkdir(parents=True)
        write_index(root, ("signal_b", "signal_a"))
        write_signal(directory / "signal_a.csv")
        write_signal(directory / "signal_b.csv", length=741)
        write_signal(directory / "combined_extra.csv")
        config = {
            "datasets": ["YAHOOA1"],
            "expected_signal_counts": {"YAHOOA1": 2},
            "model": {
                "window_size": 224,
                "step_size": 56,
                "patch_size": 16,
                "image_size": [224, 224],
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config, config_path

    def test_discovery_uses_index_order_and_ignores_unindexed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_tree(root)
            tasks, ignored = discover_tasks(config, root)
            self.assertEqual([task.signal for task in tasks], ["signal_b", "signal_a"])
            self.assertEqual(ignored, {"YAHOOA1": ["combined_extra.csv"]})

    def test_yahoo_rows_are_one_based_and_source_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_tree(root)
            tasks, _ = discover_tasks(config, root)
            frame = load_yahoo_signal(tasks[0])
            np.testing.assert_array_equal(
                frame["timestamp"].to_numpy(), np.arange(1, len(frame) + 1)
            )
            self.assertEqual(frame.iloc[0]["value"], -2.0)
            self.assertEqual(frame.iloc[-1]["value"], 3.0)

    def test_discovery_and_loaded_frame_match_paired_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_tree(root)
            tasks, _ = discover_tasks(config, root)
            runner_tasks = runner_discover_tasks(config, root)
            self.assertEqual(
                [(task.dataset, task.signal, task.path) for task in tasks],
                runner_tasks,
            )
            expected = runner_load_signal(tasks[0].path, tasks[0].dataset)
            pd.testing.assert_frame_equal(load_yahoo_signal(tasks[0]), expected)

    def test_shards_are_disjoint_complete_and_window_gate_is_exact(self):
        tasks = [CacheTask("YAHOOA1", f"s{index}", Path(f"s{index}.csv")) for index in range(11)]
        shards = [shard_tasks(tasks, index, 3) for index in range(3)]
        flattened = [task.signal for shard in shards for task in shard]
        self.assertEqual(set(flattened), {task.signal for task in tasks})
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(window_count(741), 10)
        self.assertEqual(window_count(840), 12)
        self.assertEqual(guided_reference_summary(10)["guided_min_reference_windows"], 3)
        self.assertEqual(guided_reference_summary(10)["guided_queries_below_target_5"], 6)
        self.assertTrue(guided_reference_summary(10)["guided_pool_is_ragged"])
        self.assertFalse(guided_reference_summary(12)["guided_pool_is_ragged"])

    def test_dry_run_never_constructs_clip_backend_or_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config_path = self.make_tree(root)
            cache_root = root / "cache"
            manifest_dir = root / "manifests"
            args = Namespace(
                config=config_path,
                data_root=root,
                cache_root=cache_root,
                manifest_dir=manifest_dir,
                device="cpu",
                worker_index=0,
                num_workers=2,
                limit=None,
                dry_run=True,
            )
            with patch(
                "experiments.vfg_paired.extract_visual_cache.create_backend",
                side_effect=AssertionError("backend must remain lazy"),
            ):
                self.assertEqual(run(args), 0)
            self.assertFalse(cache_root.exists())
            self.assertFalse(manifest_dir.exists())

    def test_config_count_must_match_authoritative_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_tree(root)
            config["expected_signal_counts"]["YAHOOA1"] = 3
            with self.assertRaisesRegex(ValueError, "datasets.csv indexes 2"):
                discover_tasks(config, root)

    def test_existing_cache_is_delegated_for_validation_and_not_replaced(self):
        class FakeBackend:
            def __init__(self, root: Path):
                self.root = root
                self.received_visual = None

            def _input_digest(self, values):
                self.raw = values.copy()
                return "d" * 64

            def _cache_path(self, context, digest):
                return self.root / context["dataset"] / context["signal"] / f"visual_{digest[:20]}.pt"

            def _load_or_extract_visual(self, visual_values, context, digest):
                self.received_visual = visual_values.copy()
                path = self._cache_path(context, digest)
                self.asserted_bytes = path.read_bytes()
                return {"input_digest": digest}, True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_tree(root)
            task = discover_tasks(config, root)[0][0]
            frame = load_yahoo_signal(task)
            backend = FakeBackend(root / "cache")
            path = backend._cache_path(
                {"dataset": task.dataset, "signal": task.signal}, "d" * 64
            )
            path.parent.mkdir(parents=True)
            path.write_bytes(b"existing-cache")
            record = extract_one(backend, task, frame)
            self.assertEqual(path.read_bytes(), b"existing-cache")
            self.assertEqual(backend.asserted_bytes, b"existing-cache")
            self.assertEqual(record["status"], "validated_existing")
            self.assertTrue(record["all_paired_variants_eligible"])
            self.assertTrue(record["guided_pool_is_ragged"])
            self.assertTrue(np.isfinite(backend.received_visual).all())
            self.assertGreaterEqual(backend.received_visual.min(), 0.0)
            self.assertLessEqual(backend.received_visual.max(), 1.0)

    def test_worker_manifest_is_atomic_no_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config_path = self.make_tree(root)
            args = Namespace(
                config=config_path,
                data_root=root,
                cache_root=root / "cache",
                manifest_dir=root / "manifests",
                device="cpu",
                worker_index=0,
                num_workers=1,
                limit=1,
                dry_run=False,
            )
            record = {
                "dataset": "YAHOOA1",
                "signal": "signal_b",
                "status": "validated_existing",
            }
            with patch(
                "experiments.vfg_paired.extract_visual_cache.create_backend",
                return_value=object(),
            ), patch(
                "experiments.vfg_paired.extract_visual_cache.extract_one",
                return_value=record,
            ):
                self.assertEqual(run(args), 0)
            path = manifest_path(args)
            original = path.read_bytes()
            with patch(
                "experiments.vfg_paired.extract_visual_cache.create_backend",
                return_value=object(),
            ), patch(
                "experiments.vfg_paired.extract_visual_cache.extract_one",
                return_value={**record, "status": "published_or_peer_won"},
            ):
                self.assertEqual(run(args), 0)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
