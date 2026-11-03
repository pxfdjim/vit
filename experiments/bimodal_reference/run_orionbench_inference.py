#!/usr/bin/env python3
"""Run dataset-native AMR inference without opening labels or computing F1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bimodal_reference.orionbench_protocol import DATASETS, evaluation_frame, signal_names
from experiments.temporal_matching.runner import publish_json_no_replace, publish_npz_no_replace
from src.models.bimodal_reference_vit4ts import BimodalReferenceBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--block-cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tasks = [(d, s) for d in DATASETS for s in signal_names(args.data_root, d)]
    tasks = tasks[args.worker_index::args.num_workers]
    backend = BimodalReferenceBackend(config, args.device, args.block_cache_root)
    variants = config["variants"]
    failures, records = 0, []
    for ordinal, (dataset, signal) in enumerate(tasks, 1):
        started = time.perf_counter()
        try:
            frame = evaluation_frame(args.data_root, dataset, signal)
            outputs = backend.predict_variants(frame, variants, {"dataset": dataset, "signal": signal})
            for variant, result in outputs.items():
                score_path = args.output_root / "point_scores" / dataset / signal / f"{variant}.npz"
                diagnostic_path = args.output_root / "path_diagnostics" / dataset / signal / f"{variant}.npz"
                peak = torch.cuda.max_memory_allocated(args.device) if torch.cuda.is_available() else -1
                publish_npz_no_replace(
                    score_path, scores=np.asarray(result["scores"], np.float64),
                    timestamps=np.asarray(result["timestamps"]),
                    metadata_json=np.asarray([json.dumps(result["metadata"], sort_keys=True)]),
                    timing_json=np.asarray([json.dumps(result["timing"], sort_keys=True)]),
                    peak_cuda_memory_bytes=np.asarray([peak], np.int64),
                )
                publish_npz_no_replace(diagnostic_path, **result["diagnostics"])
            records.append({"dataset": dataset, "signal": signal, "points": len(frame),
                            "variants": len(outputs), "elapsed_seconds": time.perf_counter() - started,
                            "labels_opened": False, "status": "passed"})
            print(f"[{ordinal}/{len(tasks)}] {dataset}/{signal}", flush=True)
        except Exception as error:
            failures += 1
            records.append({"dataset": dataset, "signal": signal, "status": "failed",
                            "error": f"{type(error).__name__}: {error}"})
            print(f"[{ordinal}/{len(tasks)}] failed {dataset}/{signal}: {error}", flush=True)
    manifest = {"status": "passed" if failures == 0 else "failed", "worker_index": args.worker_index,
                "num_workers": args.num_workers, "records": records, "failures": failures,
                "labels_opened": False}
    publish_json_no_replace(args.output_root / "inference_manifests" /
                            f"worker_{args.worker_index:02d}_of_{args.num_workers:02d}.json", manifest)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
