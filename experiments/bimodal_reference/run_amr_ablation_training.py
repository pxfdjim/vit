#!/usr/bin/env python3
"""Schedule immutable dataset ablation trainings across explicit GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "experiments/bimodal_reference/train_amr_ablation.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("initial", "loss_ablation", "all"), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    variants = [name for name, spec in cfg["training_variants"].items()
                if args.stage == "all" or spec["stage"] == args.stage]
    cache_root = Path(cfg["training_cache_root"])
    # Long datasets first reduces the final scheduler tail.
    windows = {}
    for dataset in cfg["datasets"]:
        import torch
        windows[dataset] = int(torch.load(cache_root / f"{dataset}.pt", map_location="cpu",
                                          weights_only=False)["windows"])
    pending = sorted(((variant, dataset) for variant in variants for dataset in cfg["datasets"]),
                     key=lambda pair: windows[pair[1]], reverse=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    active = {}
    records = []
    started = time.time()
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            variant, dataset = pending.pop(0)
            output = args.output_root / "checkpoints" / variant / f"{dataset}.pt"
            log = args.output_root / "training_logs" / variant / f"{dataset}.log"
            if output.is_file():
                records.append({"variant": variant, "dataset": dataset,
                                "status": "existing", "output": str(output)})
                continue
            output.parent.mkdir(parents=True, exist_ok=True); log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            command = [args.python, str(TRAIN), "--config", str(args.config),
                       "--variant", variant, "--cache", str(cache_root / f"{dataset}.pt"),
                       "--output", str(output), "--device", f"cuda:{gpu}"]
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
            active[gpu] = (process, handle, variant, dataset, output, log, time.time())
            print(json.dumps({"event": "started", "gpu": gpu, "variant": variant,
                              "dataset": dataset, "pid": process.pid}), flush=True)
        time.sleep(1)
        for gpu, item in list(active.items()):
            process, handle, variant, dataset, output, log, job_started = item
            code = process.poll()
            if code is None:
                continue
            handle.close(); del active[gpu]
            record = {"variant": variant, "dataset": dataset,
                      "status": "passed" if code == 0 and output.is_file() else "failed",
                      "returncode": code, "gpu": gpu, "elapsed_wall_seconds": time.time() - job_started,
                      "output": str(output), "log": str(log)}
            records.append(record); print(json.dumps({"event": "finished", **record}), flush=True)
            if record["status"] != "passed":
                for other in active.values():
                    other[0].terminate(); other[1].close()
                manifest = {"status": "failed", "stage": args.stage, "records": records,
                            "elapsed_wall_seconds": time.time() - started}
                (args.output_root / f"training_manifest_{args.stage}.json").write_text(
                    json.dumps(manifest, indent=2))
                return 1
    manifest = {"status": "passed", "stage": args.stage, "variants": variants,
                "records": records, "elapsed_wall_seconds": time.time() - started}
    (args.output_root / f"training_manifest_{args.stage}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
