#!/usr/bin/env python3
"""Schedule immutable synthetic RCCA cache builds across free GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "experiments/bimodal_reference/build_rcca_synthetic_cache.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    base_root = Path(config["base_training_cache_root"])
    datasets = args.datasets or config["datasets"]
    unknown = sorted(set(datasets) - set(config["datasets"]))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    jobs = [(seed, dataset) for seed in config["stability_seeds"]
            for dataset in datasets]
    jobs.sort(key=lambda item: (base_root / f"{item[1]}.pt").stat().st_size,
              reverse=True)
    active: dict[int, tuple] = {}
    records = []
    started = time.time()
    cache_root = args.output_root / "synthetic_training_cache"
    while jobs or active:
        for gpu in args.gpus:
            if gpu in active or not jobs:
                continue
            seed, dataset = jobs.pop(0)
            output = cache_root / f"seed_{seed}" / f"{dataset}.pt"
            log = args.output_root / "cache_logs" / f"seed_{seed}" / f"{dataset}.log"
            if output.is_file():
                records.append({"seed": seed, "dataset": dataset,
                                "status": "existing", "output": str(output)})
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w")
            command = [args.python, str(BUILDER), "--config", str(args.config),
                       "--dataset", dataset, "--seed", str(seed),
                       "--output", str(output), "--device", f"cuda:{gpu}"]
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
            active[gpu] = (process, handle, seed, dataset, output, log, time.time())
            print(json.dumps({"event": "started", "gpu": gpu, "seed": seed,
                              "dataset": dataset, "pid": process.pid}), flush=True)
        time.sleep(1)
        for gpu, job in list(active.items()):
            process, handle, seed, dataset, output, log, job_started = job
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[gpu]
            record = {"seed": seed, "dataset": dataset,
                      "status": "passed" if code == 0 and output.is_file() else "failed",
                      "returncode": code, "gpu": gpu,
                      "elapsed_wall_seconds": time.time() - job_started,
                      "output": str(output), "log": str(log)}
            records.append(record)
            print(json.dumps({"event": "finished", **record}), flush=True)
            if record["status"] == "failed":
                for other in active.values():
                    other[0].terminate()
                    other[1].close()
                (args.output_root / "cache_manifest.json").write_text(json.dumps(
                    {"status": "failed", "records": records}, indent=2))
                return 1
    (args.output_root / "cache_manifest.json").write_text(json.dumps({
        "status": "passed", "records": records,
        "elapsed_wall_seconds": time.time() - started,
        "labels_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
