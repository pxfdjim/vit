#!/usr/bin/env python3
"""Evaluate final APRA scores in parallel CPU shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "experiments/bimodal_reference/evaluate_apra.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    active = []
    for index in range(args.workers):
        log = args.output_root / "evaluation_logs" / f"worker_{index:02d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w")
        command = [args.python, str(WORKER), "--config", str(args.config),
                   "--output-root", str(args.output_root), "--worker-index", str(index),
                   "--num-workers", str(args.workers)]
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, text=True)
        active.append((process, handle, index, log))
    records = []
    while active:
        time.sleep(1)
        for job in list(active):
            process, handle, index, log = job
            code = process.poll()
            if code is None:
                continue
            handle.close()
            active.remove(job)
            records.append({"worker": index, "returncode": code,
                            "status": "passed" if code == 0 else "failed",
                            "log": str(log)})
    if not all(record["status"] == "passed" for record in records):
        print(json.dumps({"status": "failed", "records": records}, indent=2))
        return 1
    command = [args.python, str(WORKER), "--config", str(args.config),
               "--output-root", str(args.output_root), "--num-workers", str(args.workers),
               "--combine-shards"]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
