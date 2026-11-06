#!/usr/bin/env python3
"""Run final APRA inference across available GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "experiments/bimodal_reference/run_apra_inference.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    active = []
    started = time.time()
    for index, gpu in enumerate(args.gpus):
        log = args.output_root / "inference_logs" / f"worker_{index:02d}_of_{len(args.gpus):02d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w")
        command = [args.python, str(WORKER), "--config", str(args.config),
                   "--output-root", str(args.output_root), "--device", f"cuda:{gpu}",
                   "--worker-index", str(index), "--num-workers", str(len(args.gpus)),
                   "--skip-complete"]
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, text=True)
        active.append((process, handle, index, gpu, log))
        print(json.dumps({"event": "started", "worker": index, "gpu": gpu,
                          "pid": process.pid}), flush=True)
    records = []
    while active:
        time.sleep(2)
        for job in list(active):
            process, handle, index, gpu, log = job
            code = process.poll()
            if code is None:
                continue
            handle.close()
            active.remove(job)
            record = {"worker": index, "gpu": gpu, "returncode": code,
                      "status": "passed" if code == 0 else "failed", "log": str(log)}
            records.append(record)
            print(json.dumps({"event": "finished", **record}), flush=True)
    status = "passed" if all(record["status"] == "passed" for record in records) else "failed"
    (args.output_root / "inference_workers.json").write_text(json.dumps(
        {"status": status, "records": records,
         "elapsed_wall_seconds": time.time() - started}, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
