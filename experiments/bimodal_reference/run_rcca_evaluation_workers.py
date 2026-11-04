#!/usr/bin/env python3
"""Run RCCA/CADM metric evaluation shards and combine them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "experiments/bimodal_reference/evaluate_rcca_cadm.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(); active = []; records = []; started = time.time()
    for index in range(args.workers):
        log = args.output_root / "evaluation_logs" / f"worker_{index:02d}.log"
        log.parent.mkdir(parents=True, exist_ok=True); handle = log.open("w")
        command = [args.python, str(EVALUATOR), "--config", str(args.config),
                   "--output-root", str(args.output_root), "--worker-index", str(index),
                   "--num-workers", str(args.workers)]
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, text=True)
        active.append((process, handle, index, log))
    while active:
        time.sleep(2)
        for job in list(active):
            process, handle, index, log = job
            code = process.poll()
            if code is None:
                continue
            handle.close(); active.remove(job)
            record = {"worker": index, "returncode": code,
                      "status": "passed" if code == 0 else "failed", "log": str(log)}
            records.append(record); print(json.dumps(record), flush=True)
    if not all(record["status"] == "passed" for record in records):
        return 1
    command = [args.python, str(EVALUATOR), "--config", str(args.config),
               "--output-root", str(args.output_root), "--num-workers", str(args.workers),
               "--combine-shards"]
    code = subprocess.call(command, cwd=ROOT)
    (args.output_root / "evaluation_workers.json").write_text(json.dumps({
        "status": "passed" if code == 0 else "failed", "records": records,
        "elapsed_wall_seconds": time.time() - started, "combine_returncode": code,
    }, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
