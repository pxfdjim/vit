#!/usr/bin/env python3
"""Schedule all APRA+TREM dataset/variant training jobs without overwriting."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "experiments/bimodal_reference/train_apra_trem.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    datasets = args.datasets or config["datasets"]
    cache = Path(config["rcca_training_cache"])
    jobs = [(variant, dataset) for variant in config["variants"] for dataset in datasets]
    jobs.sort(key=lambda item: (cache / f"{item[1]}.pt").stat().st_size, reverse=True)
    active = {}; records = []; started = time.time()
    while jobs or active:
        for gpu in args.gpus:
            if gpu in active or not jobs:
                continue
            variant, dataset = jobs.pop(0)
            output = args.output_root / "checkpoints" / variant / f"{dataset}.pt"
            log = args.output_root / "training_logs" / variant / f"{dataset}.log"
            if output.is_file():
                records.append({"variant": variant, "dataset": dataset, "status": "existing"})
                continue
            output.parent.mkdir(parents=True, exist_ok=True); log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w")
            command = [args.python, str(TRAIN), "--config", str(args.config),
                       "--variant", variant, "--cache", str(cache / f"{dataset}.pt"),
                       "--output", str(output), "--device", f"cuda:{gpu}"]
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
            active[gpu] = (process, handle, variant, dataset, output, log, time.time())
            print(json.dumps({"event": "started", "gpu": gpu, "variant": variant,
                              "dataset": dataset, "pid": process.pid}), flush=True)
        time.sleep(1)
        for gpu, job in list(active.items()):
            process, handle, variant, dataset, output, log, tick = job
            code = process.poll()
            if code is None:
                continue
            handle.close(); del active[gpu]
            record = {"variant": variant, "dataset": dataset,
                      "status": "passed" if code == 0 and output.is_file() else "failed",
                      "returncode": code, "gpu": gpu,
                      "elapsed_wall_seconds": time.time() - tick,
                      "output": str(output), "log": str(log)}
            records.append(record); print(json.dumps({"event": "finished", **record}), flush=True)
            if record["status"] == "failed":
                for other in active.values():
                    other[0].terminate(); other[1].close()
                (args.output_root / "training_manifest.json").write_text(
                    json.dumps({"status": "failed", "records": records}, indent=2) + "\n")
                return 1
    (args.output_root / "training_manifest.json").write_text(json.dumps({
        "status": "passed", "records": records,
        "elapsed_wall_seconds": time.time() - started, "labels_used": False,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
