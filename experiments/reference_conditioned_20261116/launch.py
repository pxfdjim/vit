"""Run the three frozen-protocol model variants on separate available GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = json.loads((ROOT / "configs/model.json").read_text())
MODES = ("posterior", "no_reference", "t3_direct")


def make_configs(run: Path) -> None:
    for mode in MODES:
        config = json.loads(json.dumps(BASE))
        config["model_config"] = {**config["model_config"],
                                  "architecture": "reference_conditioned_v1",
                                  "mode": mode}
        config["checkpoint_root"] = str(run / "checkpoints" / mode)
        target = run / "configs" / f"{mode}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and json.loads(target.read_text()) != config:
            raise RuntimeError(f"existing config changed: {target}")
        if not target.exists():
            target.write_text(json.dumps(config, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 3, 6, 7])
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    run = args.run.resolve()
    make_configs(run)
    tasks = [(mode, dataset) for mode in MODES for dataset in BASE["datasets"]]
    active: dict[int, tuple[subprocess.Popen, object, str, str, Path]] = {}
    failed = []
    started = time.time()
    while tasks or active:
        for gpu in args.gpus:
            if gpu in active or not tasks:
                continue
            mode, dataset = tasks.pop(0)
            checkpoint = run / "checkpoints" / mode / f"{dataset}.pt"
            if checkpoint.exists():
                print(json.dumps({"event": "skip_checkpoint", "mode": mode,
                                  "dataset": dataset}), flush=True)
                continue
            source = ROOT / BASE["synthetic_training_cache_root"] / f"{dataset}.pt"
            if not source.exists():
                # Source caches were published under dataset-specific files.
                candidates = list((ROOT / BASE["synthetic_training_cache_root"]).glob(f"**/{dataset}.pt"))
                if len(candidates) != 1:
                    raise FileNotFoundError(f"training cache for {dataset}: {candidates}")
                source = candidates[0]
            log = run / "training_logs" / mode / f"{dataset}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w")
            command = [args.python, str(HERE / "train.py"),
                       "--config", str(run / "configs" / f"{mode}.json"),
                       "--training-data", str(source), "--output", str(checkpoint),
                       "--device", f"cuda:{gpu}"]
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT, env=os.environ.copy())
            active[gpu] = (process, handle, mode, dataset, log)
            print(json.dumps({"event": "started", "gpu": gpu, "pid": process.pid,
                              "mode": mode, "dataset": dataset}), flush=True)
        time.sleep(3)
        for gpu, job in list(active.items()):
            process, handle, mode, dataset, log = job
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[gpu]
            if status:
                failed.append({"mode": mode, "dataset": dataset,
                               "exit_code": status, "log": str(log)})
            print(json.dumps({"event": "finished", "gpu": gpu, "mode": mode,
                              "dataset": dataset, "exit_code": status}), flush=True)
    result = {"status": "passed" if not failed else "failed", "failures": failed,
              "elapsed_wall_seconds": time.time() - started}
    (run / "training_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
