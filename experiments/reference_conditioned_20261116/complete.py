"""Run all score files and the immutable OrionBench evaluator after training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODES = ("posterior", "no_reference", "t3_direct")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 3, 6, 7])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--wait", action="store_true", help="wait for all training checkpoints")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = run / "training_manifest.json"
    while args.wait and not manifest.exists():
        time.sleep(15)
    training = json.loads(manifest.read_text())
    if training["status"] != "passed":
        raise RuntimeError("all training jobs must pass before inference")
    print(json.dumps({"stage": "train_inference_input_audit"}), flush=True)
    subprocess.run([args.python, str(HERE / "audit_consistency.py"), str(run),
                    f"cuda:{args.gpus[0]}"], cwd=ROOT, check=True)
    for mode in MODES:
        config = run / "configs" / f"{mode}.json"
        output = run / "inference" / mode
        inference = [args.python, str(HERE / "predict_workers.py"),
                     "--config", str(config), "--output-root", str(output),
                     "--gpus", *map(str, args.gpus), "--python", args.python]
        print(json.dumps({"stage": "inference", "mode": mode}), flush=True)
        subprocess.run(inference, cwd=ROOT, check=True)
        evaluation = [args.python, str(ROOT / "src/evaluation/evaluate_workers.py"),
                      "--config", str(config), "--output-root", str(output),
                      "--workers", str(args.workers), "--python", args.python]
        print(json.dumps({"stage": "evaluation", "mode": mode}), flush=True)
        subprocess.run(evaluation, cwd=ROOT, check=True)
    print(json.dumps({"stage": "unlabeled_mechanism"}), flush=True)
    subprocess.run([args.python, str(HERE / "mechanism.py"), "--run", str(run),
                    "--device", f"cuda:{args.gpus[0]}"], cwd=ROOT, check=True)
    print(json.dumps({"stage": "independent_validation"}), flush=True)
    subprocess.run([args.python, str(HERE / "validate.py"), str(run)], cwd=ROOT, check=True)
    print(json.dumps({"stage": "report"}), flush=True)
    subprocess.run([args.python, str(HERE / "report.py"), str(run)], cwd=ROOT, check=True)
    print(json.dumps({"stage": "all_completed"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
