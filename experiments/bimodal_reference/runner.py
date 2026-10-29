#!/usr/bin/env python3
"""Run trained bimodal reference controls using the validated runner."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.temporal_matching.runner import parser, run

if __name__ == "__main__":
    args = parser().parse_args()
    if args.config == Path(__file__).resolve().parents[1] / "temporal_matching" / "config.json":
        args.config = HERE / "config.json"
    if args.backend_module == "src.models.temporal_vit4ts":
        args.backend_module = "src.models.bimodal_reference_vit4ts"
    raise SystemExit(run(args))
