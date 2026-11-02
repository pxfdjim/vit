#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/pengxf/anaconda3/envs/vit/bin/python}"
export CLIP_PRETRAINED=openai
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vit4ts-matplotlib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$ROOT_DIR"
mkdir -p data/runs/vit4ts_challenges/scale \
         data/runs/vit4ts_challenges/reference \
         data/runs/vit4ts_challenges/reference_real \
         data/runs/vit4ts_challenges/audit

# Three independent first-stage jobs run concurrently.  Each scale case writes
# separate files; the two reference jobs use separate output directories.
jobs=(
  "scale:aws" "scale:ad" "scale:traffic" "scale:artificial"
  "scale:tweets" "scale:msl" "scale:smap"
  "reference:synthetic" "reference:real" "audit:baseline"
)
active=0
failed=0
for job in "${jobs[@]}"; do
  kind="${job%%:*}"
  name="${job#*:}"
  case "$kind" in
    scale)
      (
        "$PYTHON_BIN" src/models/lib/scale_diagnostic.py --case "$name"
        "$PYTHON_BIN" src/models/lib/scale_diagnostic.py --case "$name" --add-normal-ref
      ) >"data/runs/vit4ts_challenges/scale/${name}.log" 2>&1 &
      ;;
    reference)
      if [[ "$name" == synthetic ]]; then
        "$PYTHON_BIN" src/models/lib/reference_diagnostic.py \
          >data/runs/vit4ts_challenges/reference/run.log 2>&1 &
      else
        "$PYTHON_BIN" src/models/lib/reference_real_anchor.py \
          >data/runs/vit4ts_challenges/reference_real/run.log 2>&1 &
      fi
      ;;
    audit)
      "$PYTHON_BIN" src/models/lib/challenge_audit.py \
        >data/runs/vit4ts_challenges/audit/run.log 2>&1 &
      ;;
  esac
  ((active+=1))
  if ((active == 3)); then
    wait -n || failed=1
    ((active-=1))
  fi
done
while ((active > 0)); do
  wait -n || failed=1
  ((active-=1))
done
if ((failed != 0)); then
  echo "At least one experiment failed; inspect per-job logs." >&2
  exit 1
fi

"$PYTHON_BIN" src/models/lib/scale_summary.py
"$PYTHON_BIN" src/models/lib/challenge_audit.py --check-challenges
