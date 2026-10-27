#!/usr/bin/env bash
set -euo pipefail
set -C

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/pengxf/anaconda3/envs/vit/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/pengxf/jim/VLM4TS/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/data/runs/vfg_vit4ts_20261027}
SUITE=${SUITE:-all}
LIMIT_PER_WORKER=${LIMIT_PER_WORKER:-}
SUMMARIZE=${SUMMARIZE:-1}

if [[ -z "${DEVICES:-}" ]]; then
    echo "DEVICES is required. Inspect GPU occupancy first, then pass at least three comma-separated devices." >&2
    echo "Example: DEVICES=cuda:1,cuda:2,cuda:3 $0" >&2
    exit 2
fi

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
NUM_WORKERS=${#DEVICE_LIST[@]}
if (( NUM_WORKERS < 3 )); then
    echo "At least three parallel workers are required; received ${NUM_WORKERS}." >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_DIR=$(mktemp -d "${OUTPUT_ROOT}/logs/attempt_XXXXXX")
pids=()
for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
    log_path=${LOG_DIR}/worker_$(printf '%03d' "${worker_id}").log
    limit_args=()
    if [[ -n "${LIMIT_PER_WORKER}" ]]; then
        limit_args=(--limit "${LIMIT_PER_WORKER}")
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/runner.py" \
        --data-root "${DATA_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --suite "${SUITE}" \
        --num-workers "${NUM_WORKERS}" \
        --worker-id "${worker_id}" \
        --device "${DEVICE_LIST[worker_id]}" \
        "${limit_args[@]}" \
        > "${log_path}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "One or more workers failed. Existing outputs were retained; inspect ${LOG_DIR}." >&2
    exit "${status}"
fi

if [[ "${SUMMARIZE}" == "1" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize.py" \
        --data-root "${DATA_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --suite "${SUITE}"
fi
