#!/usr/bin/env bash
set -euo pipefail
set -C

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/pengxf/anaconda3/envs/vit/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/pengxf/jim/VLM4TS/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/data/runs/vfg_paired_20261027}
CACHE_ROOT=${CACHE_ROOT:-${OUTPUT_ROOT}/feature_cache}
LIMIT_PER_WORKER=${LIMIT_PER_WORKER:-}
SUMMARIZE=${SUMMARIZE:-1}

if [[ -z "${DEVICES:-}" ]]; then
    echo "DEVICES is required after checking GPU occupancy; example: DEVICES=cuda:1,cuda:2,cuda:3 $0" >&2
    exit 2
fi
IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
NUM_WORKERS=${#DEVICE_LIST[@]}
if (( NUM_WORKERS < 3 )); then
    echo "At least three explicit devices are required; received ${NUM_WORKERS}." >&2
    exit 2
fi
declare -A SEEN_DEVICES=()
for device in "${DEVICE_LIST[@]}"; do
    if [[ ! "${device}" =~ ^cuda:[0-9]+$ && ! "${device}" =~ ^cpu(:[0-9]+)?$ ]]; then
        echo "Each device must be an explicit CUDA index such as cuda:2 or a CPU worker label such as cpu:0; received ${device}." >&2
        exit 2
    fi
    if [[ -n "${SEEN_DEVICES[${device}]:-}" ]]; then
        echo "Each worker must use a distinct GPU; duplicate ${device}." >&2
        exit 2
    fi
    SEEN_DEVICES[${device}]=1
done

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_DIR=$(mktemp -d "${OUTPUT_ROOT}/logs/attempt_XXXXXX")
pids=()
for ((worker_id=0; worker_id<NUM_WORKERS; worker_id++)); do
    limit_args=()
    [[ -n "${LIMIT_PER_WORKER}" ]] && limit_args=(--limit "${LIMIT_PER_WORKER}")
    "${PYTHON_BIN}" "${SCRIPT_DIR}/runner.py" \
        --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
        --cache-root "${CACHE_ROOT}" --num-workers "${NUM_WORKERS}" \
        --worker-id "${worker_id}" --device "${DEVICE_LIST[worker_id]}" \
        "${limit_args[@]}" > "${LOG_DIR}/worker_$(printf '%03d' "${worker_id}").log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if (( status != 0 )); then
    echo "One or more workers failed; existing artifacts were retained. Inspect ${LOG_DIR}." >&2
    exit "${status}"
fi
if [[ "${SUMMARIZE}" == "1" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize.py" --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
fi
