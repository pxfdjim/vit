#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/pengxf/anaconda3/envs/vit/bin/python}
N_WORKERS=${N_WORKERS:-1}

# Run experiments for multiple datasets across multiple alpha thresholds
echo "Starting VLM4TS experiments with ${N_WORKERS} worker(s)..."
echo "Start time: $(date)"

# List of datasets to process
DATASETS=(
    "realTraffic"
    "realTweets"
    "SMAP"
    "MSL"
    "artificialWithAnomaly"
    "realAdExchange"
    "realAWSCloudwatch"
    "YAHOOA1"
    "YAHOOA2"
    "YAHOOA3"
    "YAHOOA4"
)

# Download all datasets if not already present
echo ""
echo "Checking and downloading datasets if needed..."
"${PYTHON_BIN}" src/preprocessing/download_data.py "${DATASETS[@]}"

echo "Dataset download/check complete."

# Run experiments for each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running $dataset (alpha=0.1, 0.01, 0.001) with VLM4TS..."
    echo "=========================================="

    "${PYTHON_BIN}" src/run_experiment.py "$dataset" --n_workers "$N_WORKERS" --model vlm4ts
    echo "Completed $dataset successfully"
done

echo ""
echo "Aggregating results..."
"${PYTHON_BIN}" src/evaluation/aggregate_results.py

echo ""
echo "=========================================="
echo "All experiments completed."
echo "End time: $(date)"
echo "=========================================="
