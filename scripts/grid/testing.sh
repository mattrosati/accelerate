#!/bin/bin

TOTAL_DIR="/home/mr2238/scratch_pi_np442/mr2238/accelerate/total"

for DATASET_DIR in "$TOTAL_DIR"/*; do
    [[ -d "$DATASET_DIR" ]] || continue

    DATASET_NAME="$(basename "$DATASET_DIR")"
    LOG_DIR="/home/mr2238/accelerate/logs/rapid_${DATASET_NAME}"

    [[ -d "$LOG_DIR" ]] || echo "No log_dir found at $DATASET_DIR"
done