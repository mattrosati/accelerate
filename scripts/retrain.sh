#!/bin/bash
set -euo pipefail

cd ~/accelerate

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <TRAIN_DIR> <RUN_NAME>"
    exit 1
fi

TRAIN_DIR="$1"
RUN_NAME="$2"

if [[ ! -d "$TRAIN_DIR" ]]; then
    echo "Error: TRAIN_DIR does not exist: $TRAIN_DIR"
    exit 1
fi

SLURM_SCRIPTS=(
    "scripts/train_all.sh"
    # "scripts/bigmem_train_all.sh"
    # "scripts/long_train_all.sh"
)

LOGROOT="logs"
mkdir -p "$LOGROOT"

timestamp=$(date +"%Y-%m-%d_%H-%M")
dirname=$(basename "$TRAIN_DIR")
LOGDIR="$LOGROOT/${RUN_NAME}_${dirname}"
mkdir -p "$LOGDIR"

echo "SLURM Job Launcher"
echo "TRAIN_DIR = $TRAIN_DIR"
echo "Timestamp = $timestamp"
echo "Log Directory = $LOGDIR"
echo "======================================="
echo ""

# Submit each SLURM script

for SCRIPT in "${SLURM_SCRIPTS[@]}"; do
    scriptname=$(basename "$SCRIPT")
    jobname="${scriptname%%_*}"

    echo "Submitting job:"
    echo "  Script: $SCRIPT"
    echo "  Job name: $jobname"
    echo ""

    sbatch \
        --job-name="$jobname" \
        --export=TRAIN_DIR="$TRAIN_DIR",RUN_NAME="$RUN_NAME" \
        --output="$LOGDIR/${jobname}_%A_%a.out" \
        "$SCRIPT"
done

echo "All jobs submitted successfully."