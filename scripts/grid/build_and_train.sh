#!/bin/bash
set -euo pipefail

# Check if data mode is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <general mode of data> <run name>"
  exit 1
fi
# Check if run name is provided
if [ -z "$2" ]; then
  echo "Usage: $0 <general mode of data> <run name>"
  exit 1
fi


date;hostname;pwd
cd /home/mr2238/accelerate
echo "Dataset build and rapid iteration train"
MODE=$1
RUN_NAME=$2

LOGROOT="logs"
mkdir -p "$LOGROOT"

timestamp=$(date +"%Y-%m-%d_%H-%M")
LOGDIR="$LOGROOT/$MODE"
mkdir -p "$LOGDIR"


# Select model based on list of args
PARAM_LIST="/home/mr2238/accelerate/scripts/grid/dataset_array_stride.txt"

#sbatch gpu rebuild
SUBMIT_OUT=$(sbatch \
  --array=0-215 \
  --export=ALL,PARAM_LIST="$PARAM_LIST",MODE="$MODE" \
  --output="$LOGDIR/gpu_%A_%a.out" \
  scripts/grid/gpu_rebuild.sh \
)

JOBID=$(echo "$SUBMIT_OUT" | awk '/Submitted batch job/ {print $4}')

[[ -z "$JOBID" ]] && {
  echo "ERROR: Failed to capture jobid"
  echo "$SUBMIT_OUT"
  exit 1
}
echo "Submitted gpu rebuild job: $JOBID"

# sbatch cpu rebuild and train
CPU_LOG="$LOGDIR/cpu_${JOBID}_%a.out"
sbatch \
  --dependency=afterok:$JOBID \
  --array=0-215 \
  --export=ALL,PARAM_LIST="$PARAM_LIST",LOGDIR="$LOGDIR",CPU_LOG="$CPU_LOG",RUN_NAME="$RUN_NAME",MODE="$MODE" \
  --output="$CPU_LOG" \
  scripts/grid/cpu_rebuild.sh

