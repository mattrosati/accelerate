#!/bin/bash
#SBATCH --array=0-100                  # Update this range to match the number of runs, 101-200, 201-400, 401-600, 600-862 (0-100 done)
#SBATCH --partition=day
#SBATCH --output="logs/launcher/total_%a_%A.out"
#SBATCH --job-name=grid
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

set -euo pipefail

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate
echo "Dataset build and rapid iteration train"

LOGROOT="logs"
mkdir -p "$LOGROOT"

timestamp=$(date +"%Y-%m-%d_%H-%M")
LOGDIR="$LOGROOT/total"
mkdir -p "$LOGDIR"


# Select model based on list of args
PARAM_LIST="/home/mr2238/accelerate/scripts/grid/dataset_array.txt"
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_LIST")

echo "Building with params: $PARAMS"

#sbatch gpu rebuild
SUBMIT_OUT=$(sbatch \
  --export=ALL,PARAMS="$PARAMS" \
  --output="$LOGDIR/gpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
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
CPU_LOG="$LOGDIR/cpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
sbatch \
  --dependency=afterok:$JOBID \
  --export=ALL,PARAMS="$PARAMS",LOGDIR="$LOGDIR",CPU_LOG="$CPU_LOG" \
  --output="$CPU_LOG" \
  scripts/grid/cpu_rebuild.sh

conda deactivate
