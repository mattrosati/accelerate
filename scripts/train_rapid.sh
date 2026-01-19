#!/bin/bash
#SBATCH --array=0-51                  # Update this range to match the number of runs
#SBATCH --partition=day
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=50G
#SBATCH --cpus-per-task=15
#SBATCH --time=1-00:00:00

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate
echo "Rapid iteration train"

# Select model based on list of args
PARAM_LIST="/home/mr2238/accelerate/scripts/train_rapid_array.txt"
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_LIST")
OUTPUT_FILE="$LOGDIR/${jobname}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"

echo "Training with params: $PARAMS"
echo "Train dir: $TRAIN_DIR"

python -u src/train.py -d --train_dir $TRAIN_DIR $PARAMS --run_name $RUN_NAME

# get the log directory name
RAYLOG_DIR=$(sed -n 's/^RAYLOG_DIR=//p' "$OUTPUT_FILE" | tail -n 1)

# Delete the Ray session directory
if [ -n "$RAYLOG_DIR" ] && [ -d "$RAYLOG_DIR" ]; then
    echo "Deleting Ray log directory: $RAYLOG_DIR"
    rm -rf "$RAYLOG_DIR"
else
    echo "Ray log directory not found or empty"
fi

conda deactivate
