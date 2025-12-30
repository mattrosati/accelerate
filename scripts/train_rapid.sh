#!/bin/bash
#SBATCH --array=0-5                    # Update this range to match the number of runs
#SBATCH --partition=day
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00:00

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate
echo "Rapid iteration train"

# Select model based on list of args
PARAM_LIST="/home/mr2238/accelerate/scripts/train_rapid_array.txt"
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_LIST")

echo "Training with params: $PARAMS"
echo "Train dir: $TRAIN_DIR"

# if train_dir contains var and params contains raw, skip python run
if [[ "$TRAIN_DIR" == *var* ]] && [[ "$PARAMS" == *raw* ]]; then
  echo "Skipping: TRAIN_DIR contains 'var' and PARAMS contains 'raw'."
  conda deactivate
  exit 0
fi

# if train_dir contains freq and params contains raw, skip python run
if [[ "$TRAIN_DIR" == *freq* ]] && [[ "$PARAMS" == *design* ]]; then
  echo "Skipping: TRAIN_DIR contains 'freq' and PARAMS contains 'design'."
  conda deactivate
  exit 0
fi

python -u src/train.py --train_dir $TRAIN_DIR $PARAMS --run_name $RUN_NAME

conda deactivate
