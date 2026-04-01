#!/bin/bash
#SBATCH --partition=day
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

PARAM_LIST="scripts/loo_experiments.txt"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_LIST")

echo "LOO experiment: $LINE"

python -u src/loo.py $LINE

conda deactivate
