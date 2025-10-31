#!/bin/bash
#SBATCH --job-name=accelerate-umap
#SBATCH --output=logs/output_umap_%A_%a.out
#SBATCH --error=logs/output_umap_%A_%a.out
#SBATCH --array=0-1   # <-- Two jobs: task IDs 0 and 1
#SBATCH --partition day                           # Train on day
#SBATCH --requeue
#SBATCH --mem-per-cpu=20G 
#SBATCH --cpus-per-task=4                      
#SBATCH --time=1-00:00:00                         # Time limit hrs:min:sec

date;hostname;pwd
echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    echo "Delaying start of job 1 by 20 minutes..."
    sleep 20m
fi

if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    echo "Calculating umaps on raw windows."
    python -u src/umaps.py --do_big
elif [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    echo "Calculating umaps on PCA of windows."
    python -u src/umaps.py
fi

conda deactivate


