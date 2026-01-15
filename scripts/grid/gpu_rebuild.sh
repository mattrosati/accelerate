#!/bin/bash
#SBATCH --job-name=gpu-build
#SBATCH --partition gpu                           # Train on day
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --mem-per-cpu=50G 
#SBATCH --cpus-per-task=5                      
#SBATCH --time=08:00:00                         # Time limit hrs:min:sec

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

set -euo pipefail

python -u src/data_extract.py $PARAMS -t chronos -o --top_dir /home/mr2238/scratch_pi_np442/mr2238/accelerate/total

