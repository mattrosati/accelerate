#!/bin/bash
#SBATCH --partition=day
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=50G
#SBATCH --cpus-per-task=15
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/mr2238/accelerate/logs/loo/%j.out

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

echo "LOO experiment"

python -u src/loo.py --data_mode balanced --run_name hyperpar --output_dir ~/project_pi_np442/mr2238/accelerate/lopo --base_dir /home/mr2238/scratch_pi_np442/mr2238/accelerate

conda deactivate
