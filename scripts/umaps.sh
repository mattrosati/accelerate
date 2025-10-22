#!/bin/bash
#SBATCH --job-name=accelerate-umap
#SBATCH --output=logs/umap_%J.log
#SBATCH --partition day                           # Train on day
#SBATCH --requeue
#SBATCH --mem-per-cpu=20G 
#SBATCH --cpus-per-task=4                      
#SBATCH --time=1-00:00:00                         # Time limit hrs:min:sec

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

# Do umaps
echo "Calculating all umaps for plots and etc."
python src/umaps.py

conda deactivate