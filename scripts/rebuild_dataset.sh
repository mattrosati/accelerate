#!/bin/bash
#SBATCH --begin=now
#SBATCH --job-name=acc-build
#SBATCH --output=logs/output_build_%A.out
#SBATCH --error=logs/output_build_%A.out
#SBATCH --partition day                           # Train on day
#SBATCH --requeue
#SBATCH --mem-per-cpu=40G 
#SBATCH --cpus-per-task=20                      
#SBATCH --time=1-00:00:00                         # Time limit hrs:min:sec

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

# python src/dataset_building.py \
#     ~/project_pi_np442/mr2238/accelerate/data/processed/ \
#     ~/project_pi_np442/mr2238/accelerate/data/raw_data/ \
#     ~/project_pi_np442/mr2238/accelerate/data/labels/

# python src/outside_target.py \
#     ~/project_pi_np442/mr2238/accelerate/data/processed/all_data.hdf5 \
#     ~/project_pi_np442/mr2238/accelerate/data/tmp \
#     -m "mean"

# python src/train_test_split.py
set -x

python -u src/data_extract.py -g 2 -o -t separate_pca pca -w 1200 -s robust -m smooth -v rso2r rso2l abp
python -u src/data_extract.py -g 2 -o -t separate_pca pca -w 1200 -s robust -m smooth

python -u src/data_extract.py -g 1 -o -t separate_pca pca -w 1200 -s robust -m smooth -v rso2r rso2l abp
python -u src/data_extract.py -g 1 -o -t separate_pca pca -w 1200 -s robust -m smooth


