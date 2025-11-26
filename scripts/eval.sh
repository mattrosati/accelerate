#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/eval/down_log_%A.out
#SBATCH --error=logs/eval/down_log_%A.out
#SBATCH --partition=day
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G 
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00:00

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

python -u src/eval.py --train_dir ~/project_pi_np442/mr2238/accelerate/data/training/downsample_w_300s_hr_rso2r_rso2l_spo2_abp/ --run_name "train_val_analysis"
python -u src/eval.py -s --train_dir ~/project_pi_np442/mr2238/accelerate/data/training/downsample_w_300s_hr_rso2r_rso2l_spo2_abp/ --run_name "train_val_analysis"

conda deactivate
