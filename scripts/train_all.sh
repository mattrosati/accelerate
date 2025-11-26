#!/bin/bash
#SBATCH --job-name=accelerate-array
#SBATCH --output=logs/train_optuna/log_%A_%a.out
#SBATCH --error=logs/train_optuna/log_%A_%a.out
#SBATCH --array=0-35                      # Update this range to match the number of runs
#SBATCH --partition=day
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G 
#SBATCH --cpus-per-task=30       
#SBATCH --time=1-00:00:00

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

# Select model based on list of args
PARAM_LIST="/home/mr2238/accelerate/scripts/train_all_array.txt"
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_LIST")
TRAIN_DIR="/home/mr2238/project_pi_np442/mr2238/accelerate/data/training/downsample_w_300s_hr_rso2r_rso2l_spo2_abp"
# TRAIN_DIR="/home/mr2238/project_pi_np442/mr2238/accelerate/data/training/smooth_downsample_w_300s_hr_rso2r_rso2l_spo2_abp"
# TRAIN_DIR="/home/mr2238/project_pi_np442/mr2238/accelerate/data/training/smooth_w_300s_hr_rso2r_rso2l_spo2_abp"

echo "Training with params: $PARAMS"
echo "Train dir: $TRAIN_DIR"

python -u src/train.py --train_dir $TRAIN_DIR $PARAMS --run_name "optuna"

conda deactivate
