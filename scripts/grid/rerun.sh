#!/bin/bash
#SBATCH --job-name=rerun
#SBATCH --partition day                           # Train on day
#SBATCH --requeue
#SBATCH --mem-per-cpu=50G 
#SBATCH --cpus-per-task=10                     
#SBATCH --time=10:00:00                         # Time limit hrs:min:sec
#SBATCH --output=logs/rerun_%A.out

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

set -euo pipefail

# list all datasets in the TOTAL_DIR and run bash command
# Iterate over dataset dirs in TOTAL_DIR and run command for each
# shopt -s nullglob

# for DATASET_DIR in "$TOTAL_DIR"/*; do
#     [[ -d "$DATASET_DIR" ]] || continue

#     echo "Dataset dir: $DATASET_DIR"

#     DATASET_NAME="$(basename "$DATASET_DIR")"
#     LOG_DIR="/home/mr2238/accelerate/logs/rapid_${DATASET_NAME}"

#     [[ -d "$LOG_DIR" ]] || continue

#     find "$LOG_DIR" -type f -regextype posix-extended -regex '.*_(2|5|6|1[6-9]|2[0-1]|25)\.out$' -print -delete

#     bash scripts/rapid_iter_rerun.sh "$DATASET_DIR" rapid
# done

# DATASET_LIST=(
#   "/home/mr2238/scratch_pi_np442/mr2238/accelerate/total/freq1_robust_smooth0.20_downsample1_w_900s_hr_rso2r_rso2l_spo2_abp"
#   "/home/mr2238/scratch_pi_np442/mr2238/accelerate/total/robust_smooth0.46_downsample2_w_1800s_hr_rso2r_rso2l_spo2_abp"
#   "/home/mr2238/scratch_pi_np442/mr2238/accelerate/total/robust_smooth0.46_downsample2_w_1800s_rso2r_rso2l_abp"
# )

# for DATASET_DIR in "${DATASET_LIST[@]}"; do

#     [[ -d "$DATASET_DIR" ]] || {
#         echo "Skipping missing dir: $DATASET_DIR"
#         continue
#     }

#     echo "Dataset dir: $DATASET_DIR"
#     bash scripts/rapid_iter_rerun.sh "$DATASET_DIR" rapid
# done


# list all datasets in the TOTAL_DIR and run bash command
# Iterate over dataset dirs in TOTAL_DIR and run command for each
shopt -s nullglob

for DATASET_DIR in "$TOTAL_DIR"/*; do
    [[ -d "$DATASET_DIR" ]] || continue

    echo "Dataset dir: $DATASET_DIR"

    python -u src/design_feat.py -o --train_dir "$DATASET_DIR"
    python -u src/design_feat.py -o --train_dir "$DATASET_DIR" -w

    bash scripts/rapid_iter_rerun.sh "$DATASET_DIR" 2.0rap
done


