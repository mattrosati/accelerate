#!/bin/bash
#SBATCH --job-name=rerun
#SBATCH --partition day                           # Train on day
#SBATCH --requeue
#SBATCH --mem-per-cpu=50G 
#SBATCH --cpus-per-task=10                     
#SBATCH --time=10:00:00                         # Time limit hrs:min:sec
#SBATCH --output=logs/rerun/%A_%a.out

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

# # Check if data mode is provided
# if [ -z "$1" ]; then
#   echo "Usage: $0 <parent data dir> <run name>"
#   exit 1
# fi
# # Check if run name is provided
# if [ -z "$2" ]; then
#   echo "Usage: $0 <parent data dir> <run name>"
#   exit 1
# fi

# TOTAL_DIR="$1"
# RUN_NAME="$2"
DATASET_DIRS=( "$TOTAL_DIR"/* )

N=${#DATASET_DIRS[@]}
if [[ "$N" -eq 0 ]]; then
  echo "No datasets found under TOTAL_DIR=$TOTAL_DIR"
  exit 1
fi

# Guard against out-of-range task IDs
TASK_ID=${SLURM_ARRAY_TASK_ID}
if (( TASK_ID < 0 || TASK_ID >= N )); then
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID out of range (0..$((N-1)))"
  echo "Found N=$N datasets under TOTAL_DIR=$TOTAL_DIR"
  exit 2
fi

DATASET_DIR="${DATASET_DIRS[$TASK_ID]}"
echo "Task ${TASK_ID}/${N}: DATASET_DIR=$DATASET_DIR"

# python -u src/design_feat.py -o --train_dir "$DATASET_DIR"
# python -u src/design_feat.py -o --train_dir "$DATASET_DIR" -w

bash scripts/rapid_iter.sh "$DATASET_DIR" "$RUN_NAME"

