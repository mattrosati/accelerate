#!/bin/bash
set -euo pipefail

# Check if path prefix is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <hours requested for session> <gpu or cpu>"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Usage: $0 <hours requested for session> <gpu or cpu>"
  exit 1
fi

HOURS=$1
PARTITION=$2
TASK_FILE=${3:-vscode_slurm.sh}

TIME_FORMAT=$(printf "%02d:00:00" "$HOURS")

cd ~

echo "Starting vscode job and interactive job:"

sed -i "s/^#SBATCH -t .*/#SBATCH -t $TIME_FORMAT/" "$TASK_FILE"

sbatch "$TASK_FILE"


# add temporary section to .bashrc
BASHRC="$HOME/.bashrc"
TEMP_TAG="# === TEMP SECTION ADDED BY SCRIPT ==="

cd accelerate

# Add temporary lines
{
     echo "$TEMP_TAG"
     echo "module load miniconda; conda activate cppopt-dl"
     echo "$TEMP_TAG"
} >> "$BASHRC"

if [[ "$PARTITION" == 'gpu' ]]; then
  salloc -t "$TIME_FORMAT" --mem-per-cpu=8G --cpus-per-task=3 --partition=gpu_devel --gpus=1
else
  salloc -t "$TIME_FORMAT" --mem-per-cpu=8G --cpus-per-task=3
fi

sed -i "/$TEMP_TAG/,/$TEMP_TAG/d" "$BASHRC"
echo "Cleaned up .bashrc."
