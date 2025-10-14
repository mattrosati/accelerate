#!/bin/bash

# Check if path prefix is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <hours requested for session>"
  exit 1
fi

HOURS=$1
TASK_FILE=${2:-vscode_slurm.sh}

TIME_FORMAT=$(printf "%02d:00:00" "$HOURS")

cd ~

sed -i "s/^#SBATCH -t .*/#SBATCH -t $TIME_FORMAT/" "$TASK_FILE"

sbatch "$TASK_FILE"

salloc -t "$TIME_FORMAT" --mem-per-cpu=12G --cpus-per-task=4 srun --pty bash -c '
    module load miniconda
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate cppopt-dl
    cd ./accelerate

    # Hand control to you without re-reading ~/.bashrc (so PS1 stays intact)
    exec bash -i
'
