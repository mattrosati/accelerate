#!/bin/bash
#SBATCH --job-name=cpu-build
#SBATCH --partition day                           # Train on day
#SBATCH --requeue
#SBATCH --mem-per-cpu=50G 
#SBATCH --cpus-per-task=10                      
#SBATCH --time=1-00:00:00                         # Time limit hrs:min:sec

date;hostname;pwd

module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

set -euo pipefail

python -u src/data_extract.py $PARAMS -t pca separate_pca -o --save_dir /home/mr2238/scratch_pi_np442/mr2238/accelerate/total

# get the dataset directory name
DATASET_DIR=$(sed -n 's/^DATASET_NAME=//p' "$CPU_LOG" | tail -n 1)
echo "Dataset dir: $DATASET_DIR"

if [[ "$DATASET_DIR" == *w_60s* ]] || [[ "$DATASET_DIR" == *freq* ]]; then
    echo "Skipping design_feat.py because window too small (i.e., 60s) or frequency not 60."
else
    python -u src/design_feat.py -o --train_dir "$DATASET_DIR"
fi

bash scripts/rapid_iter.sh "$DATASET_DIR" rapid


