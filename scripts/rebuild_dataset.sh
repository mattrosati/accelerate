#!/bin/bash

module reset
module load miniconda
conda activate cppopt-dl

cd /home/mr2238/accelerate

python src/dataset_building.py \
    ~/project_pi_np442/mr2238/accelerate/data/processed/ \
    ~/project_pi_np442/mr2238/accelerate/data/raw_data/ \
    ~/project_pi_np442/mr2238/accelerate/data/labels/

python src/outside_target.py \
    ~/project_pi_np442/mr2238/accelerate/data/processed/all_data.hdf5 \
    ~/project_pi_np442/mr2238/accelerate/data/tmp \
    -m "mean"

python src/train_test_split.py