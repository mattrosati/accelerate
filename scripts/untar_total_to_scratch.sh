#!/bin/bash
#SBATCH --job-name=untar_total
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=untar_total_%j.log

# Transfer total/ from project to scratch, unarchiving .zarr.tar files
# and rechunking them to reduce file count.
#
# Source:      ~/project_pi_np442/mr2238/accelerate/data/total/
# Destination: ~/scratch_pi_np442/mr2238/accelerate/total/

SRC="$HOME/project_pi_np442/mr2238/accelerate/data/total"
DST="$HOME/scratch_pi_np442/mr2238/accelerate/total"

module load miniconda
conda activate cppopt-dl

mkdir -p "$DST"

n_configs=$(ls -d "$SRC"/*/ | wc -l)
echo "Processing $n_configs config directories from $SRC -> $DST"
echo "Started at $(date)"

count=0
for config_dir in "$SRC"/*/; do
    config_name=$(basename "$config_dir")
    count=$((count + 1))
    echo "[$count/$n_configs] $config_name"

    # Mirror the directory structure (permanent/train, permanent/test, etc.)
    find "$config_dir" -type d | while read -r src_subdir; do
        rel_path="${src_subdir#$SRC/}"
        mkdir -p "$DST/$rel_path"
    done

    # Copy non-tar files (labels.pkl, scalers, model files, etc.)
    find "$config_dir" -type f ! -name "*.zarr.tar" | while read -r src_file; do
        rel_path="${src_file#$SRC/}"
        dst_file="$DST/$rel_path"
        if [ ! -f "$dst_file" ]; then
            cp "$src_file" "$dst_file"
        fi
    done

    # Untar .zarr.tar files, then rechunk via zarr to reduce file count
    find "$config_dir" -name "*.zarr.tar" | while read -r tar_file; do
        rel_dir="$(dirname "${tar_file#$SRC/}")"
        zarr_name=$(basename "$tar_file" .tar)
        dst_zarr="$DST/$rel_dir/$zarr_name"

        if [ -d "$dst_zarr" ]; then
            echo "  Skipping $rel_dir/$zarr_name (already exists)"
            continue
        fi

        echo "  Extracting $rel_dir/$zarr_name"
        tar xf "$tar_file" -C "$DST/$rel_dir/"

        # Rechunk: load and resave with zarr auto-chunking to reduce file count
        echo "  Rechunking $rel_dir/$zarr_name"
        python -c "
import zarr, numpy as np, shutil, sys
path = '$dst_zarr'
try:
    arr = zarr.open(path, mode='r')
    data = arr[:]
    shutil.rmtree(path)
    zarr.save(path, data)
    new_arr = zarr.open(path, mode='r')
    print(f'    {path}: {arr.chunks} -> {new_arr.chunks} ({new_arr.nchunks} chunks)')
except Exception as e:
    print(f'    ERROR rechunking {path}: {e}', file=sys.stderr)
"
    done
done

echo "Finished at $(date)"
echo "Done."
