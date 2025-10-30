# here I want to calculate:
# - % time outside of target across whole dataset
# - duration of time outside of target for whole dataset, with summary statistics

import os
import sys
import shutil
from argparse import ArgumentParser

import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm.contrib.concurrent import process_map

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from filelock import FileLock

from data_utils import build_continuous_time, load_label
from constants import *
from process_utils import *


def main(ptid, file_path, mode, temp_dir_path):

    window_index, window_s = get_window_index(mode), WINDOW_SECONDS
    strategy = "mean" if mode == "mean" else "count"
    percentage = 0.0 if strategy == "mean" else PERCENT_IN_MIN

    # extract data
    data, _ = get_windows_var('abp', ptid, file_path, window_index, window_s, strategy=strategy, percentage=percentage)

    # writes data in defined directory
    temp_ptid_path = os.path.join(temp_dir_path, f"{ptid}.h5")

    with h5py.File(temp_ptid_path, "w") as f:
        f.attrs["no_label_overlap"] = data is None
        if f.attrs["no_label_overlap"]:
            print("Labels do not overlap with data for:", ptid)
            return None
        in_bool = data["in?"]
        f.attrs["idx_unit"] = "token"
        f.attrs["len_unit"] = "token"
        f.create_dataset("in_out", data=in_bool.astype(np.int8), dtype=np.int8)
        f.create_dataset(
            "window_idx", data=(data[["startidx", "endidx"]].to_numpy()).astype(np.int64), dtype=np.int64
        )
        f.create_dataset(
            "overlap_len", data=(data["overlap_len"].to_numpy()).astype(np.int64), dtype=np.int64
        )
        f.create_dataset(
            "total_len", data=(data["tot_len"].to_numpy()).astype(np.int64), dtype=np.int64
        )
        f.create_dataset(
            "label_timestamp", data=(data["datetime"].to_numpy()).astype(np.int64), dtype=np.int64
        )

    return None


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract information about status within limits for each patient."
    )

    parser.add_argument("data_file", help="Path to combined dataset")
    parser.add_argument(
        "temp_dir",
        help="Path to temporary directory to store processed data to parallelize",
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="Specify way to calculate status in autoregulation",
        type=str,
        choices=["before", "after", "within", "mean"],
        required=True,
    )
    parser.add_argument(
        "-e",
        "--exclusive",
        help="Deletes processed data that uses strategies that are not the input strategy.",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Does not delete temporary dir for debugging purposes.",
        action="store_true",
    )

    args = parser.parse_args()

    np.random.seed(420)

    # make dir if it doesn't exist
    os.makedirs(args.temp_dir, exist_ok=True)

    # load only to get length
    with h5py.File(args.data_file, "r") as f:
        ptids = [pt for pt in f.keys() if isinstance(f[pt], h5py.Group)]

    # will do everything and write to file in temp_dir
    func = partial(
        main, file_path=args.data_file, mode=args.mode, temp_dir_path=args.temp_dir
    )
    results = process_map(func, ptids, max_workers=os.cpu_count(), chunksize=1)

    # loop through temp_dir and add to main dataset
    with h5py.File(args.data_file, "r+") as f:
        for temp_file in os.listdir(args.temp_dir):
            p = os.path.basename(temp_file).split(".")[0]
            temp_path = os.path.join(args.temp_dir, temp_file)
            with h5py.File(temp_path, "r") as tmp:
                target_name = f"in_out_{args.mode}"
                dest_group = f[f"{p}/processed"]
                if target_name in dest_group:
                    del dest_group[target_name]
                tmp.copy(source=tmp, dest=dest_group, name=target_name)

                # delete all other processings if we specify
                if args.exclusive:
                    for k in dest_group.keys():
                        if "in_out_" in k and k != target_name:
                            del dest_group[k]

    # delete temp_dir
    if not args.debug:
        shutil.rmtree(args.temp_dir)


#     futures = [ex.submit(process_feature, feat) for feat in features]
# for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#     feature, result = fut.result()
