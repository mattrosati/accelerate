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
import zarr

from data_utils import build_continuous_time, load_label
from constants import *
from process_utils import get_window
from outside_target import get_window_index, extract_proportions


def get_windows_var(v, ptid, file_path, window_index, window_s, strategy, percentage):
    with h5py.File(file_path, "r") as f:
        # load labels[targets] and var timeseries
        labels = pd.DataFrame(f[f"{ptid}/labels"][...][TARGETS + ["DateTime"]])
        if v in f[f"{ptid}/raw/numerics"].keys():
            v_long = f"numerics/{v}"
        else:
            v_long = f"waves/{v}"
        ts = f[f"{ptid}/raw/{v_long}"][...]
        ts_index = pd.DataFrame(f[f"{ptid}/raw/{v_long}"].attrs["index"])

        # replace invalid vals and filter out
        invalid_val = f.attrs["invalid_val"]
        labels.replace(invalid_val, np.nan, inplace=True)
        labels.dropna(inplace=True)

        ts[ts == invalid_val] = np.nan
        # keep nas in, need to discard if too many in window

        # build empty dataset same rows as labels
        data = {}
        in_out = np.empty(shape=labels.shape[0])
        start_end = np.empty(shape=(labels.shape[0], 2))

        # convert index to segments start and end times
        ts_index["endtime"] = ts_index["starttime"] + (
            ts_index["length"] / ts_index["frequency"] * 1e6
        ).astype(np.int64)

        # select label timepoints in segments
        seg_start = ts_index["starttime"].to_numpy()[:, None]
        seg_end = ts_index["endtime"].to_numpy()[:, None]
        in_segment = (labels["DateTime"].to_numpy() < seg_end) & (
            labels["DateTime"].to_numpy() >= seg_start
        )  # (n_seg, data_points)
        mask = np.any(in_segment, axis=0)  # (data_points, 1)

        # find segment for each timestamp
        first_idx = np.argmax(in_segment, axis=0)

        labels["segment"] = pd.Series(first_idx, index=labels.index)
        labels = labels[mask]

        if labels.shape[0] != 0:
            # select out the windows
            df, labels = get_window(
                ts, ts_index, labels, window_index, window_s, percentage=percentage
            )

            # extract window data
            windows = [
                {"w": ts[i[0] : i[1]], "overlap_len": i[2], "total_length": i[3]}
                for i in df
            ]

            w_vectors = np.stack([k["w"] for k in windows], axis=0)

            # extract proportion_in T/F database
            in_out = extract_proportions(
                windows, labels, percentage=percentage, strategy=strategy
            )

            df = pd.DataFrame(
                df, columns=["startidx", "endidx", "overlap_len", "tot_len"]
            )
            df["datetime"] = np.array(labels["DateTime"])
            df["in?"] = in_out

            return df, w_vectors

        else:
            return None, None


def extract_data(ptid, v, file_path, temp_dir_path, window_size, mode="mean"):
    """
    Extract data for a given patient ID and variable from the HDF5 file,
    process it into windows, and save the results to a temporary directory.

    Args:
        ptids (str): Patient ID.
        v (str): Variable to extract.
        file_path (str): Path to the HDF5 file.
        temp_dir_path (str): Path to the temporary directory for saving results.
        window_size (int): Size of the window in seconds.
        mode (str): Mode for window extraction ('before', 'after', 'within', 'mean').
    Returns:
        None
    """

    window_index, window_s = (
        get_window_index(mode, window_seconds=window_size),
        window_size,
    )
    strategy = "mean" if mode == "mean" else "count"
    percentage = 0.0 if strategy == "mean" else PERCENT_IN_MIN

    # extract windows for this patient and variable
    in_out, windows = get_windows_var(
        v, ptid, file_path, window_index, window_s, strategy, percentage
    )

    # save to a temp file as a zarr array
    in_out.to_pickle(os.path.join(temp_dir_path, f"{ptid}_labels.pkl"))
    zarr.save(os.path.join(temp_dir_path, f"{ptid}_x.zarr"), windows)

    return None


def finalize(ptids, v, temp_dir_path, norm_dir):
    """
    Finalize the extracted data for a given patient ID and variable by normalizing
    and imputing missing values, then saving the final dataset.

    Args:
        ptids (str): Patient ID.
        v (str): Variable to finalize.
        temp_dir_path (str): Path to the temporary directory where intermediate results are stored.
        norm_dir (str): Path to the directory for saving normalization parameters.
    Returns:
        None
    """

    # TODO: needs to be made

    return None


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--data_file",
        help="Path to processed data HDF5 file.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/all_data.hdf5",
    )
    parser.add_argument(
        "--save_dir",
        help="Directory to save extracted data.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/training",
    )
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=60 * 5,
        help="Window size in seconds to extract values, default is 5 minutes.",
    )
    parser.add_argument(
        "--variables",
        "-v",
        nargs="+",
        default=FEATURES,
        help="List of variables to include in model data. Default is all features.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Does not delete temporary dir for debugging purposes.",
        action="store_true",
    )

    args = parser.parse_args()
    np.random.seed(420)
    dataset_name = f"w_{args.window_size}s_{'_'.join(args.variables)}"

    print(
        f"Dataset creation with window size {args.window_size}s for variables: {args.variables}."
    )

    # make test, train and temp directories, prepare for saving
    save_dir = os.path.join(args.save_dir, dataset_name)
    for split in ["train", "test", "temp", "scalers"]:
        os.makedirs(os.path.join(save_dir, split), exist_ok=True)

    temp_dir = os.path.join(save_dir, "temp")
    norm_dir = os.path.join(save_dir, "scalers")
    # need healthy ptids
    with h5py.File(args.data_file, "r") as f:
        ptids = f["healthy_ptids"][:].astype(str).tolist()

    # will do everything and write to file in temp_dir
    for var in args.variables:
        print(f"Extracting for variable {var}:")
        func = partial(
            extract_data,
            v=var,
            file_path=args.data_file,
            temp_dir_path=temp_dir,
            window_size=args.window_size,
        )
        results = process_map(func, ptids, max_workers=os.cpu_count(), chunksize=1)

        print("Merging into final datasets (with normalization and imputation):")
        func = partial(
            finalize,
            v=var,
            temp_dir_path=temp_dir,
            norm_dir=norm_dir,
        )
        results = process_map(func, ptids, max_workers=os.cpu_count(), chunksize=1)

        print("")

    # take finalized data from temp_dir, merge, and move to save_dir

    # delete temp_dir
    if not args.debug:
        shutil.rmtree(temp_dir)
