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
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import zarr
import dask.array as da

from data_utils import build_continuous_time, load_label
from constants import *
from process_utils import get_window
from outside_target import get_window_index, extract_proportions


def merge_vars(variables, save_dir):
    print(variables)
    for split in ["train", "test"]:
        for v in variables:
            labels = pd.read_pickle(
                os.path.join(save_dir, "temp", f"{split}_{v}_labels.pkl")
            )
            print(labels.shape)

    # need to do an inner merge on DateTime for all variables on labels, but KEEP the indeces
    # then subset the data arrays to match the indeces of the merged labels

    return None


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


def finalize(v, split_dict, save_dir, ptids, debug=False):
    """
    Finalize the extracted data for a given variable, split into train and test.

    Args:
        v (str): Variable to finalize.
        split_dict (dict): Dictionary with keys being splits and values being list of ptids.
        temp_dir_path (str): Path to the temporary directory where intermediate results are stored.
        norm_dir (str): Path to the directory for saving normalization parameters.
    Returns:
        None
    """
    # build separate train and test dask arrays
    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")

    for s, ptids in split_dict.items():
        zarr_all_store = os.path.join(save_dir, "temp", f"{s}_{v}_x.zarr")
        labels_all_store = os.path.join(save_dir, "temp", f"{s}_{v}_labels.pkl")

        # go through ptids and append to cumulative var arrays
        print(f"Finalizing {s}:")
        for i, p in tqdm(enumerate(ptids), total=len(ptids)):
            zarr_pt_store = os.path.join(save_dir, "temp", v, f"{p}_x.zarr")
            labels_pt_store = os.path.join(save_dir, "temp", v, f"{p}_labels.pkl")
            if i == 0:
                z_arr = da.from_zarr(zarr_pt_store)
                base = z_arr
                labels_df = pd.read_pickle(labels_pt_store)
            else:
                base = da.concatenate([base, da.from_zarr(zarr_pt_store)], axis=0)
                labels_df = pd.concat(
                    [labels_df, pd.read_pickle(labels_pt_store)],
                    axis=0,
                ).reset_index(drop=True)

        da.to_zarr(base, url=zarr_all_store, overwrite=True)
        labels_df.to_pickle(labels_all_store)

    if not debug:
        shutil.rmtree(os.path.join(save_dir, "temp", v))

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
    # overwrite save dir if it exists
    if os.path.exists(save_dir):
        print("Overwriting existing save directory.")
        shutil.rmtree(save_dir)

    # make new dirs
    for split in ["train", "test", "temp", "scalers"]:
        os.makedirs(os.path.join(save_dir, split))

    temp_dir = os.path.join(save_dir, "temp")
    norm_dir = os.path.join(save_dir, "scalers")
    split_dict = {"train": [], "test": []}

    # need healthy ptids and dict of which split they fall in
    with h5py.File(args.data_file, "r") as f:
        ptids = f["healthy_ptids"][:].astype(str).tolist()
        # for debugging
        ptids = ptids[:5]
        for p in ptids:
            if f[p].attrs["split"] == "train":
                split_dict["train"].append(p)
            else:
                split_dict["test"].append(p)

    # will do everything and write to file in temp_dir
    for var in args.variables:
        print(f"Extracting for variable {var}:")
        os.makedirs(os.path.join(temp_dir, var))
        func = partial(
            extract_data,
            v=var,
            file_path=args.data_file,
            temp_dir_path=os.path.join(temp_dir, var),
            window_size=args.window_size,
        )
        results = process_map(func, ptids, max_workers=os.cpu_count(), chunksize=1)

        print("Merging into final datasets (with normalization and imputation):")
        finalize(var, split_dict, save_dir, ptids, debug=args.debug)

        print("")

    # check labels
    merge_vars(args.variables, save_dir)

    # preprocess dataset
    # pipeline()

    # take finalized data from temp_dir, concatenate across vars, and move to save_dir

    # delete temp_dir
    # if not args.debug:
    #     shutil.rmtree(temp_dir)
