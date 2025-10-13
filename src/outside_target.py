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
from constants import TARGETS, FEATURES
from process_utils import get_window

PERCENT_IN_MIN = 0.5
WINDOW_SECONDS = 60

# TODO: refactor to be tas kagnostic
# TODO: refactor to add padding, then can do UMAPs!!!

def get_window_index(mode):
    # according to mode, get index of time t of labels for the specific duration of the window
    match mode:
        case "before":
            return WINDOW_SECONDS - 1
        case "after":
            return 0
        case "within":
            return WINDOW_SECONDS // 2 - 1
        case "mean":
            return WINDOW_SECONDS - 1


def extract_proportions(windows, labels, percentage=0.5, strategy="count"):
    if strategy == "count":
        return extract_proportions_count(windows, labels, percentage)
    elif strategy == "mean":
        return extract_proportions_mean(windows, labels)

def extract_proportions_mean(windows, labels):
    in_out = np.empty(shape=len(windows), dtype=bool)
    for i, w in enumerate(windows):
        lower_limit = labels["LLA_Yale_affected_beta"].iloc[i]
        upper_limit = labels["ULA_Yale_affected_beta"].iloc[i]

        w_vector = w["w"]
        if w["total_length"].sum() == 0:
            print("Record with no windows, unclear why")

        # find average of window
        w_mean = np.mean(w_vector)
        in_out[i] = (w_mean > lower_limit) and (w_mean < upper_limit)

    return in_out

def extract_proportions_count(windows, labels, percentage=0.5):
    in_out = np.empty(shape=len(windows), dtype=bool)
    for i, w in enumerate(windows):
        lower_limit = labels["LLA_Yale_affected_beta"].iloc[i]
        upper_limit = labels["ULA_Yale_affected_beta"].iloc[i]

        w_vector = w["w"]

        # note: all lengths are in tokens

        # dropping nas from window
        valid = np.isfinite(w_vector)
        w_vector = w_vector[valid]
        proportion_na = (~valid).sum() / w["total_length"]

        abp_status = (w_vector >= lower_limit) & (w_vector <= upper_limit)
        proportion_in = abp_status.sum() / w["total_length"]
        proportion_out = 1 - proportion_in

        proportion_gap = w["overlap_len"] / w["total_length"]

        if (proportion_in - proportion_out) < proportion_na + proportion_gap:
            in_out[i] = np.nan
            # write na
        else:
            in_out[i] = proportion_in > proportion_out

    return in_out


def extractor(ptid, file_path, window_index, window_s, strategy="count"):
    # if finding the mean, do not allow any windows that go over gaps
    percentage = 0.0 if strategy == "mean" else PERCENT_IN_MIN

    with h5py.File(file_path, "r") as f:
        # load labels[targets] and abp
        labels = pd.DataFrame(f[f"{ptid}/labels"][...][TARGETS + ["DateTime"]])
        abp = f[f"{ptid}/raw/waves/abp"][...]
        abp_index = pd.DataFrame(f[f"{ptid}/raw/waves/abp"].attrs["index"])

        # replace invalid vals and filter out
        invalid_val = f.attrs["invalid_val"]
        labels.replace(invalid_val, np.nan, inplace=True)
        labels.dropna(inplace=True)

        abp[abp == invalid_val] = np.nan
        # keep nas in, need to discard if too many in window
        # valid = np.isfinite(abp)
        # abp = abp[valid]

        # build empty dataset same rows as labels
        data = {}
        in_out = np.empty(shape=labels.shape[0])
        start_end = np.empty(shape=(labels.shape[0], 2))

        # convert index to segments start and end times
        abp_index["endtime"] = abp_index["starttime"] + (
            abp_index["length"] / abp_index["frequency"] * 1e6
        ).astype(np.int64)

        # select label timepoints in segments
        seg_start = abp_index["starttime"].to_numpy()[:, None]
        seg_end = abp_index["endtime"].to_numpy()[:, None]
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
                abp, abp_index, labels, window_index, window_s, percentage=percentage
            )

            # extract window data
            windows = [
                {"w": abp[i[0] : i[1]], "overlap_len": i[2], "total_length": i[3]}
                for i in df
            ]

            # extract proportion_in T/F database
            in_out = extract_proportions(windows, labels, percentage=percentage, strategy=strategy)

            df = np.concatenate([ df, np.array(labels["DateTime"])[:, np.newaxis] ], axis=1)

            return in_out, df

        else:
            return None, None


def main(ptid, file_path, mode, temp_dir_path):

    window_index, window_s = get_window_index(mode), WINDOW_SECONDS
    strategy = "mean" if mode == "mean" else "count"

    # extract data
    data, windows = extractor(ptid, file_path, window_index, window_s, strategy)

    # writes data in defined directory
    temp_ptid_path = os.path.join(temp_dir_path, f"{ptid}.h5")

    with h5py.File(temp_ptid_path, "w") as f:
        f.attrs["no_label_overlap"] = (data is None and windows is None)
        if f.attrs["no_label_overlap"]:
            print(ptid)
            return None
        f.attrs["idx_unit"] = "token"
        f.attrs["len_unit"] = "token"
        f.create_dataset("in_out", data=data.astype(np.int8), dtype=np.int8)
        f.create_dataset(
            "window_idx", data=windows[:, :2].astype(np.int64), dtype=np.int64
        )
        f.create_dataset(
            "overlap_len", data=windows[:, 2].astype(np.int64), dtype=np.int64
        )
        f.create_dataset(
            "total_len", data=windows[:, 3].astype(np.int64), dtype=np.int64
        )
        f.create_dataset(
            "label_timestamp", data=windows[:, 4].astype(np.int64), dtype=np.int64
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
        ptids = list(f.keys())

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
