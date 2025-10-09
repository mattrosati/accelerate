# here I want to calculate:
# - % time outside of target across whole dataset
# - duration of time outside of target for whole dataset, with summary statistics

import os
import sys
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

PERCENT_IN_MIN = 0.5
OUTSIDE_SECONDS = 60

def extract_proportions(windows, labels, percentage=0.5):

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
            in_out[i] = (proportion_in > proportion_out)
    
    return in_out


def get_window(data, index, coords, window_index, window_s, percentage=0.5):
    # compute closest idx for data that matches coords
    seg_start = index['starttime'].iloc[coords["segment"]].to_numpy()
    seg_end = index['endtime'].iloc[coords["segment"]].to_numpy()
    seg_freq = index['frequency'].iloc[coords["segment"]].to_numpy()
    tokens_from_seg_start = ((coords['DateTime'].to_numpy() - seg_start) / (1e6 * (1/seg_freq))).round().astype(np.int64)
    closest_idx = index['startidx'].iloc[coords["segment"]].to_numpy() + tokens_from_seg_start
    assert seg_start.shape == coords["segment"].shape

    # get position of window bounds in absolute time to check if outside segments 
    window_us = window_s * 1e6
    window_index_us = (window_index + 1) * 1e6
    window_start = coords['DateTime'].to_numpy() - float(window_index_us)
    window_end = coords['DateTime'].to_numpy() + (window_us - window_index_us)
    window_length = window_end - window_start
    total_window_token_length = (window_length / (1e6 * (1/seg_freq))).round().astype(np.int64)


    # check whether overlapping with gap
    # shape (n_seg, n_windows)
    overlap_start = np.clip(seg_start - window_start, a_min=0, a_max=None) # pos only if window overlaps with seg start, clips to zero if no overlap
    overlap_end   = np.clip(window_end - seg_end, a_min=0, a_max=None) # pos only if window overlaps with seg end, clips to zero if no overlap

    # overlap length per (segment, window)
    overlap_len = overlap_end + overlap_start

    # fraction of each window overlapped by *any* segment
    frac_overlap = overlap_len / window_length

    # filter out if frac is > percentage
    mask = frac_overlap > percentage

    clean = coords[~mask]
    seg_start, window_start = seg_start[~mask], window_start[~mask]
    seg_end, window_end = seg_end[~mask], window_end[~mask]
    seg_freq = seg_freq[~mask]
    overlap_len = overlap_len[~mask]
    total_window_token_length = total_window_token_length[~mask]
    
    # extract abp windows, add na if in gap
    # window params
    window_start_clipped = np.maximum(seg_start, window_start)
    window_end_clipped = np.minimum(seg_end, window_end)
    window_start_tokens = ((window_start_clipped - seg_start) / (1e6 * (1/seg_freq))).round().astype(np.int64)
    window_end_tokens = ((window_end_clipped - seg_start) / (1e6 * (1/seg_freq))).round().astype(np.int64)

    w_start_idx = index['startidx'].iloc[clean["segment"]].to_numpy() + window_start_tokens
    w_end_idx = index['startidx'].iloc[clean["segment"]].to_numpy() + window_end_tokens

    df = np.concatenate([w_start_idx[:, None], w_end_idx[:, None], overlap_len[:, None], total_window_token_length[:, None]], axis=1).astype(np.int64)

    return df, clean


def get_window_params(mode):
    # unpack mode into parameters to feed extractor
    match mode:
        case "before":
            return OUTSIDE_SECONDS - 1
        case "after":
            return 0
        case "within":
            return OUTSIDE_SECONDS // 2 - 1
    

def extractor(ptid, file_path, window_index, window_s):
    with h5py.File(file_path, "r") as f:
        #load labels[targets] and abp
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
        in_out = np.empty(shape = labels.shape[0])
        start_end = np.empty(shape = (labels.shape[0], 2))

        # convert index to segments start and end times
        abp_index["endtime"] = abp_index['starttime'] + (abp_index['length'] / abp_index['frequency'] * 1e6).astype(np.int64)
        
        # select label timepoints in segments
        seg_start = abp_index['starttime'].to_numpy()[:, None]
        seg_end = abp_index["endtime"].to_numpy()[:, None]
        in_segment = (labels["DateTime"].to_numpy() < seg_end) & (labels["DateTime"].to_numpy() >= seg_start) # (n_seg, data_points)
        mask = np.any(in_segment, axis=0) # (data_points, 1)

        # find segment for each timestamp
        first_idx = np.argmax(in_segment, axis=0)
        
        labels["segment"] = pd.Series(first_idx, index=labels.index)
        labels = labels[mask]

        # select out the windows
        df, labels = get_window(abp, abp_index, labels, window_index, window_s, percentage=PERCENT_IN_MIN)

        # extract window data
        windows = [{"w" : abp[i[0]:i[1]], "overlap_len" : i[2], "total_length" : i[3]} for i in df]

        # extract proportion_in T/F database
        in_out = extract_proportions(windows, labels, percentage=PERCENT_IN_MIN)

        return in_out, df



def main(ptid, file_path, mode, temp_dir_path):

    window_index, window_s = get_window_params(mode), OUTSIDE_SECONDS

    # extract data
    data, windows = extractor(ptid, file_path, window_index, window_s)

    # writes data in defined directory
    temp_ptid_path = os.path.join(temp_dir_path, f"{ptid}.h5")

    with h5py.File(temp_ptid_path, "w") as f:
        f.attrs['idx_unit'] = "token"
        f.attrs['len_unit'] = "token"
        f.create_dataset("in_out", data=data.astype(np.int8), dtype=np.int8)
        f.create_dataset("window_idx", data=windows[:, :2].astype(np.int64), dtype=np.int64)
        f.create_dataset("overlap_len", data=windows[:, 2].astype(np.int64), dtype=np.int64)
        f.create_dataset("total_len", data=windows[:, 3].astype(np.int64), dtype=np.int64)

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
        choices=["before", "after", "within"],
        required=True,
    )

    args = parser.parse_args()

    np.random.seed(420)

    # make dir if it doesn't exist
    os.makedirs(args.temp_dir, exist_ok=True)

    # load only to get length
    with h5py.File(args.data_file, "r") as f:
        ptids = list(f.keys())

    # will do everything and write to file in temp_dir
    func = partial(main, 
                file_path = args.data_file,
                mode = args.mode,
                temp_dir_path = args.temp_dir
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




#     futures = [ex.submit(process_feature, feat) for feat in features]
# for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#     feature, result = fut.result()
