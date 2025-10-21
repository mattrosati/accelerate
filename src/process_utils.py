import os
import sys

import h5py
import numpy as np
import pandas as pd


def get_window(data, index, coords, window_index, window_s, percentage=0.5):
    # will obtain window start and end idx, plus total length and overlap length (in tokens)
    # inputs: data (dataset with data of interest), index (contains segment information),
    # coords (contains labels and their coordinates in time), window_index (token index of label value in window),
    # window_s (duration of window in seconds), percentage (max percentage window allowed to overlap with gap)
    # compute closest idx for data that matches coords
    seg_start = index["starttime"].iloc[coords["segment"]].to_numpy()
    seg_end = index["endtime"].iloc[coords["segment"]].to_numpy()
    seg_freq = index["frequency"].iloc[coords["segment"]].to_numpy()
    tokens_from_seg_start = (
        ((coords["DateTime"].to_numpy() - seg_start) / (1e6 * (1 / seg_freq)))
        .round()
        .astype(np.int64)
    )
    closest_idx = (
        index["startidx"].iloc[coords["segment"]].to_numpy() + tokens_from_seg_start
    )
    assert seg_start.shape == coords["segment"].shape

    # get position of window bounds in absolute time to check if outside segments
    window_us = window_s * 1e6
    window_index_us = (
        window_index
    ) * 1e6  # I don't think I need a plus one here, if I am indexing 4 seconds of data, i want the 3 index or the last number at ms 3e3
    window_start = coords["DateTime"].to_numpy() - float(window_index_us)
    window_end = window_start + window_us
    window_length = window_end - window_start
    total_window_token_length = (
        (window_length / (1e6 * (1 / seg_freq))).round().astype(np.int64)
    )

    # check whether overlapping with gap
    # shape (n_seg, n_windows)
    overlap_start = np.clip(
        seg_start - window_start, a_min=0, a_max=None
    )  # pos only if window overlaps with seg start, clips to zero if no overlap
    overlap_end = np.clip(
        window_end - seg_end, a_min=0, a_max=None
    )  # pos only if window overlaps with seg end, clips to zero if no overlap

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

    # extract abp windows
    # window params
    window_start_clipped = np.maximum(seg_start, window_start)
    window_end_clipped = np.minimum(seg_end, window_end)
    window_start_tokens = (
        ((window_start_clipped - seg_start) / (1e6 * (1 / seg_freq)))
        .round()
        .astype(np.int64)
    )
    window_end_tokens = (
        ((window_end_clipped - seg_start) / (1e6 * (1 / seg_freq)))
        .round()
        .astype(np.int64)
    )

    w_start_idx = (
        index["startidx"].iloc[clean["segment"]].to_numpy() + window_start_tokens
    )
    w_end_idx = index["startidx"].iloc[clean["segment"]].to_numpy() + window_end_tokens

    df = np.concatenate(
        [
            w_start_idx[:, None],
            w_end_idx[:, None],
            overlap_len[:, None],
            total_window_token_length[:, None],
        ],
        axis=1,
    ).astype(np.int64)

    return df, clean


def make_pad(data_file, window_df):
    # gets window df, returns
    return


# TODO: needs a window combiner for when I will ask it to do more than one var
