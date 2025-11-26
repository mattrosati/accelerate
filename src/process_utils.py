import os
import sys

import h5py
import numpy as np
import pandas as pd

from constants import *


def get_window_index(mode, window_seconds=WINDOW_SECONDS):
    # according to mode, get index of time t of labels for the specific duration of the window
    match mode:
        case "before":
            return window_seconds - 1
        case "after":
            return 0
        case "within":
            return window_seconds // 2 - 1
        case "mean":
            return window_seconds - 1
        case "smooth":
            return window_seconds - 1


def extract_proportions(windows, labels, percentage=0.5, strategy="count", ref=None):
    if strategy == "count":
        return extract_proportions_count(windows, labels, percentage)
    elif strategy == "mean":
        return extract_proportions_mean(windows, labels)
    elif strategy == "smooth":
        return extract_proportions_smooth(windows, labels, percentage, ref)


def extract_proportions_smooth(windows, labels, percentage, ref):
    in_out = np.empty(shape=len(windows), dtype=bool)
    for i, w in enumerate(windows):       
        start = labels["start_idx"].iloc[ref[i]]
        end = labels["end_idx"].iloc[ref[i]]
        lower_limits = labels["LLA_Yale_affected_beta"].loc[start:end]
        upper_limits = labels["ULA_Yale_affected_beta"].loc[start:end]

        w_vector = w["w"]
        
        if w["total_length"].sum() == 0:
            print("Record with no windows, unclear why")

        # split in minute by minute
        intervals = len(lower_limits)
        split_window = w_vector.reshape((intervals, -1), order='C')

        # find average of window
        if (np.isnan(split_window)).all(axis=1).any():
            # these entries will get removed once we filter for windows that have a lot of nans
            in_out[i] = np.nan
            continue

        mean_arr = np.nanmean(split_window, axis=1)

        # find fraction of time within limits
        out = ((mean_arr < lower_limits.values) | (mean_arr > upper_limits.values))
        frac_out = out.mean()

        # set according to percentage
        if frac_out >= SMOOTH_FRAC:
            print(np.concatenate((mean_arr[:, None], lower_limits.values[:, None], upper_limits.values[:, None]), axis=1))
        in_out[i] = frac_out < SMOOTH_FRAC

    return in_out


def extract_proportions_mean(windows, labels):
    in_out = np.empty(shape=len(windows), dtype=bool)
    for i, w in enumerate(windows):
        lower_limit = labels["LLA_Yale_affected_beta"].iloc[i]
        upper_limit = labels["ULA_Yale_affected_beta"].iloc[i]

        w_vector = w["w"]
        if w["total_length"].sum() == 0:
            print("Record with no windows, unclear why")

        # find average of window
        if np.isnan(w_vector).all():
            # these entries will get removed once we filter for windows that have a lot of nans
            in_out[i] = np.nan
            continue
        w_mean = np.nanmean(w_vector)
        # if np.isnan(w_mean):
        #     print("Window has nans")
        in_out[i] = (w_mean >= lower_limit) and (w_mean <= upper_limit)

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


def impute(window, strategy="lin_interpolate"):
    # imputes missing values given a window according to the specified strategy
    if PERCENT_NA_MAX == 1:
        return window
    if np.isnan(window).sum() / len(window) > PERCENT_NA_MAX:
        # print("WARNING: large amount of Nas in window.")
        return None
    if strategy == "lin_interpolate":
        x_coords = np.arange(len(window))
        w_vals = window[~np.isnan(window)]
        if len(w_vals) == 0:
            print("Window completely NaN, cannot impute.")
            return None
        window = np.interp(x=x_coords, xp=x_coords[~np.isnan(window)], fp=w_vals)
    else:
        pass  # implement other strategies as needed
    return window


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

    # need idx of start of window for the labels to extract all labels in window later
    clean = clean.copy()
    clean["end_idx"] = clean.index
    clean["start_idx"] = clean["end_idx"] - (window_s // 60) + 1

    return df, clean


def make_pad(data_file, window_df):
    # gets window df, returns
    return


# TODO: needs a window combiner for when I will ask it to do more than one var


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

        # replace with nan if value is 0 or less (incompatible with life)
        # replace invalid values with nan
        ts[ts == invalid_val] = np.nan
        ts[ts <= 0] = np.nan
        # keep nas in, need to discard if too many in window

        # build empty dataset same rows as labels
        data = {}
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
        labels = labels[mask] # keep only labels that fall within the segments

        if labels.shape[0] != 0:
            # select out the windows
            df, labels = get_window(
                ts, ts_index, labels, window_index, window_s, percentage=percentage
            )

            # if I am smoothing, I will drop the windows without enough labels
            assert len(df) == labels.shape[0]
            if strategy == "smooth":
                df2 = []
                for i, d in enumerate(df):
                    if labels["start_idx"].iloc[i] in labels.index:
                        d = np.append(d, i)
                        df2.append([d])
                df = np.concatenate(df2, axis = 0)

            # in_out computation should not be based on imputed values
            if v == "abp":
                windows = [
                    {"w": ts[i[0] : i[1]], "overlap_len": i[2], "total_length": i[3]}
                    for i in df
                ] 
                ref = None
                if strategy == "smooth":
                    ref = list([int(v[4]) for v in df])
                in_out = extract_proportions(
                    windows, labels, percentage=percentage, strategy=strategy, ref=ref
                )

            # extract window data: impute
            windows = [
                {
                    "w": impute(ts[i[0] : i[1]]),
                    "overlap_len": i[2],
                    "total_length": i[3],
                }
                for i in df
            ]

            # drop if imputation returned none
            windows_filtered = [
                {
                    "w": w["w"],
                    "overlap_len": w["overlap_len"],
                    "total_length": w["total_length"],
                }
                for i, w in enumerate(windows)
                if w["w"] is not None
            ]
            labels = labels.iloc[
                [i for i, w in enumerate(windows) if w["w"] is not None]
            ]
            df = df[[i for i, w in enumerate(windows) if w["w"] is not None]]

            # extract proportion_in T/F data
            if v == "abp":
                in_out = in_out[
                    [i for i, w in enumerate(windows) if w["w"] is not None]
                ]

                # drop ref
                df = np.array(df)[:, :4]
                
                df = pd.DataFrame(
                    df, columns=["startidx", "endidx", "overlap_len", "tot_len"]
                )
                df["datetime"] = np.array(labels["DateTime"])
                df["in?"] = in_out
                df["ptid"] = ptid

                if len(in_out) == 0:
                    print("No valid windows extracted for patient:", ptid)
                return df, windows_filtered
            else:
                return (
                    pd.DataFrame(labels["DateTime"])
                    .reset_index(drop=True)
                    .rename(columns={"DateTime": "datetime"}),
                    windows_filtered,
                )  # only compute in_out for abp

        else:
            return None, None
