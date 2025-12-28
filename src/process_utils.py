import os
import sys

import h5py
import numpy as np
import pandas as pd

from constants import *

from dataclasses import dataclass
    
def robust_floor(x, tol=1e-5):
    return np.floor(x + tol)

def robust_ceil(x, tol=1e-5):
    return np.ceil(x - tol)

def merge_quality_intervals(valid_df, bad_df):

    bad_df = bad_df.copy()
    valid_df = valid_df.copy()
    valid_df["segment_id"] = range(len(valid_df))

    # Collect events for sweep-line processing
    events = []

    # Valid segment events
    for _, r in valid_df.iterrows():
        events.append((r.starttime, 1, "valid", r.segment_id, r.frequency))
        events.append((r.endtime, -1, "valid", r.segment_id, r.frequency))

    # Bad-quality events
    for _, r in bad_df.iterrows():
        events.append((r.starttime, 1, "bad", None, None))
        events.append((r.endtime, -1, "bad", None, None))

    # Sort by time, with end(-1) before start(+1)
    events = sorted(events, key=lambda x: (x[0], x[1]))

    # Sweep-line state
    active_segments = {}  # segment_id → frequency
    bad_active = 0
    current_start = None
    current_seg = None
    current_freq = None
    pieces = []

    # Sweep
    for t, delta, kind, seg_id, freq in events:

        # If we were in a "good" region and it's ending
        if active_segments and bad_active == 0 and current_start is not None:
            pieces.append(
                {
                    "segment_id": current_seg,
                    "starttime": current_start,
                    "endtime": t,
                    "frequency": current_freq,
                }
            )
            current_start = None

        # Update state
        if kind == "valid":
            if delta == 1:
                active_segments[seg_id] = freq
            else:
                active_segments.pop(seg_id, None)
        else:  # bad interval
            bad_active += delta

        # If entering a "good" region
        if active_segments and bad_active == 0 and current_start is None:
            # Choose the (only) active segment
            # If overlapping segments exist, choose the earliest start
            current_seg = sorted(active_segments.keys())[0]
            current_freq = active_segments[current_seg]
            current_start = t

    pieces_df = pd.DataFrame(pieces)

    # If no pieces exist (all bad), return empty
    if pieces_df.empty:
        return pieces_df

    # Finalize (recompute length + startidx)
    final = finalize_segments(pieces_df, valid_df)

    return final


def finalize_segments(df, valid_df):
    """Recalculate length (sample count) and startidx per segment."""
    out = df.copy()

    # Pull original segment metadata
    seg_info = valid_df.set_index("segment_id")[["starttime", "startidx", "frequency"]]

    # Length = number of samples
    duration = (out["endtime"] - out["starttime"]) / 1e6
    out["length"] = robust_ceil((duration * out["frequency"])).astype(np.uint64)

    # Compute true startidx based on original raw data index
    def compute_true_startidx(row):
        seg = row.segment_id
        orig = seg_info.loc[seg]
        # Offset in seconds
        dt = (row.starttime - orig.starttime) / 1e6
        # Convert to samples
        sample_offset = robust_ceil((dt * orig.frequency)).astype(np.uint64)
        return orig.startidx + sample_offset

    out["startidx"] = out.apply(compute_true_startidx, axis=1)
    out.drop(columns="segment_id", inplace=True)

    return out


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


def extract_proportions(windows, labels, config, ref=None):
    percentage = config.percentage
    strategy = config.strategy
    if strategy == "count":
        return extract_proportions_count(windows, labels, percentage)
    elif strategy == "mean":
        return extract_proportions_mean(windows, labels)
    elif strategy == "smooth":
        return extract_proportions_smooth(windows, labels, percentage, ref, config)


def extract_proportions_smooth(windows, labels, percentage, ref, config):
    smooth_frac = config.smooth_frac
    r2_min = config.r2_threshold
    in_out = np.empty(shape=len(windows))
    for i, w in enumerate(windows):
        start = labels["start_idx"].iloc[ref[i]]
        end = labels["end_idx"].iloc[ref[i]]
        lower_limits = labels["LLA_Yale_affected_beta"].loc[start:end]
        upper_limits = labels["ULA_Yale_affected_beta"].loc[start:end]

        r2 = labels["Yale_R2full_affected"].loc[start:end]
        if r2_min > 0.0 and (r2 < r2_min).any():
            in_out[i] = np.nan
            continue

        # if there's missing data in the LA calculation, label is na
        if lower_limits.isna().any() or upper_limits.isna().any():
            in_out[i] = np.nan
            continue
        w_vector = w["w"]

        if w["total_length"].sum() == 0:
            print("Record with no windows, unclear why")

        # split in minute by minute
        intervals = len(lower_limits)
        if len(w_vector) % intervals != 0:
            print(ref[i])
            print(start, end)
            print(w_vector.shape)
            print(lower_limits)
            # print(upper_limits)
        split_window = w_vector.reshape((intervals, -1), order="C")

        # find average of window
        if (np.isnan(split_window)).all(axis=1).any():
            # these entries will get removed once we filter for windows that have a lot of nans
            in_out[i] = np.nan
            continue

        mean_arr = np.nanmean(split_window, axis=1)

        # find fraction of time outside limits
        out = (mean_arr < lower_limits.values) | (mean_arr > upper_limits.values)
        frac_out = out.mean()

        # set true if outside <= 20% of the time
        in_out[i] = frac_out <= smooth_frac
        # label as “outside AR” if >46% of time outside AR: which means that "inside AR" if >=54% of time inside
        # in_out[i] = frac_out <= 0.46


    return in_out


def extract_proportions_mean(windows, labels):
    in_out = np.empty(shape=len(windows), dtype=bool)
    for i, w in enumerate(windows):
        lower_limit = labels["LLA_Yale_affected_beta"].iloc[i]
        upper_limit = labels["ULA_Yale_affected_beta"].iloc[i]

        if pd.isna(lower_limit) or pd.isna(upper_limit):
            in_out[i] = np.nan
            continue

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


def get_window(data, index, coords, window_index, window_s, config):
    # will obtain window start and end idx, plus total length and overlap length (in tokens)
    # inputs: data (dataset with data of interest), index (contains segment information),
    # coords (contains labels and their coordinates in time), window_index (token index of label value in window),
    # window_s (duration of window in seconds), percentage (max percentage window allowed to overlap with gap)
    # compute closest idx for data that matches coords
    percentage = config.percentage
    seg_start = index["starttime"].iloc[coords["segment"]].to_numpy()
    seg_end = index["endtime"].iloc[coords["segment"]].to_numpy()
    seg_freq = index["frequency"].iloc[coords["segment"]].to_numpy()
    # tokens_from_seg_start = (
    #     robust_ceil(((coords["DateTime"].to_numpy() - seg_start) * seg_freq) / 1e6 )
    #     .astype(np.int64)
    # )
    # closest_idx = (
    #     index["startidx"].iloc[coords["segment"]].to_numpy() + (tokens_from_seg_start - 1)
    # )
    assert seg_start.shape == coords["segment"].shape

    # get position of window bounds in absolute time to check if outside segments
    window_us = window_s * 1e6
    window_index_us = (
        window_index + 1
    ) * 1e6  # I want to look back at least index + 1 tokens to set it as start of the time of the window and then pull only the data within there.
    window_start = coords["DateTime"].to_numpy() - window_index_us
    window_end = window_start + window_us
    window_length = window_end - window_start
    total_window_token_length = (
        ((window_length * seg_freq) / 1e6 ).round().astype(np.uint64)
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
        (robust_floor(((window_start_clipped - seg_start) * seg_freq) / 1e6))
        .astype(np.uint64)
    )
    window_end_tokens = (
        (robust_floor(((window_end_clipped - seg_start) * seg_freq) / 1e6))
        .astype(np.uint64)
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


def get_windows_var(v, ptid, window_index, window_s, config):
    file_path = config.data_file
    strategy = config.strategy
    percentage = config.percentage

    with h5py.File(file_path, "r") as f:
        # load labels[targets] and var timeseries
        labels = pd.DataFrame(f[f"{ptid}/labels"][...][TARGETS + ["DateTime"]])
        if v in f[f"{ptid}/raw/numerics"].keys():
            v_long = f"numerics/{v}"
        else:
            v_long = f"waves/{v}"
        ts = f[f"{ptid}/raw/{v_long}"][...]
        ts_index = pd.DataFrame(f[f"{ptid}/raw/{v_long}"].attrs["index"])

        # replace invalid vals, do not filter out
        invalid_val = f.attrs["invalid_val"]
        labels.replace(invalid_val, np.nan, inplace=True)
        # labels.dropna(inplace=True)

        #make everything uint64
        labels.DateTime = labels.DateTime.astype(np.uint64)
        ts_index = ts_index.astype(np.uint64)

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
            ((ts_index["length"] - 1) / ts_index["frequency"]) * 1e6
        ).astype(np.uint64)

        # add segments with non-zero quality to index to filter bad quality out
        quality = pd.DataFrame(
            f[f"{ptid}/raw/{v_long}"].attrs["quality"]
        )  # df with columns: time, quality code
        assert (
            quality["value"].isna().sum() == 0
        ), "Unexpected: quality value contains nas"
        # shift to get end time of quality segments
        quality["endtime"] = quality["time"].shift(
            -1, fill_value=ts_index["endtime"].max()
        )
        quality.rename(columns={"time": "starttime"}, inplace=True)

        # find segments with non-zero quality
        problem_segments = quality[quality["value"] != 0]

        # merge
        ts_index_orig = ts_index.copy()
        ts_index = merge_quality_intervals(ts_index, problem_segments)
        ts_index = ts_index.astype(np.uint64)

        if np.any(
            ts_index.endtime
            > ts_index.starttime.shift(-1, fill_value=ts_index.endtime.max())
        ):
            print("Warning: overlapping segments detected after quality merging.")
            print(
                pd.concat(
                    [
                        ts_index.endtime,
                        ts_index.starttime.shift(-1, fill_value=ts_index.endtime.max()),
                    ],
                    axis=1,
                )
            )
            print(v, ptid)

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
        labels = labels[mask]  # keep only labels that fall within the segments

        if labels.shape[0] != 0:
            # select out the windows
            df, labels = get_window(
                ts, ts_index, labels, window_index, window_s, config
            )

            # if I am smoothing, I will drop the windows without enough labels
            assert len(df) == labels.shape[0]
            if strategy == "smooth":
                df2 = []
                for i, d in enumerate(df):
                    if labels["start_idx"].iloc[i] in labels.index:
                        d = np.append(d, i)
                        df2.append([d])
                if len(df2) == 0:
                    print(f"no windows for ptid {ptid}")
                    return None, None
                df = np.concatenate(df2, axis=0)

            # in_out computation should not be based on imputed values
            if v == "abp":
                windows = [
                    {"w": ts[i[0] : i[1]], "overlap_len": i[2], "total_length": i[3]}
                    for i in df
                ]
                ref = None
                if strategy == "smooth":
                    ref = df[:, 4].tolist()
                in_out = extract_proportions(
                    windows, labels, config, ref=ref
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
                df["invalid"] = df["in?"].isna()


                if len(in_out) == 0:
                    print("No valid windows extracted for patient:", ptid)
                return df, windows_filtered
            else:
                df = (
                    pd.DataFrame(labels["DateTime"])
                    .rename(columns={"DateTime": "datetime"})
                    .reset_index(drop=True)
                )
                return df, windows_filtered  # only compute in_out for abp

        else:
            return None, None
