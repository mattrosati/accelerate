import os
import sys
import random
from argparse import ArgumentParser

import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from data_utils import build_continuous_time, load_label, printname, find_time_elapsed
from constants import TARGETS

from sklearn.model_selection import train_test_split


def make_qcut_codes(series, q=3):
    """Return stable quantile-bin codes even when the series has few uniques."""
    bins = min(q, series.nunique())
    if bins <= 1:
        return pd.Series(np.zeros(series.shape[0], dtype=int), index=series.index)
    return pd.qcut(series, q=bins, labels=False, duplicates="drop").astype(int)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train and test splits.")

    parser.add_argument(
        "--data_dir",
        help="Path to processed data file.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/all_data.hdf5",
    )
    parser.add_argument(
        "--train_frac", help="Fraction of data to be in training set.", default=0.8
    )
    parser.add_argument(
        "--in_out_mode", help="Mode of in_out, will be 'mean'.", default="mean"
    )
    parser.add_argument(
        "--labels_dir",
        help="Path to processed data file.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/labels",
    )

    args = parser.parse_args()

    mode = args.in_out_mode
    np.random.seed(420)
    random.seed(420)

    # make destination h5py file
    f = h5py.File(args.data_dir, "a")

    # for train test split:
    # want to control for the time spent in and out, time required to get targets, and duration of recording
    # so I likely want to save both of these stats in the dataset
    # and then extract train_test_splits based on a df with these stats

    # get % time out and time to mapopt
    df_rows = []
    calc = TARGETS
    ptids = np.array(f["healthy_ptids"][...]).astype(str)
    for pt in ptids:
        in_out_df = pd.Series(f[f"{pt}/processed/in_out_{mode}/in_out"][...]).astype(
            bool
        )
        idx_window = pd.DataFrame(f[f"{pt}/processed/in_out_{mode}/window_idx"][...])
        len_window = idx_window.iloc[:, 1] - idx_window.iloc[:, 0]
        in_percent = (in_out_df * len_window).sum() / len_window.sum()
        if len_window.sum() == 0:
            print(pt)
            print(idx_window)
            print(in_out_df)

        time_to_mapopt = find_time_elapsed(
            pt,
            calc,
            args.labels_dir,
            time="s",
            start_time=int(f[f"{pt}/raw/"].attrs["dataStartTimeUnix"][0]),
        )

        # add to attrs
        f[f"{pt}/processed"].attrs["out_percent"] = 1 - in_percent
        f[f"{pt}/processed"].attrs["time_to_mapopt_s"] = time_to_mapopt

        # append row of pt, % in, time to mapopt
        df_rows.append([pt, 1 - in_percent, time_to_mapopt, in_out_df.shape[0]])

    df = pd.DataFrame(
        df_rows, columns=["ptid", "out_percent", "to_mapopt_s", "seg_len"]
    ).dropna()
    df["base_ptid"] = df["ptid"].astype(str).str.split("_").str[0]

    patient_rows = []
    for base_ptid, group in df.groupby("base_ptid"):
        weights = group["seg_len"].to_numpy(dtype=float)
        if weights.sum() == 0:
            weights = np.ones(group.shape[0], dtype=float)

        # Aggregate multipart recordings to one row so splitting happens at the
        # true patient level rather than the file level.
        patient_rows.append(
            {
                "base_ptid": base_ptid,
                "out_percent": np.average(group["out_percent"], weights=weights),
                "to_mapopt_s": group["to_mapopt_s"].min(),
                "seg_len": group["seg_len"].sum(),
            }
        )

    patient_df = pd.DataFrame(patient_rows)

    # make keys by cutting into tertiles for out_percent, time to map opt, and length
    patient_df["out_percent_cat"] = make_qcut_codes(patient_df["out_percent"], q=3)
    patient_df["mapopt_cat"] = make_qcut_codes(patient_df["to_mapopt_s"], q=3)
    patient_df["seg_len_cat"] = make_qcut_codes(patient_df["seg_len"], q=3)
    patient_df["class"] = (
        patient_df["out_percent_cat"].astype(str)
        + patient_df["mapopt_cat"].astype(str)
        + patient_df["seg_len_cat"].astype(str)
    )

    print(patient_df)
    print(
        patient_df.describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.90,
            ]
        )
    )
    print(patient_df["class"].value_counts())

    keys = patient_df["class"]
    stratify = keys if keys.value_counts().min() >= 2 else None
    train, test = train_test_split(
        patient_df, random_state=42, train_size=args.train_frac, stratify=stratify
    )

    # apply train or test label to f.ptid.attrs
    train_ids = set(train["base_ptid"])
    test_ids = set(test["base_ptid"])
    for t in df["ptid"]:
        base_ptid = t.split("_")[0]
        f[t].attrs["base_ptid"] = base_ptid
        if base_ptid in train_ids:
            f[t].attrs["split"] = "train"
        elif base_ptid in test_ids:
            f[t].attrs["split"] = "test"
