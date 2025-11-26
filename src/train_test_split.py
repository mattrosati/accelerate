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

    df = pd.DataFrame(df_rows, columns=["ptid", "out_percent", "to_mapopt_s", "seg_len"]).dropna()

    # make keys by cutting into tertiles for out_percent, time to map opt, and length
    df["out_percent_cat"] = pd.qcut(df["out_percent"], q=3, labels=False)
    df["mapopt_cat"] = pd.qcut(df["to_mapopt_s"], q=3, labels=False)
    df["seg_len_cat"] = pd.qcut(df["seg_len"], q=3, labels=False)
    df["class"] = df["out_percent_cat"].astype(str) + df["mapopt_cat"].astype(str) + df["seg_len_cat"].astype(str)

    print(df)
    print(df.describe(percentiles=[.25, .5, .75, .90,]))
    print(df["class"].value_counts())

    keys = df["class"]
    train, test = train_test_split(
        df, random_state=42, train_size=args.train_frac, stratify=keys
    )

    # apply train or test label to f.ptid.attrs
    for t in train["ptid"]:
        f[t].attrs["split"] = "train"
    for t in test["ptid"]:
        f[t].attrs["split"] = "test"
