import os
import sys
from argparse import ArgumentParser

import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import tqdm

from data_utils import build_continuous_time, load_label, printname
from constants import TARGETS


if __name__ == "__main__":
    parser = ArgumentParser(description="Train and test splits.")

    parser.add_argument("data_dir", help="Path to processed data file.")

    args = parser.parse_args()

    np.random.seed(420)

    # make destination h5py file
    global_f = h5py.File(args.data_dir, "a")

    # for train test split:
    # want to control for the time spent in and out and time required to get targets
    # so I likely want to save both of these stats in the dataset
    # and then extract train_test_splits based on a df with these stats

    # get time out

    # copied:
    true_false = {}
    ins = []
    mode = "mean"
    with h5py.File(global_path, "r") as f:
        for pt in f:
            if f[f"{pt}/processed/in_out_{mode}"].attrs["no_label_overlap"]:
                continue
            in_out_df = pd.Series(
                f[f"{pt}/processed/in_out_{mode}/in_out"][...]
            ).astype(bool)
            idx_window = pd.DataFrame(
                f[f"{pt}/processed/in_out_{mode}/window_idx"][...]
            )

            # to actually get percentage of time spent outside autoregulation, we need to get actual window length, we can't weigh all the windows equally
            len_window = idx_window.iloc[:, 1] - idx_window.iloc[:, 0]
            in_out = (in_out_df * len_window).sum() / len_window.sum()
            if len_window.sum() == 0:
                print(pt)
                print(idx_window)
                print(in_out_df)

            true_false[pt] = [in_out]
            ins.append(in_out_df)

    # get list of time spent in and out and add to processed data

    # get list of time to target calc and add to processed data
