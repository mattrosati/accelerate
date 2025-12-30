import os
import sys
import shutil
from argparse import ArgumentParser
from pickle import dump
import random

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
from process_utils import *

from dask_ml.preprocessing import StandardScaler, RobustScaler
from dask_ml.decomposition import PCA

from sklearn.preprocessing import PowerTransformer


def do_design(X_train, num_channels, channels):

    timesteps = X_train.shape[1] // num_channels
    chan_to_index = {c: i for i, c in enumerate(channels)}

    # reshape to [N, C, T]
    X_train = X_train.reshape(X_train.shape[0], num_channels, timesteps)
    print(f"Reshaped dataset to {X_train.shape}")

    # 2: Nirs asymmetry if present
    nirs_diff = (
        X_train[:, chan_to_index["rso2r"], :] - X_train[:, chan_to_index["rso2l"], :]
    )
    nirs_diff = nirs_diff[:, None, :]

    # attach nirs_diff to X-train
    X_train = da.concatenate([X_train, nirs_diff], axis=1)
    chan_to_index["nirs_diff"] = len(channels)

    # for each window do the following
    # 1: mean, std of window for each channel
    means = X_train.mean(axis=2).reshape((X_train.shape[0], -1))
    stds = X_train.std(axis=2).reshape((X_train.shape[0], -1))
    print(means.mean().compute())
    print(stds.mean().compute())

    # whiten
    # if args.whiten:
    #     X_train = X_train[:, :, 1:] - X_train[:, :, :-1]

    # cross-correlation of abp with each of the other variables
    print("Doing cross correl")
    abp_idx = chan_to_index["abp"]
    rows = []
    X_train = X_train.compute()
    for i in tqdm(range(X_train.shape[0])):
        channel_corrs = []
        X = X_train[i, :, :]
        for c in chan_to_index.keys():
            idx = chan_to_index[c]
            if c != "abp":
                x = X[abp_idx, :]
                y = X[idx, :]
                corr = np.correlate(x - x.mean(), y - y.mean(), mode="full")
                corr /= np.std(x) * np.std(y) * X.shape[-1]
                assert corr.shape[0] == 2 * X.shape[-1] - 1

                # corr will be of length 2 * timepoints - 1
                # we want to extract all correlations for +/- (timepoints/60 - 1) minutes
                mid = corr.shape[0] // 2
                relevant_corr = np.concat(
                    [corr[mid::-60][::-1], corr[mid + 60 :: 60][:-1]]
                )
                assert relevant_corr.shape[0] == 38

                channel_corrs.append(relevant_corr)
        rows.append(np.concatenate(channel_corrs))

    correlations_x = np.stack(rows, axis=0)

    # merge all vectors together
    designed_feats = [means, stds, correlations_x]
    hello = np.concat(designed_feats, axis=-1)
    return np.concat(designed_feats, axis=-1)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--train_dir",
        type=str,
        help="Directory to training data.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--overwrite_permanent",
        help="Will overwrite the permanent directory.",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--whiten",
        help="Will whiten the signal with t-1.",
        action="store_true",
    )

    args = parser.parse_args()
    config = args
    np.random.seed(420)
    random.seed(420)
    pd.options.display.float_format = "{:.2f}".format

    # parse the save directory to get the information of interest
    train_dir_name = os.path.basename(args.train_dir.rstrip("/"))
    train_params = train_dir_name.split("_")

    seconds = None
    channels = []
    for token in train_params:
        if token.endswith("s") and token[:-1].isdigit():
            seconds = int(token[:-1])
        elif seconds is not None:
            channels.append(token)
        if token.startswith("freq"):
            freq = int(token.removeprefix(prefix="freq"))
    if "freq" not in train_dir_name:
        freq = 60

    num_channels = len(channels)
    print("Detected channels:", channels)
    print("num_channels =", num_channels)
    print("frequency:", freq)
    assert freq == 60

    # load the raw training values
    X_train = da.from_zarr(os.path.join(args.train_dir, "permanent", "train", "x.zarr"))
    X_test = da.from_zarr(os.path.join(args.train_dir, "permanent", "test", "x.zarr"))

    dfeat = do_design(X_train, num_channels, channels)
    dfeat_test = do_design(X_test, num_channels, channels)
    print(f"Final array has shape {dfeat.shape}")
    print(f"Final test array has shape {dfeat_test.shape}")
    print("Mean:", dfeat.mean().compute())

    # 5: save
    if (
        os.path.exists(os.path.join(args.train_dir, "permanent"))
        and not args.overwrite_permanent
    ):
        print("WARNING: not overwriting permanent to avoid data chaos.")
    else:
        # if overwriting, check if file names are same (then delete old and overwrite) or different (then just add new)
        for split in ["train", "test"]:
            permanent_path = os.path.join(args.train_dir, "permanent", split)
            for f in os.listdir(permanent_path):
                if "design" in f:
                    shutil.rmtree(os.path.join(permanent_path, f))
        da.to_zarr(
            dfeat,
            url=os.path.join(args.train_dir, "permanent", "train", f"design_x.zarr"),
        )
        da.to_zarr(
            dfeat_test,
            url=os.path.join(args.train_dir, "permanent", "test", f"design_x.zarr"),
        )
