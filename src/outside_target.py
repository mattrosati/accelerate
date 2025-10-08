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

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from data_utils import build_continuous_time, load_label
from constants import TARGETS, FEATURES


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract information about status within limits for each patient."
    )

    parser.add_argument("data_file", help="Path to combined dataset")
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

    # need to output dataframe per patient N x 1 with True or False depending on whether timepoint is in (True) or out (False)
    # N/A if window overlaps with a gap which makes 50% identification impossible

    # func_tbd needs to load dataset
    # global_f = h5py.File(args.data_file, "r+")

    # load only to get length
    with h5py.File(args.data_file, "r") as f:
        ptids = list(global_f.keys())

    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(
            tqdm(
                ex.map(
                    partial(func_TBD),
                    ptids,
                )
            )
        )

    distributions = dict(results)

    pt_group.create_dataset("labels", data=arr, dtype=arr.dtype)
