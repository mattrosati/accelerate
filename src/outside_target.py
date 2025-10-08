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

from data_utils import build_continuous_time, load_label
from constants import TARGETS


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
