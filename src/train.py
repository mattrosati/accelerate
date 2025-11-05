import os
import sys
import shutil
from argparse import ArgumentParser
from pickle import dump

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

from sklearn.neighbors import KNeighborsClassifier

from data_utils import build_continuous_time, load_label
from constants import *
from process_utils import *

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--train_dir",
        help="Directory to training data and to store models.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/training/w_300s_hr_rso2r_rso2l_spo2_abp",
        required=True,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Type of model to train",
        choices=["log_reg", "xgboost", "svm", "knn", "rand_forest"],
        required=True,
    )

    args = parser.parse_args()
    np.random.seed(420)

    # make model saving dir
    model_store = os.path.join(args.train_dir, "models")
    os.makedirs(model_store, exist_ok=True)

    # init model
    if args.model == "log_reg":
        pass
    elif args.model == "xgboost":
        pass
    elif args.model == "svm":
        pass
    elif args.model == "knn":
        model = KNeighborsClassifier()
        params = {
            "n_neighbors": np.linspace(1, 1_000, num=20).astype(int),
            "weights": ["uniform", "distance"],
        }
    elif args.model == "rand_forest":
        pass

    # load data

    # init random search
    # n_examples = 4 * len(X_train)
    n_params = 20
    n_iters = n_params
    search = HyperbandSearchCV(
        model,
        params,
        max_iter=max_iter,
        patience=True,
    )

    # train
    search.fit(X_train, y_train)

    # print best scores

    # save model
    dump(
        search.best_estimator_,
        open(os.path.join(model_store, f"{args.model}.pkl"), "wb"),
    )
