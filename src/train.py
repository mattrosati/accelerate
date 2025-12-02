import os
import sys
import shutil
from argparse import ArgumentParser
from pickle import dump
from datetime import datetime
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

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, auc

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from dask_ml.model_selection import HyperbandSearchCV

from ray import tune

import xgboost as xgb

from aeon.classification.convolution_based import MultiRocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

from data_utils import build_continuous_time, load_label
from constants import *
from process_utils import *
from tuner import RayAdaptiveRepeatedCVSearch

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--train_dir",
        help="Directory to training data and to store models.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/training/w_300s_hr_rso2r_rso2l_spo2_abp",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Type of model to train",
        choices=[
            "log_reg",
            "svm",
            "knn",
            "rand_forest",
            "decision_tree",
            "xgb",
            "rocket",
            "knn_multivar",
        ],
        required=True,
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        choices=["raw", "pca", "fpca", "separate_pca"],
        help="Transofrmations applied to mode prior to loading.",
        default="raw",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Trains on smaller subset of data. Saves models separately.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Names run, defaults to timestamp if empty.",
    )

    args = parser.parse_args()
    np.random.seed(420)
    random.seed(420)

    print(f"Training model {args.model} with {args.data_mode} embeddings.")

    # decide what embeddings to use and load data
    if args.data_mode == "raw":
        f = "x.zarr"
    elif args.data_mode == "pca":
        f = "pca_x.zarr"
    elif args.data_mode == "fpca":
        f = "fpca_x.zarr"
    elif args.data_mode == "separate_pca":
        f = "separate_decomp_x.zarr"
    X_train = da.from_zarr(os.path.join(args.train_dir, "permanent", "train", f))
    labels = pd.read_pickle(
        os.path.join(args.train_dir, "permanent", "train", "labels.pkl")
    )
    y_train = labels["in?"].astype(int)
    groups = labels["ptid"].astype(str)

    # shuffle and index if debugging
    if args.debug:
        print("Using a smaller dataset size to be speedy.")
        length = 10_000
        keys = da.random.random(X_train.shape[0], chunks=X_train.chunksize[0])
        perm = np.argsort(keys)[:length]
        X_train = X_train[perm]
        y_train = y_train[perm]
        groups = groups[perm]
        print(f"Training dataset shape: {X_train.shape}.")

    # make model saving dir, timestamped or run_name
    if args.run_name == "":
        run_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
    else:
        run_name = args.run_name
    models_dir_name = "models_debug" if args.debug else "models"
    models_dir_name = f"{models_dir_name}_{run_name}"
    model_store = os.path.join(args.train_dir, models_dir_name)
    os.makedirs(model_store, exist_ok=True)

    # init accuracies
    metrics = {
        "auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }

    # init model
    n_iter = 30
    multivar = False
    if args.model == "log_reg":
        model = LogisticRegression(n_jobs=1, solver="saga", max_iter=10_000)
        params = {
            "penalty": tune.choice(["l2", "l1"]),
            "C": tune.loguniform(0.001, 100),
        }
    elif args.model == "decision_tree":
        model = DecisionTreeClassifier()
        params = {
            "max_depth": tune.qrandint(5, 50, 5),
            "max_features": tune.choice([0.02, 0.05, 0.1, "sqrt"]),
            "min_samples_split": tune.randint(2, 25),
            "min_samples_leaf": tune.randint(1, 15),
            "class_weight": tune.choice([None, "balanced"]),
        }
        n_iter = len(list(params.keys())) * 10
    elif args.model == "svm":
        model = svm.SVC()
        params = {
            "C": tune.loguniform(0.001, 100),
            "gamma": tune.loguniform(1 / (X_train.shape[1] ** 3), 0.01),
        }
    elif args.model == "knn":
        model = KNeighborsClassifier(n_jobs=1)
        params = {
            "n_neighbors": tune.lograndint(5, 1_000),
            "weights": tune.choice(["uniform", "distance"]),
        }
    elif args.model == "rand_forest":
        model = RandomForestClassifier()
        params = {
            "n_estimators": tune.lograndint(50, 0.05 * X_train.shape[0]),
            "max_depth": tune.qrandint(5, 50, 5),
            "max_features": tune.choice([0.02, 0.05, 0.1, "sqrt"]),
            "min_samples_split": tune.randint(2, 25),
            "min_samples_leaf": tune.randint(1, 15),
            "class_weight": tune.choice([None, "balanced"]),
        }
        n_iter = (len(list(params.keys())) - 1) * 10
    elif args.model == "xgb":
        model = xgb.XGBClassifier(tree_method="hist", eval_metric="logloss")
        params = {
            "n_estimators": tune.lograndint(50, 0.05 * X_train.shape[0]),
            "max_depth": tune.randint(2, 10),
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "subsample": tune.quniform(0.25, 0.75, 0.05),
            "colsample_bytree": tune.quniform(0.5, 1.0, 0.05),
            "gamma": tune.quniform(0, 5, 0.25),
            "reg_alpha": tune.loguniform(1e-5, 20.0),
            "reg_lambda": tune.loguniform(1e-5, 20.0),
        }
        n_iter = (len(list(params.keys())) - 2) * 10
    elif args.model == "rocket":
        multivar = True
        params = {
            "n_kernels": tune.choice([5_000, 10_000, 15_000]),
            "max_dilations_per_kernel": tune.choice([32, 64]),
            "n_features_per_kernel": tune.qrandint(4, 16, 2),
            "estimator__alpha": tune.loguniform(1e-3, 1e3),  # Ridge reg
            "class_weight": tune.choice([None, "balanced"]),
        }
        model = MultiRocketClassifier(estimator=RidgeClassifier())
    elif args.model == "knn_multivar":
        multivar = True
        params = {
            "n_neighbors": tune.lograndint(5, 1_000),
            "distance": tune.choice(["euclidean", "msm", "erp"]),
            "weights": tune.choice(["uniform", "distance"]),
        }
        model = KNeighborsTimeSeriesClassifier()
    elif args.model == "rocket_xgb":
        multivar = True
        # needs params
        # needs model

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)  # repeats five folds 2 times
    search = RayAdaptiveRepeatedCVSearch(
        estimator=model,
        search_space=params,
        scoring=metrics,
        num_samples=n_iter,
        rank_metric="mean_val_auc",
        cv=[5, 3],
        store_path=model_store,
        model_name=f"{args.model}_{args.data_mode}",
    )
    X_train = X_train.compute()

    if multivar and "downsample" in args.train_dir:
        train_dir_name = os.path.basename(args.train_dir.rstrip("/"))
        train_params = train_dir_name.split("_")

        seconds = None
        channels = []
        for token in train_params:
            if token.endswith("s") and token[:-1].isdigit():
                seconds = int(token[:-1])
            elif seconds is not None:
                channels.append(token)

        num_channels = len(channels)
        print("Detected channels:", channels)
        print("num_channels =", num_channels)

        timesteps = X_train.shape[1] // num_channels

        # Reshape to AEON format: (N, num_channels, timesteps)
        X_train = X_train.reshape(X_train.shape[0], num_channels, timesteps)
        print(f"Reshaped training set to {X_train.shape}")
    assert not (
        multivar and "downsample" not in args.train_dir
    ), "Cannot train multivariable model on not downsampled data."

    # train
    search = search.fit(X_train, y_train, groups=groups)

    # save model
    dump(
        search.best_estimator_,
        open(os.path.join(model_store, f"{args.model}_{args.data_mode}.pkl"), "wb"),
    )

    print("Search results:")
    df = pd.DataFrame(search.cv_results_())
    print(df)

    # results will be saved in ray tuner logs
