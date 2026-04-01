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

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import zarr
import dask.array as da

from data_utils import build_continuous_time, load_label
from constants import *
from process_utils import *

from sklearn.preprocessing import StandardScaler, RobustScaler
from dask_ml.decomposition import PCA
from sklearn.decomposition import PCA as skPCA

from sklearn.preprocessing import PowerTransformer

import multiprocessing as mp


def do_tests(X):
    # Check non-finite counts
    n_nan = da.isnan(X).sum()
    n_inf = da.isinf(X).sum()
    mx = da.nanmax(da.abs(X))

    print("nan:", n_nan.compute())
    print("inf:", n_inf.compute())
    print("max|x|:", mx.compute())

    # Check for all-NaN rows or columns
    all_nan_rows = da.isnan(X).all(axis=1).sum()
    all_nan_cols = da.isnan(X).all(axis=0).sum()
    print("all-NaN rows:", all_nan_rows.compute())
    print("all-NaN cols:", all_nan_cols.compute())

    # Check for zero-variance columns (after ignoring NaNs)
    col_var = da.nanvar(X, axis=0)
    n_zero_var = (col_var == 0).sum()
    print("zero-var cols:", n_zero_var.compute())

    return


def _scale_and_mark_outliers(z_arr_flat, v, scaler_mode):
    """Fit (or select) scaler, transform, and mark 3-SD outliers as NaN.

    Parameters
    ----------
    z_arr_flat : np.ndarray, shape (-1, 1)
        Flattened raw values for one variable.
    v : str
        Variable name (needed for spo2 flip).
    scaler_mode : str or None
        "robust" or None (standard/power).

    Returns
    -------
    scaler, scaled_values (with outliers as NaN), col_mean, col_std
    """
    if scaler_mode == "robust":
        if v == "spo2":
            z_arr_flat = 100.0 - z_arr_flat
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
    else:
        if v == "spo2":
            scaler = PowerTransformer()
            z_arr_flat = 100.0 - z_arr_flat
        else:
            scaler = StandardScaler()

    scaled = scaler.fit_transform(z_arr_flat)

    col_std = np.sqrt(scaler.var_[0])
    col_mean = scaler.mean_[0]
    outlier_mask = np.abs(z_arr_flat - col_mean) >= 3 * col_std
    scaled[outlier_mask] = np.nan

    return scaler, scaled, col_mean, col_std


def _graph_norm_effect(
    raw_samples, scaled_arr, v, split_label, img_dir, n_samples=100_000
):
    """Plot before/after normalization histograms for one variable + split."""
    sampled_pre = np.random.choice(raw_samples, size=min(n_samples, len(raw_samples)), replace=False)
    flat_scaled = scaled_arr.ravel()
    sampled_post = np.random.choice(flat_scaled, size=min(n_samples, len(flat_scaled)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.histplot(sampled_pre, ax=axes[0], stat="probability", edgecolor=(0, 0, 0, 0.5), alpha=0.5)
    axes[0].set_title("Before Normalization")
    axes[0].set_xlabel(f"{v}")
    sns.histplot(sampled_post, ax=axes[1], stat="probability", edgecolor=(0, 0, 0, 0.5), alpha=0.5)
    axes[1].set_title("After Normalization")
    axes[1].set_xlabel(f"{v}")
    fig.suptitle(f"Effect of normalization on {v} ({split_label})")
    plt.savefig(os.path.join(img_dir, f"{v}_norm_effect_{split_label.lower().replace(' ', '_')}.png"))
    plt.close()


def normalize(
    save_dir,
    variables,
    img_dir="/home/mr2238/project_pi_np442/mr2238/accelerate/imgs/normalize_impute_rs_new",
    graph=False,
    scaler_mode=None,
    na_threshold=PERCENT_NA_MAX,
):
    os.makedirs(img_dir, exist_ok=True)

    # Collect raw samples per variable for optional graphing
    graph_samples = {}  # (v, split) -> raw_samples array

    # ------------------------------------------------------------------
    # PHASE 1: Scale and mark outliers (no imputation yet)
    # ------------------------------------------------------------------
    print("  Phase 1: scaling and marking outliers...")
    for v in tqdm(variables, desc="scale+outlier"):
        z_arr_store = os.path.join(save_dir, "train", f"{v}_x.zarr")
        scaler_store = os.path.join(save_dir, "scalers", f"{v}_scaler.pkl")
        z_arr = da.from_zarr(z_arr_store)
        z_arr = z_arr.rechunk({1: z_arr.shape[1]})
        orig_shape = z_arr.shape
        z_arr_flat = z_arr.reshape(-1, 1).compute()

        if graph:
            graph_samples[(v, "train")] = z_arr_flat.ravel().copy()

        scaler, scaled, col_mean, col_std = _scale_and_mark_outliers(
            z_arr_flat, v, scaler_mode
        )
        scaled = scaled.reshape(orig_shape)
        dump(scaler, open(scaler_store, "wb"))
        da.to_zarr(
            da.from_array(scaled),
            url=os.path.join(save_dir, "train", f"{v}_x_marked.zarr"),
        )

        # --- Test ---
        z_arr_store_test = os.path.join(save_dir, "test", f"{v}_x.zarr")
        z_arr_test = da.from_zarr(z_arr_store_test)
        orig_shape_test = z_arr_test.shape
        z_arr_test_flat = z_arr_test.reshape(-1, 1).compute()

        if graph:
            graph_samples[(v, "test")] = z_arr_test_flat.ravel().copy()

        if v == "spo2":
            z_arr_test_flat = 100.0 - z_arr_test_flat
        scaled_test = scaler.transform(z_arr_test_flat)
        outlier_mask_test = np.abs(z_arr_test_flat - col_mean) >= 3 * col_std
        scaled_test[outlier_mask_test] = np.nan
        scaled_test = scaled_test.reshape(orig_shape_test)
        da.to_zarr(
            da.from_array(scaled_test),
            url=os.path.join(save_dir, "test", f"{v}_x_marked.zarr"),
        )

    # ------------------------------------------------------------------
    # PHASE 2: Cross-variable window filtering
    # ------------------------------------------------------------------
    print("  Phase 2: filtering windows with too many outlier-NaNs...")
    keep_masks = {}
    for split in ["train", "test"]:
        drop = None
        n = None
        per_var_drops = {}
        for v in variables:
            arr = da.from_zarr(
                os.path.join(save_dir, split, f"{v}_x_marked.zarr")
            ).compute()
            if drop is None:
                n = arr.shape[0]
                drop = np.zeros(n, dtype=bool)
            nan_frac = np.isnan(arr).mean(axis=1)
            var_drop = nan_frac > na_threshold
            per_var_drops[v] = int(var_drop.sum())
            drop |= var_drop

        keep = ~drop
        keep_masks[split] = keep
        print(
            f"    {split}: dropping {drop.sum()}/{n} windows "
            f"({drop.mean():.1%})"
        )
        for v, cnt in per_var_drops.items():
            if cnt > 0:
                print(f"      {v}: {cnt} windows exceeded threshold")

        # Update labels
        labels_path = os.path.join(save_dir, split, "labels.pkl")
        labels = pd.read_pickle(labels_path)
        labels = labels[keep].reset_index(drop=True)
        labels.to_pickle(labels_path)

    # ------------------------------------------------------------------
    # PHASE 3: Filter, impute, and save final scaled arrays
    # ------------------------------------------------------------------
    print("  Phase 3: filtering, imputing, and saving...")
    for v in tqdm(variables, desc="impute+save"):
        for split in ["train", "test"]:
            marked_store = os.path.join(save_dir, split, f"{v}_x_marked.zarr")
            arr = da.from_zarr(marked_store).compute()
            arr = arr[keep_masks[split]]
            arr = da.from_array(arr)

            arr = da.map_blocks(
                lambda block: np.apply_along_axis(impute, axis=1, arr=block),
                arr,
                dtype=arr.dtype,
            )
            assert not da.isnan(arr).any().compute()
            da.to_zarr(
                arr, url=os.path.join(save_dir, split, f"{v}_x_scaled.zarr")
            )

            # Graph after imputation
            if graph:
                raw = graph_samples.get((v, split))
                if raw is not None:
                    _graph_norm_effect(
                        raw, arr.compute(), v,
                        "Train Set" if split == "train" else "Test Set",
                        img_dir,
                    )

            # Clean up intermediate
            shutil.rmtree(marked_store)

        # Clean up original unscaled zarr files
        for split in ["train", "test"]:
            orig_store = os.path.join(save_dir, split, f"{v}_x.zarr")
            if os.path.exists(orig_store):
                shutil.rmtree(orig_store)

    return None


def generate_final(save_dir, variables, transform=""):
    for s in ["train", "test"]:
        for i, v in tqdm(enumerate(variables), total=len(variables)):
            z_arr_store = os.path.join(save_dir, s, f"{transform}{v}_x_scaled.zarr")
            z_arr = da.from_zarr(z_arr_store)

            # concatenate
            if i == 0:
                base = z_arr
            else:
                base = da.concatenate([base, z_arr], axis=-1)

        da.to_zarr(
            base.rechunk({0: 10000, 1: -1}),
            url=os.path.join(save_dir, s, f"{transform}x.zarr"),
        )

    return None


def intersection_windows(variables, split_dict, temp_dir):
    for s, pts in split_dict.items():
        for p in pts:
            for i, v in enumerate(variables):
                labels = pd.read_pickle(os.path.join(temp_dir, v, p, "labels.pkl"))
                labels = labels.reset_index()
                labels = labels.rename(columns={"index": f"{v}_index"})
                # labels = labels.drop(columns="invalid")
                if i == 0:
                    combined = labels
                else:
                    combined = combined.merge(
                        labels, how="inner", on="datetime", validate="one_to_one"
                    )

            # save combined
            combined.to_pickle(os.path.join(temp_dir, f"{p}_combined_labels.pkl"))

    return None


def extract_data(ptid, v, temp_dir_path, config):
    """
    Extract data for a given patient ID and variable from the HDF5 file,
    process it into windows, and save the results to a temporary directory.

    Args:
        ptids (str): Patient ID.
        v (str): Variable to extract.

        temp_dir_path (str): Path to the temporary directory for saving results.
        config: Contains at least the following
            file_path (str): Path to the HDF5 file.
            window_size (int): Size of the window in seconds.
            mode (str): Mode for window extraction ('before', 'after', 'within', 'mean').
    Returns:
        None
    """
    file_path = config.data_file
    window_size = config.window_size
    mode = config.mode

    window_index, window_s = (
        get_window_index(mode, window_seconds=window_size),
        window_size,
    )
    if mode in ["mean", "smooth"]:
        config.strategy = mode
        config.percentage = 0.0
    else:
        config.strategy = "count"
        config.percentage = PERCENT_IN_MIN

    # extract windows for this patient and variable
    in_out, windows = get_windows_var(
        v,
        ptid,
        window_index,
        window_s,
        config,
    )

    broken_bool = in_out is None or len(in_out) == 0
    if broken_bool:
        print(f"In var {v}, invalid data for file:", ptid)
        return ptid

    w_vectors = np.stack([k["w"] for k in windows], axis=0)

    assert w_vectors.shape[0] == in_out.shape[0]

    # will filter out overlapping windows if stride functionality desired
    stride = config.stride
    if stride:
        # print(f"prior to striding, {len(in_out)} windows")
        in_out, w_vectors = stride_filter(
            labels=in_out, df=w_vectors, window_s=window_s
        )

        if len(in_out) <= 0:
            print(f"No labels with striding in {ptid} and var {v}")
        # print(f"after striding, {len(in_out)} windows")

    # save to a temp file as a zarr array
    temp_dir_path = os.path.join(temp_dir_path, ptid)
    os.makedirs(temp_dir_path, exist_ok=True)
    in_out.to_pickle(os.path.join(temp_dir_path, f"labels.pkl"))
    zarr.save(os.path.join(temp_dir_path, f"x.zarr"), w_vectors)

    return None


def finalize(variables, split_dict, save_dir):
    """
    Finalize the extracted data for a given variable, split into train and test.

    Args:
        v (str): Variable to finalize.
        split_dict (dict): Dictionary with keys being splits and values being list of ptids.
        temp_dir_path (str): Path to the temporary directory where intermediate results are stored.
        norm_dir (str): Path to the directory for saving normalization parameters.
    Returns:
        None
    """
    # build separate train and test dask arrays
    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")

    for s, ptids in split_dict.items():
        labels_all_store = os.path.join(save_dir, s, f"labels.pkl")

        # go through ptids and append to cumulative var arrays
        for v in variables:
            zarr_all_store = os.path.join(save_dir, s, f"{v}_x.zarr")
            print(f"Finalizing {v} for split {s}:")
            for i, p in tqdm(enumerate(ptids), total=len(ptids)):
                zarr_pt_store = os.path.join(save_dir, "temp", v, p, "x.zarr")
                labels_pt_store = os.path.join(temp_dir, f"{p}_combined_labels.pkl")

                labels_df = pd.read_pickle(labels_pt_store)

                # filter invalids out of label
                labels_df = labels_df[~labels_df.invalid].copy()

                # grab and filter
                z_arr = da.from_zarr(zarr_pt_store)
                z_arr = z_arr[labels_df[f"{v}_index"], :]
                if i == 0:
                    base = z_arr
                    combo_label = labels_df
                else:
                    base = da.concatenate([base, z_arr], axis=0)
                    combo_label = pd.concat(
                        [combo_label, labels_df],
                        axis=0,
                    ).reset_index(drop=True)

            da.to_zarr(base, url=zarr_all_store)

        combo_label.to_pickle(labels_all_store)

    return None


def downsample(variables, save_dir, strategy="mean", frequency=60):
    print("Downsampling:")
    for s in ["train", "test"]:
        # find minimum frequency
        min_points = 0
        for v in variables:
            zarr_all_store = os.path.join(save_dir, s, f"{v}_x.zarr")
            z_arr = da.from_zarr(zarr_all_store)
            if min_points == 0:
                min_points = z_arr.shape[1]
            elif z_arr.shape[1] < min_points:
                min_points = z_arr.shape[1]

        # downsample based on minimum frequency
        for v in variables:
            zarr_all_store = os.path.join(save_dir, s, f"{v}_x.zarr")
            z_arr = da.from_zarr(zarr_all_store)
            print(f"Downsampling variable {v} in split {s}, shape is {z_arr.shape}.")

            if z_arr.shape[1] != min_points:
                time_grid_mult = z_arr.shape[1] // min_points
                if strategy == "mean":
                    downsampled = da.nanmean(
                        da.reshape(z_arr, shape=(z_arr.shape[0], -1, time_grid_mult)),
                        axis=-1,
                    )
                elif strategy == "median":
                    downsampled = da.nanmedian(
                        da.reshape(z_arr, shape=(z_arr.shape[0], -1, time_grid_mult)),
                        axis=-1,
                    )
                print(f"Downsampled to {downsampled.shape}")
            else:
                downsampled = z_arr

            # if s == "test":
            #     print("Checking test dataset for problems:")
            #     do_tests(downsampled)

            # downsample even further if desired
            if frequency < 60:
                points_per_freq = 60 // frequency
                # Reshape to (N, timesteps, points_per_freq)
                downsampled = da.reshape(
                    downsampled, shape=(downsampled.shape[0], -1, points_per_freq)
                )

                if strategy == "mean":
                    downsampled = da.nanmean(downsampled, axis=-1)
                elif strategy == "median":
                    downsampled = da.nanmedian(downsampled, axis=-1)

                print(f"Further downsampled to {downsampled.shape}")
            da.to_zarr(downsampled, url=os.path.join(save_dir, s, f"{v}_x_ds.zarr"))
            shutil.rmtree(zarr_all_store)
            shutil.move(os.path.join(save_dir, s, f"{v}_x_ds.zarr"), zarr_all_store)

    return None


def do_pca(save_dir, z_arr_train, z_arr_test, variance=0.95):
    z_arr_train = z_arr_train.rechunk({1: z_arr_train.shape[1]})
    z_arr_test = z_arr_test.rechunk({1: z_arr_test.shape[1]})
    skinny_long = z_arr_train.shape[0] > z_arr_train.shape[1]

    if not skinny_long:
        pca = skPCA(n_components=variance)
        z_arr_train = z_arr_train.compute()
        z_arr_test = z_arr_test.compute()
    else:
        n_dim = np.min([z_arr_train.shape[0] - 1, z_arr_train.shape[1], 5_000])

        col_var = z_arr_train.var(axis=0).compute()
        print("- Num components fit:", n_dim)
        pca = PCA(n_components=n_dim)

    pca = pca.fit(z_arr_train)
    print("- Max var in PCA:", pca.explained_variance_ratio_.cumsum()[-1])

    if not skinny_long:
        X_train = pca.transform(z_arr_train)
        X_test = pca.transform(z_arr_test)
        print(
            f"- Not skinny long matrix, using {X_train.shape[1]} dims for target variance {variance*100:0.0f}%."
        )
        X_train = da.from_array(X_train)
        X_test = da.from_array(X_test)
    else:
        selected_dim = (
            np.arange(n_dim)[pca.explained_variance_ratio_.cumsum() > variance][0] + 1
        )
        print(
            f"- Using {selected_dim} dimensions at threshold variance of {variance*100:0.0f}%."
        )
        X_train = pca.transform(z_arr_train)[:, :selected_dim]
        X_test = pca.transform(z_arr_test)[:, :selected_dim]

    return X_train, X_test


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--data_file",
        help="Path to processed data HDF5 file.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/all_data.hdf5",
    )
    parser.add_argument(
        "--top_dir",
        help="Directory to save extracted data.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/training",
    )
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=60 * 5,
        help="Window size in seconds to extract values, default is 5 minutes.",
    )
    parser.add_argument(
        "--variables",
        "-v",
        nargs="+",
        default=FEATURES,
        help="List of variables to include in model data. Default is all features.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Does not delete temporary dir for debugging purposes.",
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--match_grid",
        type=int,
        default=0,
        help="Will downsample all variables to match the sampling grid of the lowest frequency variable. 0: none, 1: downsample with mean, 2: downsample with median.",
    )
    parser.add_argument(
        "-o",
        "--overwrite_permanent",
        help="Will overwrite the permanent directory.",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--transforms",
        nargs="+",
        choices=["pca", "separate_pca", "fpca", "multivar_fpca", "none", "chronos"],
        help="What kind of downstream transforms to do and save.",
        default=["pca"],
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["mean", "smooth"],
        help="Mode of window extraction.",
        default="mean",
    )
    parser.add_argument(
        "-s",
        "--scaler",
        choices=["standard", "robust"],
        help="Scaler to use.",
        default="robust",
    )
    parser.add_argument(
        "-x",
        "--variance",
        type=int,
        help="Percentage variance to get pcas for.",
        default=95,
    )
    parser.add_argument(
        "--frequency",
        "-f",
        type=int,
        default=60,
        help="Number of datapoints per minute, works only after dataset is downsampled to 1 Hz and frequency is factor of 60.",
    )
    parser.add_argument(
        "--smooth_frac",
        "-sf",
        type=float,
        default=SMOOTH_FRAC_OUT_MIN,
        help="Minimum fraction of labels to be out to label window as out if smoothing.",
    )
    parser.add_argument(
        "--r2_threshold",
        "-r",
        type=float,
        default=-1e2,
        help="Minimum R2 value for labels to be considered good.",
    )
    parser.add_argument(
        "--stride",
        "-z",
        help="Non-overlapping windows.",
        action="store_true",
    )
    parser.add_argument(
        "--chop",
        "-c",
        help="What length of time from first limits calculation to cut recording at.",
        type=int,
        default=-1,
    )

    args = parser.parse_args()
    config = args
    np.random.seed(420)
    random.seed(420)
    pd.options.display.float_format = "{:.2f}".format

    assert 60 % args.frequency == 0, "Frequency needs to be a factor of 60."

    dataset_name = f"w_{args.window_size}s_{'_'.join(args.variables)}"
    if args.match_grid != 0:
        dataset_name = f"downsample{args.match_grid}_" + dataset_name
    if args.mode == "smooth":
        dataset_name = f"smooth{args.smooth_frac:.2f}_" + dataset_name
    if args.scaler == "robust":
        dataset_name = "robust_" + dataset_name
    if args.variance != 95:
        dataset_name = f"var{args.variance}_" + dataset_name
    if args.frequency < 60:
        dataset_name = f"freq{args.frequency}_" + dataset_name
    if args.r2_threshold > 0.0:
        dataset_name = f"{args.r2_threshold:.2f}r2_" + dataset_name
    if args.chop > 0:
        dataset_name = f"chop{args.chop}_" + dataset_name
    print(f"DATASET = {os.path.join(args.top_dir, dataset_name)}")

    print(
        f"Dataset creation with window size {args.window_size}s for variables: {args.variables}."
    )

    # make test, train and temp directories, prepare for saving
    save_dir = os.path.join(args.top_dir, dataset_name, "train_data")
    # overwrite save dir if it exists
    if os.path.exists(save_dir):
        print("Overwriting existing save directory.")
        shutil.rmtree(save_dir)

    # make new dirs
    for split in ["train", "test", "temp", "scalers"]:
        os.makedirs(os.path.join(save_dir, split))

    temp_dir = os.path.join(save_dir, "temp")
    norm_dir = os.path.join(save_dir, "scalers")
    split_dict = {"train": [], "test": []}

    # need healthy ptids and dict of which split they fall in
    with h5py.File(args.data_file, "r") as f:
        ptids = f["healthy_ptids"][:].astype(str).tolist()
        # for debugging
        rng = np.random.default_rng(2323)
        idx = rng.choice(len(ptids), size=20, replace=False).tolist()
        ptids = [ptids[i] for i in idx] if args.debug else ptids
        for p in ptids:
            if f[p].attrs["split"] == "train":
                split_dict["train"].append(p)
            else:
                split_dict["test"].append(p)

    # will do everything and write to file in temp_dir
    ctx = mp.get_context("spawn")
    for var in args.variables:
        print(f"Extracting for variable {var}:")
        os.makedirs(os.path.join(temp_dir, var))
        func = partial(
            extract_data,
            v=var,
            temp_dir_path=os.path.join(temp_dir, var),
            config=config,
        )

        results = process_map(
            func,
            ptids,
            max_workers=1 if args.debug else os.cpu_count(),
            chunksize=1,
        )

        # remove invalid patients from split dict
        broken_pts = [r for r in results if r is not None]
        for bp in broken_pts:
            for s in split_dict.keys():
                if bp in split_dict[s]:
                    split_dict[s].remove(bp)

    # check labels
    intersection_windows(args.variables, split_dict, temp_dir)

    # finalizing
    finalize(args.variables, split_dict, save_dir)

    # downsample if we want to match the time grid
    if args.match_grid == 1:
        downsample(args.variables, save_dir, strategy="mean", frequency=args.frequency)
    elif args.match_grid == 2:
        downsample(
            args.variables, save_dir, strategy="median", frequency=args.frequency
        )

    # delete temp_dir
    if not args.debug:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("")

    # preprocess dataset
    # normalize, remove 3 SD values, and impute
    print("Normalizing, removing outliers, and imputing missing values:")
    normalize(save_dir, args.variables, scaler_mode=args.scaler)

    # generates final base dataset
    print("\nGenerating whole dataset:")
    generate_final(save_dir, args.variables)
    data_string = "x.zarr"

    base_arr_train = da.from_zarr(os.path.join(save_dir, "train", data_string))
    base_arr_test = da.from_zarr(os.path.join(save_dir, "test", data_string))
    print(
        f"Train and test datasets generated adequately. {base_arr_train.shape[0]} windows in train and {base_arr_test.shape[0]} in test with {base_arr_train.shape[1]} dimensions."
    )
    print("Checking train dataset for problems:")
    do_tests(base_arr_train)
    print("Checking test dataset for problems:")
    do_tests(base_arr_test)

    # basically adds itself between scaling and concatenation
    if "separate_pca" in args.transforms:
        for v in args.variables:
            z_arr_train = da.from_zarr(
                os.path.join(save_dir, "train", f"{v}_x_scaled.zarr")
            )
            z_arr_test = da.from_zarr(
                os.path.join(save_dir, "test", f"{v}_x_scaled.zarr")
            )

            print(f"Doing PCA for var {v}:")
            X_train, X_test = do_pca(
                save_dir, z_arr_train, z_arr_test, variance=args.variance / 100
            )

            da.to_zarr(
                X_train.rechunk({0: 10000, 1: -1}),
                url=os.path.join(
                    save_dir, "train", f"separate_decomp_{v}_x_scaled.zarr"
                ),
                overwrite=True,
            )
            da.to_zarr(
                X_test.rechunk({0: 10000, 1: -1}),
                url=os.path.join(
                    save_dir, "test", f"separate_decomp_{v}_x_scaled.zarr"
                ),
                overwrite=True,
            )

        print("Done.")

        print("\nGenerating separate PCA dataset:")
        generate_final(save_dir, args.variables, transform="separate_decomp_")
        data_string = "separate_decomp_x.zarr"

        z_arr_train = da.from_zarr(os.path.join(save_dir, "train", data_string))
        z_arr_test = da.from_zarr(os.path.join(save_dir, "test", data_string))
        print(
            f"Train and test datasets generated adequately with separate PCAs. {z_arr_train.shape[0]} windows in train and {z_arr_test.shape[0]} in test with {z_arr_train.shape[1]} dimensions."
        )

    if "pca" in args.transforms:
        print("Doing global PCA")
        X_train, X_test = do_pca(
            save_dir, base_arr_train, base_arr_test, variance=args.variance / 100
        )
        print(
            f"Train and test datasets generated adequately with combined PCAs. {X_train.shape[0]} windows in train and {X_test.shape[0]} in test with {X_train.shape[1]} dimensions."
        )
        da.to_zarr(
            X_train.rechunk({0: 10000, 1: -1}),
            url=os.path.join(save_dir, "train", f"pca_x.zarr"),
        )
        da.to_zarr(
            X_test.rechunk({0: 10000, 1: -1}),
            url=os.path.join(save_dir, "test", f"pca_x.zarr"),
        )

        print("Done.")

    # clean up in scaled
    for s in ["train", "test"]:
        for f in os.listdir(os.path.join(save_dir, s)):
            if "_scaled" in f:
                shutil.rmtree(os.path.join(save_dir, s, f))

    if "chronos" in args.transforms and args.match_grid >= 0:
        from chronos import Chronos2Pipeline
        import torch

        assert torch.cuda.is_available(), "Chronos transform requires a GPU."
        print("Generating Chronos embeddings:")

        pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map="cuda"
        )
        batch_size = 1024

        for split in ["train", "test"]:
            chronos_path = os.path.join(save_dir, split, "chronos_x.zarr")
            s = da.from_zarr(os.path.join(save_dir, split, data_string))
            z = zarr.open(
                chronos_path,
                mode="w",
                shape=(s.shape[0], pipeline.model.model_dim),
                dtype="float32",
            )

            n_batches = s.shape[0] // batch_size + 1
            for i in tqdm(range(n_batches)):
                end = min((i + 1) * batch_size, s.shape[0])
                x_input = s[i * batch_size : end, :].compute()
                x_input = x_input.reshape(x_input.shape[0], len(args.variables), -1)
                x_tensor = torch.tensor(x_input).float()
                out = [
                    np.nanmean(i.cpu().numpy(), axis=(0, 1))
                    for i in pipeline.embed(inputs=x_tensor)[0]
                ]
                out_embed = np.vstack(out)

                # needs to be saved back as zarr, can't concatenate into memory
                z[i * batch_size : end, :] = out_embed

    if (
        os.path.exists(os.path.join(args.top_dir, dataset_name, "permanent"))
        and not args.overwrite_permanent
    ):
        print(
            "WARNING: not overwriting permanent to avoid data chaos, current run in train_data. The latter will get overwritten if run again"
        )
    elif not os.path.exists(os.path.join(args.top_dir, dataset_name, "permanent")):
        shutil.move(save_dir, os.path.join(args.top_dir, dataset_name, "permanent"))
    else:
        # if overwriting, check if file names are same (then delete old and overwrite) or different (then just add new)
        for split in ["train", "test"]:
            permanent_path = os.path.join(
                args.top_dir, dataset_name, "permanent", split
            )
            new_path = os.path.join(save_dir, split)
            for f in os.listdir(new_path):
                if f in os.listdir(permanent_path):
                    if os.path.isdir(os.path.join(permanent_path, f)):
                        shutil.rmtree(os.path.join(permanent_path, f))
                    else:
                        os.remove(os.path.join(permanent_path, f))
                shutil.move(os.path.join(new_path, f), os.path.join(permanent_path, f))

        # overwrite scalers
        perm_scalers = os.path.join(args.top_dir, dataset_name, "permanent", "scalers")
        shutil.rmtree(perm_scalers)
        shutil.move(os.path.join(save_dir, "scalers"), perm_scalers)

        # delete copying dir once done overwriting
        shutil.rmtree(save_dir)

    X_train = da.from_zarr(
        os.path.join(args.top_dir, dataset_name, "permanent", "train", "x.zarr")
    )
    labels = pd.read_pickle(
        os.path.join(args.top_dir, dataset_name, "permanent", "train", "labels.pkl")
    )
    assert X_train.shape[0] == labels.shape[0]
    y_train = labels["in?"].astype(int)
    groups = labels["ptid"].astype(str)

    y_train = y_train.dropna()
    print(
        f"{y_train.sum() / y_train.shape[0] * 100:0.1f}% of training windows are inside AR limits for mode {args.mode}."
    )
    print("Dataset creation complete.")
    print(f"DATASET_NAME={os.path.join(args.top_dir, dataset_name)}")
    print(f"Random seed {np.random.default_rng().integers(0, 1e6)}")
    print("")
