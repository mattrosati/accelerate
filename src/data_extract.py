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

from data_utils import build_continuous_time, load_label
from constants import *
from process_utils import *

from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA

from scipy.stats import shapiro
from sklearn.preprocessing import PowerTransformer


def normalize(save_dir, variables, img_dir="/home/mr2238/project_pi_np442/mr2238/accelerate/imgs/normalize"):
    n_samples = 100_000
    os.makedirs(img_dir, exist_ok=True)
    for v in tqdm(variables):
        z_arr_store = os.path.join(save_dir, "train", f"{v}_x.zarr")
        scaler_store = os.path.join(save_dir, "scalers", f"{v}_scaler.pkl")
        z_arr = da.from_zarr(z_arr_store)

        # print Shapiro-Wilk Test for unnormalized
        z_arr = z_arr.rechunk({1: z_arr.shape[1]})
        sampled_data_pre = da.random.choice(da.ravel(z_arr), size=n_samples, replace=False).compute()

        if v == 'spo2':
            # flip tail, then box cox normalization
            scaler = PowerTransformer()
            z_arr = 1.0 - z_arr

            scaled_values = scaler.fit_transform(z_arr)
            scaled_values = da.from_array(scaled_values, chunks=z_arr.chunks)
        else:
            # for others, do classic normalization
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(z_arr)

        dump(scaler, open(scaler_store, 'wb'))
        da.to_zarr(scaled_values, url=os.path.join(save_dir, "train", f"{v}_x_scaled.zarr"))

        sampled_data_post = da.random.choice(da.ravel(scaled_values), size=n_samples, replace=False).compute()

        # graph effect of vars
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.histplot(sampled_data_pre, ax=ax, stat="probability", label="Before Norm", edgecolor=(0, 0, 0, 0.5), alpha=0.5)
        sns.histplot(sampled_data_post, ax=ax, stat="probability", label="After Norm", edgecolor=(0, 0, 0, 0.5), alpha=0.5)
        ax.set_xlabel(f"{v}")
        plt.title(f"Effect of normalization on {v}")
        ax.legend()
        img_name = f"{v}_norm_effect.png"
        plt.savefig(os.path.join(img_dir, img_name))
        plt.close()


        # transform test set
        z_arr_store_test = os.path.join(save_dir, "test", f"{v}_x.zarr")
        z_arr_test = da.from_zarr(z_arr_store_test)
        z_arr_test = z_arr_test.rechunk({1: z_arr_test.shape[1]})
        scaled_values = scaler.transform(z_arr_test)
        da.to_zarr(scaled_values, url=os.path.join(save_dir, "test", f"{v}_x_scaled.zarr"))

        # clean up
        shutil.rmtree(z_arr_store)
        shutil.rmtree(z_arr_store_test)

    return None



def generate_final(save_dir, variables):
    for s in ["train", "test"]:
        for i, v in tqdm(enumerate(variables), total=len(variables)):
            if v != 'spo2':
                z_arr_store = os.path.join(save_dir, s, f"{v}_x_scaled.zarr")
            else:
                z_arr_store = os.path.join(save_dir, s, f"{v}_x.zarr")
            z_arr = da.from_zarr(z_arr_store)

            if i == 0:
                base = z_arr
            else:
                base = da.concatenate([base, z_arr], axis=1)
            
        da.to_zarr(base, url=os.path.join(save_dir, s, f"x.zarr"))

        # clean up
        for f in os.listdir(os.path.join(save_dir, s)):
            if "_x" in f:
                shutil.rmtree(os.path.join(save_dir, s, f))
        

    return None


def intersection_windows(variables, split_dict, temp_dir):
    for s, pts in split_dict.items():
        for p in pts:
            for i, v in enumerate(variables):
                labels = pd.read_pickle(os.path.join(temp_dir, v, p, "labels.pkl"))
                labels = labels.reset_index()
                labels = labels.rename(columns={"index": f"{v}_index"})
                if i == 0:
                    combined = labels
                else:
                    combined = combined.merge(labels, how="inner", on="datetime")

                # save combined
                combined.to_pickle(os.path.join(temp_dir, f"{p}_combined_labels.pkl"))

    return None


def extract_data(ptid, v, file_path, temp_dir_path, window_size, mode="mean"):
    """
    Extract data for a given patient ID and variable from the HDF5 file,
    process it into windows, and save the results to a temporary directory.

    Args:
        ptids (str): Patient ID.
        v (str): Variable to extract.
        file_path (str): Path to the HDF5 file.
        temp_dir_path (str): Path to the temporary directory for saving results.
        window_size (int): Size of the window in seconds.
        mode (str): Mode for window extraction ('before', 'after', 'within', 'mean').
    Returns:
        None
    """
    window_index, window_s = (
        get_window_index(mode, window_seconds=window_size),
        window_size,
    )
    strategy = "mean" if mode == "mean" else "count"
    percentage = 0.0 if strategy == "mean" else PERCENT_IN_MIN

    # extract windows for this patient and variable
    in_out, windows = get_windows_var(
        v, ptid, file_path, window_index, window_s, strategy, percentage
    )

    broken_bool = in_out is None or len(in_out) == 0
    if broken_bool:
        print(f"In var {v}, invalid data for file:", ptid)
        return ptid

    w_vectors = np.stack([k["w"] for k in windows], axis=0)

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

                # grab and filter
                z_arr = da.from_zarr(zarr_pt_store)
                z_arr = z_arr[labels_df[f"{v}_index"], :]
                if i == 0:
                    base = z_arr
                    combo_label = labels_df
                else:
                    base = da.concatenate([base, z_arr], axis=0)
                    labels_df = pd.concat(
                        [combo_label, pd.read_pickle(labels_pt_store)],
                        axis=0,
                    ).reset_index(drop=True)

            da.to_zarr(base, url=zarr_all_store)

        labels_df.to_pickle(labels_all_store)

    return None


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--data_file",
        help="Path to processed data HDF5 file.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/all_data.hdf5",
    )
    parser.add_argument(
        "--save_dir",
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

    args = parser.parse_args()
    np.random.seed(420)
    dataset_name = f"w_{args.window_size}s_{'_'.join(args.variables)}"

    print(
        f"Dataset creation with window size {args.window_size}s for variables: {args.variables}."
    )

    # make test, train and temp directories, prepare for saving
    save_dir = os.path.join(args.save_dir, dataset_name)
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
        ptids = ptids
        for p in ptids:
            if f[p].attrs["split"] == "train":
                split_dict["train"].append(p)
            else:
                split_dict["test"].append(p)

    # will do everything and write to file in temp_dir
    for var in args.variables:
        print(f"Extracting for variable {var}:")
        os.makedirs(os.path.join(temp_dir, var))
        func = partial(
            extract_data,
            v=var,
            file_path=args.data_file,
            temp_dir_path=os.path.join(temp_dir, var),
            window_size=args.window_size,
        )
        results = process_map(func, ptids, max_workers=os.cpu_count(), chunksize=1)

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

    # delete temp_dir
    if not args.debug:
        shutil.rmtree(temp_dir)


    print("")

    # preprocess dataset
    # normalize (imputation already done)
    print("Normalizing:")
    normalize(save_dir, args.variables)

    print("Generating whole dataset:")
    generate_final(save_dir, args.variables)

    z_arr_train = da.from_zarr(os.path.join(save_dir, 'train', 'x.zarr'))
    z_arr_test = da.from_zarr(os.path.join(save_dir, 'test', 'x.zarr'))
    print(f"Train and test datasets generated adequately. {z_arr_train.shape[0]} windows in train and {z_arr_test.shape[0]} in test with {z_arr_train.shape[1]} dimensions.")


    print("Doing PCAs:")
    n_dim = 1_000
    pca = PCA(n_components=n_dim)
    pca = pca.fit(z_arr_train)
    selected_dim = (
        np.arange(n_dim)[pca.explained_variance_ratio_.cumsum() > 0.95][0] + 1
    )
    print(f"Using {selected_dim} dimensions at threshold variance of 0.95.")
    X_train = pca.transform(z_arr_train)[:, :selected_dim]
    X_test = pca.transform(z_arr_test)[:, :selected_dim]
    da.to_zarr(X_train, url=os.path.join(save_dir, "train", f"pca_x.zarr"))
    da.to_zarr(X_test, url=os.path.join(save_dir, "test", f"pca_x.zarr"))

    print("Done.")
