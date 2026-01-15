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


def normalize(
    save_dir,
    variables,
    img_dir="/home/mr2238/project_pi_np442/mr2238/accelerate/imgs/normalize_impute_rs_new",
    graph=False,
    scaler=None,
):
    n_samples = 100_000
    os.makedirs(img_dir, exist_ok=True)
    for v in tqdm(variables):
        z_arr_store = os.path.join(save_dir, "train", f"{v}_x.zarr")
        scaler_store = os.path.join(save_dir, "scalers", f"{v}_scaler.pkl")
        z_arr = da.from_zarr(z_arr_store)

        # find the scales
        z_arr = z_arr.rechunk({1: z_arr.shape[1]})
        if graph:
            sampled_data_pre = da.random.choice(
                da.ravel(z_arr), size=n_samples, replace=False
            ).compute()

        orig_shape = z_arr.shape
        z_arr = z_arr.reshape(-1, 1)
        if scaler == "robust":
            if v == "spo2":
                # flip tail
                z_arr = 100.0 - z_arr
            scaler = RobustScaler(quantile_range=(10.0, 90.0))
            scaled_values = scaler.fit_transform(z_arr)
        else:
            if v == "spo2":
                # flip tail, then box cox normalization
                scaler = PowerTransformer()
                z_arr = 100.0 - z_arr

                scaled_values = scaler.fit_transform(z_arr.compute())
                scaled_values = da.from_array(scaled_values).rechunk(
                    {1: scaled_values.shape[1]}
                )
            else:
                # for others, do classic normalization
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(z_arr)

        scaled_values = scaled_values.reshape(orig_shape)
        dump(scaler, open(scaler_store, "wb"))
        da.to_zarr(
            scaled_values, url=os.path.join(save_dir, "train", f"{v}_x_scaled.zarr")
        )

        # graph effect of norm
        if graph:
            sampled_data_post = da.random.choice(
                da.ravel(scaled_values), size=n_samples, replace=False
            ).compute()

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

            # --- Left plot: Before normalization
            sns.histplot(
                sampled_data_pre,
                ax=axes[0],
                stat="probability",
                edgecolor=(0, 0, 0, 0.5),
                alpha=0.5,
            )
            axes[0].set_title("Before Normalization")
            axes[0].set_xlabel(f"{v}")

            # --- Right plot: After normalization
            sns.histplot(
                sampled_data_post,
                ax=axes[1],
                stat="probability",
                edgecolor=(0, 0, 0, 0.5),
                alpha=0.5,
            )
            axes[1].set_title("After Normalization")
            axes[1].set_xlabel(f"{v}")

            fig.suptitle(f"Effect of normalization on {v} (Train Set)")

            img_name = f"{v}_norm_effect_train.png"
            plt.savefig(os.path.join(img_dir, img_name))
            plt.close()

        # transform test set
        z_arr_store_test = os.path.join(save_dir, "test", f"{v}_x.zarr")
        z_arr_test = da.from_zarr(z_arr_store_test)
        orig_shape = z_arr_test.shape
        z_arr_test = z_arr_test.reshape(-1, 1)

        if scaler == "robust":
            if v == "spo2":
                z_arr_test = 100.0 - z_arr_test
            z_arr_test = z_arr_test.rechunk({1: z_arr_test.shape[1]})
            scaled_values = scaler.transform(z_arr_test)
        else:
            if v == "spo2":
                z_arr_test = 100.0 - z_arr_test
                scaled_values = scaler.transform(z_arr_test.compute())
                scaled_values = da.from_array(scaled_values).rechunk(
                    {1: scaled_values.shape[1]}
                )
            else:
                z_arr_test = z_arr_test.rechunk({1: z_arr_test.shape[1]})
                scaled_values = scaler.transform(z_arr_test)
        scaled_values = scaled_values.reshape(orig_shape)
        da.to_zarr(
            scaled_values, url=os.path.join(save_dir, "test", f"{v}_x_scaled.zarr")
        )

        # graph effect of norm
        if graph:
            sampled_data_post = da.random.choice(
                da.ravel(scaled_values), size=n_samples, replace=False
            ).compute()

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            # --- Left plot: Before normalization
            sns.histplot(
                sampled_data_pre,
                ax=axes[0],
                stat="probability",
                edgecolor=(0, 0, 0, 0.5),
                alpha=0.5,
            )
            axes[0].set_title("Before Normalization")
            axes[0].set_xlabel(f"{v}")

            # --- Right plot: After normalization
            sns.histplot(
                sampled_data_post,
                ax=axes[1],
                stat="probability",
                edgecolor=(0, 0, 0, 0.5),
                alpha=0.5,
            )
            axes[1].set_title("After Normalization")
            axes[1].set_xlabel(f"{v}")

            fig.suptitle(f"Effect of normalization on {v} (Test Set)")

            img_name = f"{v}_norm_effect_test.png"
            plt.savefig(os.path.join(img_dir, img_name))
            plt.close()

        # clean up
        shutil.rmtree(z_arr_store)
        shutil.rmtree(z_arr_store_test)

    return None


def generate_final(save_dir, variables, transform=""):
    for s in ["train", "test"]:
        for i, v in tqdm(enumerate(variables), total=len(variables)):
            z_arr_store = os.path.join(save_dir, s, f"{transform}{v}_x_scaled.zarr")
            z_arr = da.from_zarr(z_arr_store)

            if i == 0:
                base = z_arr
            else:
                base = da.concatenate([base, z_arr], axis=-1)

        da.to_zarr(base, url=os.path.join(save_dir, s, f"{transform}x.zarr"))

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
                    combined = combined.merge(labels, how="inner", on="datetime")

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
                    downsampled = da.reshape(
                        z_arr, shape=(z_arr.shape[0], -1, time_grid_mult)
                    ).mean(axis=-1)
                elif strategy == "median":
                    downsampled = da.median(
                        da.reshape(z_arr, shape=(z_arr.shape[0], -1, time_grid_mult)),
                        axis=-1,
                    )
                print(f"Downsampled to {downsampled.shape}")
            else:
                downsampled = z_arr

            # downsample even further if desired
            if frequency < 60:
                points_per_freq = 60 // frequency
                # Reshape to (N, timesteps, points_per_freq)
                downsampled = da.reshape(
                    downsampled, shape=(downsampled.shape[0], -1, points_per_freq)
                )

                if strategy == "mean":
                    downsampled = downsampled.mean(axis=-1)
                elif strategy == "median":
                    downsampled = da.median(downsampled, axis=-1)

                print(f"Further downsampled to {downsampled.shape}")
            da.to_zarr(downsampled, url=os.path.join(save_dir, s, f"{v}_x_ds.zarr"))
            shutil.rmtree(zarr_all_store)
            shutil.move(os.path.join(save_dir, s, f"{v}_x_ds.zarr"), zarr_all_store)

    return None


def do_pca(save_dir, z_arr_train, z_arr_test, variance=0.95):
    z_arr_train = z_arr_train.rechunk({1: z_arr_train.shape[1]})
    z_arr_test = z_arr_test.rechunk({1: z_arr_test.shape[1]})
    n_dim = np.min([z_arr_train.shape[0], z_arr_train.shape[1], 5_000])

    col_var = z_arr_train.var(axis=0).compute()
    pca = PCA(n_components=n_dim)
    pca = pca.fit(z_arr_train)
    print("- Max var in PCA:", pca.explained_variance_ratio_.cumsum()[-1])
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
    # parser.add_argument(
    #     "--stride",
    #     "-z",
    #     help="Non-overlapping windows.",
    #     action="store_true",
    # )

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
            func, ptids, max_workers=1 if args.debug else os.cpu_count(), chunksize=1
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
    # normalize (imputation already done)
    print("Normalizing:")
    normalize(save_dir, args.variables, scaler=args.scaler)

    # generates final base dataset
    print("\nGenerating whole dataset:")
    generate_final(save_dir, args.variables)
    data_string = "x.zarr"

    base_arr_train = da.from_zarr(os.path.join(save_dir, "train", data_string))
    base_arr_test = da.from_zarr(os.path.join(save_dir, "test", data_string))
    print(
        f"Train and test datasets generated adequately. {base_arr_train.shape[0]} windows in train and {base_arr_test.shape[0]} in test with {base_arr_train.shape[1]} dimensions."
    )

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
                X_train,
                url=os.path.join(
                    save_dir, "train", f"separate_decomp_{v}_x_scaled.zarr"
                ),
                overwrite=True,
            )
            da.to_zarr(
                X_test,
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
        da.to_zarr(X_train, url=os.path.join(save_dir, "train", f"pca_x.zarr"))
        da.to_zarr(X_test, url=os.path.join(save_dir, "test", f"pca_x.zarr"))

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
            print(z.shape)

            n_batches = s.shape[0] // batch_size + 1
            for i in tqdm(range(n_batches)):
                end = min((i + 1) * batch_size, s.shape[0])
                x_input = s[i * batch_size : end, :].compute()
                x_input = x_input.reshape(x_input.shape[0], len(args.variables), -1)
                x_tensor = torch.tensor(x_input).float()
                out = [
                    i.cpu().numpy().mean(axis=(0, 1))
                    for i in pipeline.embed(inputs=x_tensor)[0]
                ]
                out_embed = np.vstack(out)

                # needs to be saved back as zarr, can't concatenate into memory
                z[i * batch_size : end, :] = out_embed

            # print(z.mean().compute())

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
