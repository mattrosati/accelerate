import os
import sys
import random
import pickle
from argparse import ArgumentParser

import h5py

import numpy as np
import zarr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import umap
import itertools

from tqdm import tqdm

from data_utils import *
from constants import *
from process_utils import impute

# pending useful sklearn imports
from scipy.stats import pearsonr

import dask
import dask.array as da
import dask.dataframe as dd

from dask_ml.impute import SimpleImputer
from dask_ml.decomposition import PCA

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--do_big",
        action="store_true",
        help="Whether to do the big umap or the pca+umap",
    )
    args = parser.parse_args()

    pd.options.display.float_format = "{:10,.2f}".format
    np.random.seed(420)
    random.seed(420)
    # path constants
    data_dir = "/home/mr2238/project_pi_np442/mr2238/accelerate/data"
    img_dir = "/home/mr2238/project_pi_np442/mr2238/accelerate/imgs/in_out"
    global_path = (
        "/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/all_data.hdf5"
    )
    labels_path = os.path.join(data_dir, "labels")
    raw_data_path = os.path.join(data_dir, "raw_data")

    # make combined info df
    information = []
    mode = "mean"
    with h5py.File(global_path, "r") as f:
        for pt in np.array(f["healthy_ptids"][...]).astype(str):
            if f[f"{pt}/processed/in_out_{mode}"].attrs["in_out_broken"]:
                continue
            in_out_df = pd.Series(
                f[f"{pt}/processed/in_out_{mode}/in_out"][...]
            ).astype(bool)
            idx_window = pd.DataFrame(
                f[f"{pt}/processed/in_out_{mode}/window_idx"][...]
            )
            tmp = pd.concat([in_out_df, idx_window], axis=1)
            tmp.columns = ["in?", "startidx", "endidx"]
            tmp["ptid"] = pt
            information.append(tmp)
    info_df = pd.concat(information).reset_index(drop=True)
    # check if windows have unequal length (they should not)
    info_df["len"] = info_df["endidx"] - info_df["startidx"]
    assert len(info_df["len"].unique()) == 1

    os.makedirs(
        "/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/windows/",
        exist_ok=True,
    )
    zarr_store = "/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/windows/array.zarr"
    if not os.path.exists(zarr_store):
        print("Making windows...")
        # make windows zarr store

        big_arr_path = "/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/windows/memmap.dat"

        # save all windows as zarr file
        # saving all windows into a numpy memmap file
        windows = np.memmap(
            big_arr_path,
            dtype="float32",
            mode="w+",
            shape=[info_df.shape[0], info_df["len"][0] + 1],
        )
        with h5py.File(global_path, "r") as f:
            nan_value = f.attrs["invalid_val"]
            uniques = info_df.ptid.unique()
            for pt in tqdm(uniques):
                abp_arr = f[f"{pt}/raw/waves/abp"]
                ptid_info = info_df[info_df["ptid"] == pt]
                for w, row in ptid_info.iterrows():
                    abp_arr.read_direct(
                        windows,
                        source_sel=np.s_[row["startidx"] : row["endidx"]],
                        dest_sel=np.s_[w, 1:],
                    )
                    windows[w, 0] = w
                    windows[w, windows[w] == nan_value] = np.nan
                    windows[w, 1:] = impute(windows[w, 1:], strategy="lin_interpolate")
                # print(windows[w, :])

        windows.flush()

        # need to transfer windows memmap to zarr
        zarr.save_array(zarr_store, windows, chunks=(5000, 7501))

        del windows
        os.remove(big_arr_path)

    wind_dask = da.from_zarr(zarr_store)

    # manual scaling
    mean_all = da.nanmean(wind_dask[:, 1:]).compute()
    std_all = da.nanstd(wind_dask[:, 1:]).compute()
    scaled_values = (wind_dask[:, 1:] - mean_all) / std_all

    # do umap
    if args.do_big:
        print("Doing large UMAP....")
        fit = umap.UMAP(verbose=True)
        u = fit.fit_transform(scaled_values)
        u_df = pd.DataFrame(u, columns=["dim_1", "dim_2"])
        index_w = wind_dask[:, 0].compute()
        u_df = u_df.set_index(index_w)
        info_df[["dim_1", "dim_2"]] = u_df
        info_df.to_pickle(
            "/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/windows/large_umap.pkl"
        )
        print("Done")
        del fit
    else:
        print("Doing PCA")
        n_dim = 600
        pca = PCA(n_components=n_dim)
        pca = pca.fit(scaled_values)
        umap_dim = (
            np.arange(n_dim)[pca.explained_variance_ratio_.cumsum() > 0.95][0] + 1
        )
        print(f"Using {umap_dim} dimensions for UMAP at threshold variance of 0.95.")
        X = pca.transform(scaled_values)[:, :umap_dim]
        print("Done.")

        print("Doing UMAP grid search...")
        # umap and graph with changes in basic parameters
        index_w = wind_dask[:, 0].compute()
        for n_neighbors, min_dist in itertools.product([15, 50], [0.1, 0.25, 0.5]):
            print(n_neighbors, min_dist)
            fit2 = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, verbose=True)
            u2 = fit2.fit_transform(X)
            u2_df = pd.DataFrame(u2, columns=["dim_1b", "dim_2b"])
            u2_df = u2_df.set_index(index_w)
            # combine datasets:
            info_df[
                [f"dim_1_{n_neighbors}_{min_dist}", f"dim_2_{n_neighbors}_{min_dist}"]
            ] = u2_df

        info_df.to_pickle(
            f"/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/windows/total_umaps2.pkl"
        )

        print("Done.")
