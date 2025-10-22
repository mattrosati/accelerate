import os
import sys
import random
import pickle

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

# pending useful sklearn imports
from scipy.stats import pearsonr

import dask
import dask.array as da
import dask.dataframe as dd

from dask_ml.impute import SimpleImputer
from dask_ml.decomposition import PCA

if __name__ == "__main__":
    pd.options.display.float_format = "{:10,.2f}".format
    plt.rcParams.update({"xtick.bottom": True, "ytick.left": True})
    np.random.seed(420)
    random.seed(420)
    sns.set_theme(context="talk")
    # path constants
    data_dir = "/home/mr2238/project_pi_np442/mr2238/accelerate/data"
    img_dir = "/home/mr2238/project_pi_np442/mr2238/accelerate/imgs/in_out"
    global_path = (
        "/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/all_data.hdf5"
    )
    labels_path = os.path.join(data_dir, "labels")
    raw_data_path = os.path.join(data_dir, "raw_data")
    # check img directory exists, if not make it
    os.makedirs(img_dir, exist_ok=True)

    # make combined info df
    information = []
    mode = "mean"
    with h5py.File(global_path, "r") as f:
        for pt in np.array(f["healthy_ptids"][...]).astype(str):
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

    zarr_store = "/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/windows/array.zarr"
    wind_dask = da.from_zarr(zarr_store)

    imputer = SimpleImputer(strategy="mean")
    windows_imp = imputer.fit_transform(wind_dask[:, 1:])

    # manual scaling
    mean_all = da.nanmean(wind_dask[:, 1:]).compute()
    std_all = da.nanstd(wind_dask[:, 1:]).compute()
    scaled_values = (windows_imp - mean_all) / std_all
    scaled_values.compute()

    # do umap
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

    print("Doing PCA")
    n_dim = 600
    pca = PCA(n_components=n_dim)
    pca = pca.fit(scaled_values)
    umap_dim = np.arange(n_dim)[pca.explained_variance_ratio_.cumsum() > 0.95][0] + 1
    X = pca.transform(scaled_values)[:, :umap_dim]
    print("Done.")

    print("Doing UMAP grid search...")
    # umap and graph with changes in basic parameters
    index_w = wind_dask[:, 0].compute()
    for n_neighbors, min_dist in itertools.product([2, 15, 50, 200], [0.1, 0.25, 0.5]):
        print(n_neighbors, min_dist)
        fit2 = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, verbose=True)
        u2 = fit2.fit_transform(X)
        u2_df = pd.DataFrame(u2, columns=["dim_1b", "dim_2b"])
        u2_df = u2_df.set_index(index_w)
        # combine datasets:
        info_df[
            [f"dim_1_{n_neighbors}_{min_dist}", f"dim_2_{n_neighbors}_{min_dist}"]
        ] = u2_df

        # graph and color by in or out
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(
            data=info_df,
            x=f"dim_1_{n_neighbors}_{min_dist}",
            y=f"dim_2_{n_neighbors}_{min_dist}",
            hue="in?",
            palette="Set1",
            s=1,
            linewidth=0,
            alpha=0.4,
        )
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        plt.title("UMAP colored by label")
        leg = plt.legend(title="Inside AR Limits?", markerscale=10)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
        img_name = f"pca_umap_neigh{n_neighbors}_dist{min_dist}.png"
        plt.savefig(os.path.join(img_dir, img_name))
        plt.close()

    info_df.to_pickle(
        f"/home/mr2238/project_pi_np442/mr2238/accelerate/data/processed/windows/total_umaps.pkl"
    )

    print("Done.")
