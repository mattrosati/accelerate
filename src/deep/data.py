import os
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler


class SequenceDataset(Dataset):
    def __init__(self, X, y, groups):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.tensor(y, dtype=torch.float32)
        self.groups = np.asarray(groups).astype(str)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def infer_base_ptid(labels):
    if "base_ptid" in labels.columns:
        return labels["base_ptid"].astype(str).to_numpy()
    return labels["ptid"].astype(str).str.split("_").str[0].to_numpy()


def parse_channels_from_train_dir(train_dir):
    train_dir_name = os.path.basename(train_dir.rstrip("/"))
    train_params = train_dir_name.split("_")

    seconds = None
    channels = []
    for token in train_params:
        if token.endswith("s") and token[:-1].isdigit():
            seconds = int(token[:-1])
        elif seconds is not None:
            channels.append(token)

    if not channels:
        raise ValueError(f"Could not infer channels from dataset name: {train_dir_name}")

    return channels


def reshape_flat_windows(X, num_channels):
    if X.shape[1] % num_channels != 0:
        raise ValueError(
            f"Input width {X.shape[1]} is not divisible by num_channels={num_channels}."
        )

    timesteps = X.shape[1] // num_channels
    X = X.reshape(X.shape[0], num_channels, timesteps)
    return np.transpose(X, (0, 2, 1))


def load_split_arrays(train_dir, split, data_mode="raw"):
    if data_mode != "raw":
        raise ValueError("Deep recurrent models currently support only raw windows.")

    if "downsample" not in train_dir:
        raise ValueError(
            "Deep recurrent models require a downsampled dataset so all channels share a time grid."
        )

    data_file = "x.zarr"
    X = da.from_zarr(os.path.join(train_dir, "permanent", split, data_file)).compute()
    labels = pd.read_pickle(os.path.join(train_dir, "permanent", split, "labels.pkl"))

    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Feature rows ({X.shape[0]}) and labels ({labels.shape[0]}) do not match "
            f"for split `{split}`."
        )

    valid_mask = labels["in?"].notna().to_numpy()
    if "invalid" in labels.columns:
        valid_mask &= ~labels["invalid"].astype(bool).to_numpy()

    if not valid_mask.all():
        labels = labels.loc[valid_mask].reset_index(drop=True)
        X = X[valid_mask]

    if labels.empty:
        raise ValueError(f"No valid labels found for split `{split}` in {train_dir}.")

    y = labels["in?"].astype(int).to_numpy()
    groups = infer_base_ptid(labels)

    channels = parse_channels_from_train_dir(train_dir)
    X = reshape_flat_windows(X, len(channels)).astype(np.float32, copy=False)

    return X, y, groups, labels, channels


def make_grouped_split(y, groups, n_splits=5, seed=42, fold_idx=0):
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < 2:
        raise ValueError("Need at least 2 unique patients to make a grouped split.")

    effective_splits = min(n_splits, unique_groups.shape[0])
    if effective_splits < n_splits:
        warnings.warn(
            f"Reducing n_splits from {n_splits} to {effective_splits} because only "
            f"{unique_groups.shape[0]} unique patients are available.",
            stacklevel=2,
        )

    try:
        splitter = StratifiedGroupKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=seed,
        )
        folds = list(splitter.split(np.zeros_like(y), y, groups))
    except ValueError as exc:
        warnings.warn(
            f"Falling back to GroupKFold because StratifiedGroupKFold failed: {exc}",
            stacklevel=2,
        )
        splitter = GroupKFold(n_splits=effective_splits)
        folds = list(splitter.split(np.zeros_like(y), y, groups))

    if fold_idx >= len(folds):
        raise ValueError(
            f"fold_idx={fold_idx} is invalid for effective n_splits={len(folds)}."
        )
    return folds[fold_idx]


def make_patient_weights(groups):
    groups = np.asarray(groups).astype(str)
    if groups.size == 0:
        return np.array([], dtype=np.float64)
    unique_groups, counts = np.unique(groups, return_counts=True)
    count_map = dict(zip(unique_groups, counts))
    weights = np.array([1.0 / count_map[g] for g in groups], dtype=np.float64)
    return weights * (len(weights) / weights.sum())


def build_weighted_sampler(groups):
    if len(groups) == 0:
        raise ValueError("Cannot build a weighted sampler with no groups.")
    weights = torch.tensor(make_patient_weights(groups), dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
