"""Dataset loading helpers for deep recurrent models.

The rest of the repository stores windows as flattened feature vectors. This
module validates those arrays, reshapes them back to multivariate sequences,
and prepares patient-level grouping utilities for training and evaluation.
"""

import os
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import torch
from datasets import Array2D, Dataset as HFDataset, Features, Value
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import WeightedRandomSampler

from patient_utils import infer_base_ptid, make_patient_weights  # noqa: E402

DATA_MODE_FILES = {
    "raw": "x.zarr",
    "design": "design_x.zarr",
    "whiten": "white_design_x.zarr",
}


def parse_channels_from_train_dir(train_dir):
    """Infer channel order from the dataset directory naming convention."""
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
    """Convert flattened windows into ``[n_samples, timesteps, channels]``."""
    if X.shape[1] % num_channels != 0:
        raise ValueError(
            f"Input width {X.shape[1]} is not divisible by num_channels={num_channels}."
        )

    timesteps = X.shape[1] // num_channels
    X = X.reshape(X.shape[0], num_channels, timesteps)
    # The recurrent models expect time-major ordering per sample.
    return np.transpose(X, (0, 2, 1))


def resolve_data_file(data_mode):
    """Map a supported data mode to the stored zarr filename."""
    try:
        return DATA_MODE_FILES[data_mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported data_mode `{data_mode}`. Expected one of {sorted(DATA_MODE_FILES)}."
        ) from exc


def load_split_arrays(train_dir, split, data_mode="raw", target_col="in?"):
    """Load one repository split and reshape it into recurrent-model input.

    Returns feature tensors, binary labels, base-patient groups, the filtered
    labels dataframe, and inferred channel names.
    """
    if "downsample" not in train_dir:
        raise ValueError(
            "Deep recurrent models require a downsampled dataset so all channels share a time grid."
        )

    data_file = resolve_data_file(data_mode)
    X = da.from_zarr(os.path.join(train_dir, "permanent", split, data_file)).compute()
    labels = pd.read_pickle(os.path.join(train_dir, "permanent", split, "labels.pkl"))

    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Feature rows ({X.shape[0]}) and labels ({labels.shape[0]}) do not match "
            f"for split `{split}`."
        )

    # Drop invalid or missing labels before any reshaping so downstream tensor
    # dimensions stay aligned with the filtered dataframe.
    if target_col not in labels.columns:
        raise ValueError(f"Target column `{target_col}` not found in labels dataframe.")

    valid_mask = labels[target_col].notna().to_numpy()
    if "invalid" in labels.columns:
        valid_mask &= ~labels["invalid"].astype(bool).to_numpy()

    if not valid_mask.all():
        labels = labels.loc[valid_mask].reset_index(drop=True)
        X = X[valid_mask]

    if labels.empty:
        raise ValueError(f"No valid labels found for split `{split}` in {train_dir}.")

    y = pd.to_numeric(labels[target_col], errors="raise").astype(np.float32).to_numpy()
    groups = infer_base_ptid(labels)

    channels = parse_channels_from_train_dir(train_dir)
    X = reshape_flat_windows(X, len(channels)).astype(np.float32, copy=False)

    return X, y, groups, labels, channels


def build_hf_dataset(X, y, task="classification"):
    """Build a Hugging Face Dataset for one split of recurrent windows."""
    X = np.asarray(X, dtype=np.float32)
    if task == "multiclass":
        y = np.asarray(y, dtype=np.int64)
        label_feature = Value("int64")
    else:
        y = np.asarray(y, dtype=np.float32)
        label_feature = Value("float32")

    features = Features(
        {
            "features": Array2D(
                shape=(int(X.shape[1]), int(X.shape[2])),
                dtype="float32",
            ),
            "labels": label_feature,
        }
    )
    dataset = HFDataset.from_dict(
        {
            "features": X,
            "labels": y,
        },
        features=features,
    )
    return dataset.with_format("torch", columns=["features", "labels"])


def make_grouped_split(y, groups, n_splits=5, seed=42, fold_idx=0, task="classification"):
    """Build one grouped train/validation split.

    Stratified grouping is preferred, but the helper falls back to ``GroupKFold``
    when the label/group layout is too constrained to stratify safely. For
    regression targets, plain grouped folds are used because continuous labels
    cannot be stratified directly.
    """
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

    if task == "regression":
        splitter = GroupKFold(n_splits=effective_splits)
        folds = list(splitter.split(np.zeros_like(y), y, groups))
    else:
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


def build_weighted_sampler(groups):
    """Create a replacement sampler that equalizes patient contribution."""
    if len(groups) == 0:
        raise ValueError("Cannot build a weighted sampler with no groups.")
    weights = torch.tensor(make_patient_weights(groups), dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
