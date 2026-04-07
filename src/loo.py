"""Leave-one-patient-out (LOPO) analysis for classical ML models.

For each trained experiment, loads the best model (by val AUC), then
iterates over every patient in the training set: trains on all *other*
patients and evaluates on the held-out one.  Saves per-patient metrics
to a CSV so downstream analysis can flag low-quality patients.

Usage (single experiment):
    python src/loo.py --data_mode balanced --run_name hyperpar \
        --dataset chop2_freq10_robust_smooth0.20_downsample1_w_300s_hr_rso2r_rso2l_spo2_abp \
        --model_filter xgb_raw

Usage (full sweep):
    python src/loo.py --data_mode balanced --run_name hyperpar
"""

import os, shutil
import sys
import json
from argparse import ArgumentParser
from pickle import load

import numpy as np
import pandas as pd
import dask.array as da
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from tuner import (
    make_patient_weights,
    get_fit_kwargs,
    predict_scores,
    subsample_by_patient,
)
from patient_utils import infer_base_ptid
from deep.data import parse_channels_from_train_dir

# -----------------------------------------------------------------------
# Data-mode → zarr filename mapping (mirrors src/train.py and overfit.ipynb)
# -----------------------------------------------------------------------
DATAMODE_TO_FILE = {
    "raw": "x.zarr",
    "pca": "pca_x.zarr",
    "fpca": "fpca_x.zarr",
    "separate_pca": "separate_decomp_x.zarr",
    "separatepca": "separate_decomp_x.zarr",
    "chronos": "chronos_x.zarr",
    "design": "design_x.zarr",
    "design_w": "white_design_x.zarr",
    "whiten": "white_design_x.zarr",
}

# Reverse mapping used to normalise model-name suffixes to canonical keys
_SUFFIX_ALIASES = {
    "separate_decomp": "separate_pca",
    "separate_pca": "separate_pca",
    "separatepca": "separate_pca",
    "design_w": "design_w",
    "whiten": "design_w",
}


def _infer_datamode(model_name):
    """Extract the data-mode suffix from a model-name string like 'xgb_pca'."""
    for alias, canonical in sorted(_SUFFIX_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if model_name.endswith(f"_{alias}"):
            return canonical
    # Fall back: last token after splitting on '_'
    parts = model_name.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else "raw"


# -----------------------------------------------------------------------
# Experiment discovery  (mirrors overfit.ipynb Cell 26)
# -----------------------------------------------------------------------


def model_path_iter(train_dir, dataset_names, run_name):
    """Yield (experiment_path, is_debug, dataset_name, model_name) tuples."""
    for ds in dataset_names:
        ds_path = os.path.join(train_dir, ds)
        if not os.path.isdir(ds_path):
            continue
        for model_dir in sorted(os.listdir(ds_path)):
            if run_name not in model_dir:
                continue
            md_path = os.path.join(ds_path, model_dir)
            if not os.path.isdir(md_path):
                continue
            for m in sorted(os.listdir(md_path)):
                full = os.path.join(md_path, m)
                if m.endswith(".pkl") or not os.path.isdir(full):
                    continue
                yield full, "debug" in model_dir, ds, m


# -----------------------------------------------------------------------
# Data loading  (mirrors overfit.ipynb Cell 66 / src/train.py)
# -----------------------------------------------------------------------


def load_train_data(train_dir, dataset, datamode):
    """Return (X, y, groups) numpy arrays for the training split."""
    zarr_file = DATAMODE_TO_FILE.get(datamode)
    if zarr_file is None:
        raise ValueError(f"Unknown data mode: {datamode}")

    X = da.from_zarr(
        os.path.join(train_dir, dataset, "permanent", "train", zarr_file)
    ).compute()
    labels = pd.read_pickle(
        os.path.join(train_dir, dataset, "permanent", "train", "labels.pkl")
    )
    y = labels["in?"].astype(int).to_numpy()
    groups = infer_base_ptid(labels)

    return X, y, groups


def _maybe_reshape_multivar(X, model_name, dataset):
    """Reshape X to (N, channels, timesteps) for multivariate models."""
    multivar_prefixes = ("rocket", "knn_multivar", "kn_multivar")
    is_multivar = any(model_name.startswith(p) for p in multivar_prefixes)
    if not is_multivar or "downsample" not in dataset:
        return X

    channels = parse_channels_from_train_dir(dataset)
    num_channels = len(channels)
    timesteps = X.shape[1] // num_channels
    return X.reshape(X.shape[0], num_channels, timesteps)


# -----------------------------------------------------------------------
# LOO core
# -----------------------------------------------------------------------


def _loo_single_patient(
    patient_id,
    estimator,
    X,
    y,
    groups,
    labels,
    balance_mode,
    max_windows_per_patient,
):
    """Train on all patients except *patient_id* and evaluate on the held-out one."""
    val_mask = groups == patient_id
    train_mask = ~val_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    groups_tr = groups[train_mask]

    model = clone(estimator)

    # Patient balancing — mirrors src/tuner.py train_cv logic
    fit_kwargs = {}
    if balance_mode == "weight":
        fit_kwargs = get_fit_kwargs(model, make_patient_weights(groups_tr))
    elif balance_mode == "subsample":
        rng = np.random.default_rng(42)
        idx = subsample_by_patient(
            np.arange(len(y_tr)), groups_tr, max_windows_per_patient, rng
        )
        X_tr, y_tr, groups_tr = X_tr[idx], y_tr[idx], groups_tr[idx]

    model.fit(X_tr, y_tr, **fit_kwargs)

    scores, threshold = predict_scores(model, X_val)
    y_pred = (scores >= threshold).astype(int)

    if len(set(y_val)) < 2:
        bal_acc = float("nan")
    else:
        bal_acc = balanced_accuracy_score(y_val, y_pred)

    record = {
        "patient_id": patient_id,
        "n_windows": int(val_mask.sum()),
        "n_positive": int(y_val.sum()),
        "n_negative": int(len(y_val) - y_val.sum()),
        "balanced_accuracy": bal_acc,
        "sensitivity": (
            np.nan if y_val.sum() == 0 else (y_pred[y_val == 1] == 1).mean()
        ),
        "specificity": (
            np.nan if y_val.sum() == len(y_val) else (y_pred[y_val == 0] == 0).mean()
        ),
    }
    if np.unique(y_val).shape[0] > 1:
        record["auc"] = roc_auc_score(y_val, scores, labels=labels)
    else:
        record["auc"] = np.nan

    return record


def run_loo_for_experiment(
    estimator,
    X,
    y,
    groups,
    balance_mode,
    max_windows_per_patient,
):
    """Run leave-one-patient-out evaluation in parallel using all CPUs."""
    unique_patients = np.unique(groups)
    labels = np.unique(y)

    from functools import partial

    fn = partial(
        _loo_single_patient,
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        labels=labels,
        balance_mode=balance_mode,
        max_windows_per_patient=max_windows_per_patient,
    )
    results = process_map(fn, unique_patients, desc="LOO patients")

    return results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def build_parser():
    parser = ArgumentParser(description="Leave-one-patient-out analysis.")
    parser.add_argument(
        "--data_mode",
        type=str,
        required=True,
        help="Top-level data directory (balanced, total, stride, deep).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name to filter model directories (e.g. hyperpar).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Restrict to a single dataset name.",
    )
    parser.add_argument(
        "--model_filter",
        type=str,
        default="",
        help="Restrict to a single model name (e.g. xgb_pca).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/loo",
        help="Directory to save LOO result CSVs.",
    )
    parser.add_argument(
        "--patient_balance",
        type=str,
        choices=["none", "weight", "subsample"],
        default="weight",
    )
    parser.add_argument(
        "--max_windows_per_patient",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/mr2238/scratch_pi_np442/mr2238/accelerate",
        help="Root of the data tree.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    train_dir = os.path.join(args.base_dir, args.data_mode)

    # Discover datasets
    if args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = sorted(
            d
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )

    selector = (
        "mean_val_patient_auc" if args.data_mode == "balanced" else "mean_val_auc"
    )

    out_dir = os.path.join(args.output_dir)
    if os.path.exists(out_dir):
        # remove existing CSVs for this selector to avoid appending to old results
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Cache loaded data per (dataset, datamode) to avoid reloading
    data_cache = {}

    for exp_path, is_debug, dataset, model_name in model_path_iter(
        train_dir, dataset_names, args.run_name
    ):
        if args.model_filter and model_name != args.model_filter:
            continue

        print(f"\n{'='*60}")
        print(f"Experiment: {dataset} / {model_name}  (debug={is_debug})")
        print(f"{'='*60}")

        # 1. Load the saved best estimator (pkl)
        model_dir = os.path.dirname(exp_path)
        pkl_path = os.path.join(model_dir, f"{model_name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  No pkl found at {pkl_path}, skipping.")
            continue
        with open(pkl_path, "rb") as fh:
            best_estimator = load(fh)

        # 2. Infer data mode from model name
        datamode = _infer_datamode(model_name)

        # 3. Load training data
        cache_key = (dataset, datamode)
        if cache_key not in data_cache:
            print(f"  Loading data: {dataset} / {datamode}")
            try:
                X, y, groups = load_train_data(train_dir, dataset, datamode)
            except Exception as e:
                print(f"  Failed to load data: {e}")
                continue
            data_cache[cache_key] = (X, y, groups)
        X, y, groups = data_cache[cache_key]

        # 4. Reshape for multivariate models if needed
        X_use = _maybe_reshape_multivar(X, model_name, dataset)

        n_patients = np.unique(groups).shape[0]
        print(f"  Patients: {n_patients}, Windows: {len(y)}")
        print(f"  Estimator: {best_estimator.__class__.__name__}")

        # 5. Run LOO
        loo_results = run_loo_for_experiment(
            estimator=best_estimator,
            X=X_use,
            y=y,
            groups=groups,
            balance_mode=args.patient_balance,
            max_windows_per_patient=args.max_windows_per_patient,
        )

        # 6. Save results
        df = pd.DataFrame(loo_results)
        df["model_name"] = model_name
        df["dataset"] = dataset
        df["datamode"] = datamode
        df["debug"] = is_debug

        tag = f"{args.data_mode}_{args.run_name}_loo"
        csv_path = os.path.join(out_dir, f"{tag}.csv")
        if os.path.exists(csv_path):
            print(f"  Appending to existing file {csv_path}")
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False, header=True)
            print(f"  Saved {csv_path}  ({len(df)} patients)")

        # Summary
        summary = {
            "dataset": dataset,
            "model_name": model_name,
            "datamode": datamode,
            "debug": is_debug,
            "n_patients": n_patients,
            "n_windows": int(len(y)),
            "balance_mode": args.patient_balance,
            "median_balanced_accuracy": float(df["balanced_accuracy"].median()),
            "mean_balanced_accuracy": float(df["balanced_accuracy"].mean()),
            "std_balanced_accuracy": float(df["balanced_accuracy"].std()),
            "median_auc": float(df["auc"].median()),
            "mean_auc": float(df["auc"].mean()),
            "n_single_class_patients": int(df["auc"].isna().sum()),
        }
        summary_path = os.path.join(out_dir, f"{tag}_summary.json")
        summ_df = pd.DataFrame(summary, index=[0])
        if os.path.exists(summary_path):
            print(f"  Appending to existing summary file {summary_path}")
            summ_df.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            summ_df.to_csv(summary_path, mode="a", header=False, index=False)
            print(f"  Saved {summary_path}")

        print(
            f"  AUC: median={summary['median_auc']:.3f}, "
            f"mean={summary['mean_auc']:.3f}, "
            f"single-class={summary['n_single_class_patients']}"
        )


if __name__ == "__main__":
    main()
