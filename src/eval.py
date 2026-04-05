import os
import sys
import shutil
import re
from argparse import ArgumentParser
from pickle import dump, load

import numpy as np
import pandas as pd

import zarr
import dask.array as da

from ray import tune
from tuner import train_cv, predict_scores as _predict_scores

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

from patient_utils import make_patient_weights, infer_base_ptid


def predict_scores(estimator, X, multiclass=False):
    """Return (y_prob, y_pred) for backward-compat with eval call sites."""
    if multiclass:
        proba, preds = _predict_scores(estimator, X, multiclass=True)
        return proba, preds
    scores, threshold = _predict_scores(estimator, X)
    return scores, (scores >= threshold).astype(int)


def compute_test_metrics(y, y_prob, y_pred, patient_weights, is_mc):
    """Compute window-level and patient-balanced test metrics."""
    r = {"test_balanced_accuracy": balanced_accuracy_score(y, y_pred)}
    if is_mc:
        r["test_f1_macro"] = f1_score(y, y_pred, average="macro")
        if y_prob is not None and len(np.unique(y)) > 1:
            try:
                r["test_auc"] = roc_auc_score(
                    y, y_prob, multi_class="ovr", average="macro"
                )
            except ValueError:
                r["test_auc"] = np.nan
        else:
            r["test_auc"] = np.nan
    else:
        r["test_auc"] = roc_auc_score(y, y_prob)
    r["test_patient_balanced_accuracy"] = balanced_accuracy_score(
        y, y_pred, sample_weight=patient_weights
    )
    if is_mc:
        if y_prob is not None and len(np.unique(y)) > 1:
            try:
                r["test_patient_auc"] = roc_auc_score(
                    y, y_prob, multi_class="ovr", average="macro",
                    sample_weight=patient_weights,
                )
            except ValueError:
                r["test_patient_auc"] = np.nan
        else:
            r["test_patient_auc"] = np.nan
    else:
        r["test_patient_auc"] = roc_auc_score(
            y, y_prob, sample_weight=patient_weights
        )
    return r


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--train_dir",
        help="Directory to test data and model store.",
        default="/home/mr2238/project_pi_np442/mr2238/accelerate/data/training/w_300s_hr_rso2r_rso2l_spo2_abp",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["raw", "pca", "fpca", "all", "separate_decomp"],
        help="Subset of data types to evaluate all models.",
        default="all",
    )
    parser.add_argument(
        "--small",
        "-s",
        action="store_true",
        help="Whether to evaluate on models trained on smaller dataset",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of training run to evaluate.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
        help="Classification task type.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="",
        help="Label column. Defaults to 'in?' for binary and 'ar_class' for multiclass.",
    )

    args = parser.parse_args()
    if not args.target_col:
        args.target_col = "ar_class" if args.task == "multiclass" else "in?"
    np.random.seed(420)

    print(f"Testing all models with {args.subset} embeddings.")
    if args.small:
        print("Note: small dataset models.")

    # loop through data types
    if args.subset == "all":
        data_types = ["raw", "pca", "fpca", "separate_decomp"]
    else:
        data_types = [args.subset]

    model_dir = "models_debug" if args.small else "models"
    model_dir = model_dir + f"_{args.run_name}" if args.run_name != "" else model_dir
    model_store = os.path.join(args.train_dir, model_dir)
    assert os.path.isdir(model_store)

    rows = []

    # dataset has model, embedding mode, train metrics, test metrics
    for mode in data_types:
        if mode == "fpca":
            continue  # TODO: remove once implemented
        print(f"\nEvaluating models for {mode}.")

        # load test data
        data_string = f"{mode}_x.zarr" if mode != "raw" else "x.zarr"
        X = da.from_zarr(os.path.join(args.train_dir, "permanent", "test", data_string))
        labels = pd.read_pickle(
            os.path.join(args.train_dir, "permanent", "test", "labels.pkl")
        )
        y = labels[args.target_col].astype(int)
        groups = infer_base_ptid(labels)
        patient_weights = make_patient_weights(groups)

        # loop over directory of models
        if "adaptive_repeated_cv_search" in os.listdir(model_store):
            # load results, best model and evaluate
            for f in os.listdir(model_store):
                r = {}
                check = f.replace(
                    "_separate_pca", "_separate_decomp"
                )  # this is bc I made some bad naming practices earlier
                # load search
                if not check.endswith(f"{mode}.pkl"):
                    continue
                model = load(open(os.path.join(model_store, f), "rb"))

                # print model name and evaluate
                model_name = model.__class__.__name__
                print(f"- {model_name}")
                print(f"  Best params: {model.get_params()}")

                # extract training metrics
                model_string = f.removesuffix(f".pkl")
                results_df = pd.read_csv(
                    os.path.join(model_store, f"{model_string}_cv_results.csv")
                )
                print(results_df)
                best = results_df.iloc[0]
                of_interest = [
                    col
                    for col in best.columns
                    if (("mean" in col or "std" in col) and "time" not in col)
                ]
                best = best[of_interest]
                print(best.T)

                # calculate testing metrics
                estimator = model
                is_mc = args.task == "multiclass"
                y_prob, y_pred = predict_scores(estimator, X, multiclass=is_mc)
                r |= compute_test_metrics(y, y_prob, y_pred, patient_weights, is_mc)

                r["mode"] = mode
                r["model"] = model_name
                r = r | {k: v[0] for k, v in best.to_dict(orient="list").items()}
                rows.append(r)
        else:
            for f in os.listdir(model_store):
                r = {}
                check = f.replace(
                    "_separate_pca", "_separate_decomp"
                )  # this is bc I made some bad naming practices earlier
                # load search
                if not check.endswith(f"{mode}_search.pkl"):
                    continue
                search = load(open(os.path.join(model_store, f), "rb"))

                # print model name and evaluate
                model_name = search.best_estimator_.__class__.__name__
                print(f"- {model_name}")
                print(f"  Best params: {search.best_params_}")

                # extract training metrics
                results_df = pd.DataFrame(search.cv_results_)
                print(results_df)
                best_mask = results_df["rank_test_balanced_accuracy"] == 1
                best = results_df[best_mask]
                of_interest = [
                    col
                    for col in best.columns
                    if (("mean" in col or "std" in col) and "time" not in col)
                ]
                best = best[of_interest]
                # best = best.rename(columns=lambda c: re.sub(r"_test_", "_val_", c))
                best = best.rename(columns=lambda c: re.sub(r"_test_", "_val_", c))
                print(best.T)

                # calculate testing metrics
                estimator = search.best_estimator_
                y_prob, y_pred = predict_scores(estimator, X, multiclass=is_mc)
                r |= compute_test_metrics(y, y_prob, y_pred, patient_weights, is_mc)

                r["mode"] = mode
                r["model"] = model_name
                r = r | {k: v[0] for k, v in best.to_dict(orient="list").items()}
                rows.append(r)

        # concatenate rows in one df
        df = pd.DataFrame(rows)
        sort_col = (
            "mean_val_patient_auc" if "mean_val_patient_auc" in df.columns else "mean_val_auc"
        )
        # Prefer the patient-balanced validation metric when it is available.
        df = df.sort_values(sort_col, ascending=False)

        # save as csv
        pd.set_option("display.max_columns", None)
        pd.set_option("display.float_format", "{:.4f}".format)
        print("\nAll results:")
        display_cols = [
            "model",
            "mode",
            "mean_train_auc",
            "mean_val_auc",
            "mean_val_patient_auc",
            "test_balanced_accuracy",
            "test_auc",
            "test_patient_balanced_accuracy",
            "test_patient_auc",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        print(
            df[display_cols]
        )
        df.to_csv(os.path.join(model_store, "results.csv"), index=False)
