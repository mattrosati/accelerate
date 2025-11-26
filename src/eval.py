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
from tuner import train_cv

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

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
        choices=['raw', 'pca', 'fpca', 'all', 'separate_decomp'],
        help="Subset of data types to evaluate all models.",
        default='all',
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

    args = parser.parse_args()
    np.random.seed(420)

    print(f"Testing all models with {args.subset} embeddings.")
    if args.small:
        print("Note: small dataset models.")
    
    # loop through data types
    if args.subset == "all":
        data_types = ['raw', 'pca', 'fpca', 'separate_decomp']
    else:
        data_types = [args.subset]
    
    model_dir = "models_debug" if args.small else "models"
    model_dir = model_dir + f"_{args.run_name}" if args.run_name != "" else model_dir
    model_store = os.path.join(args.train_dir, model_dir)
    assert os.path.isdir(model_store)
    
    rows = []

    # dataset has model, embedding mode, train metrics, test metrics 
    for mode in data_types:
        if mode == 'fpca':
            continue # TODO: remove once implemented
        print(f"\nEvaluating models for {mode}.")
        

        # load test data
        data_string = f'{mode}_x.zarr' if mode != "raw" else 'x.zarr'
        X = da.from_zarr(os.path.join(args.train_dir, 'permanent', 'test', data_string))
        labels = pd.read_pickle(os.path.join(args.train_dir, 'permanent', 'test', "labels.pkl"))
        y = labels["in?"].astype(int)
    
        # loop over directory of models
        if "adaptive_repeated_cv_search" in os.listdir(model_store):
            # load results, best model and evaluate
            for f in os.listdir(model_store):
                r = {}
                check = f.replace("_separate_pca", "_separate_decomp") # this is bc I made some bad naming practices earlier
                # load search
                if not check.endswith(f"{mode}.pkl"):
                    continue
                model = load(open(os.path.join(model_store, f), 'rb'))

                # print model name and evaluate
                model_name = model.__class__.__name__
                print(f"- {model_name}")
                print(f"  Best params: {model.get_params()}")

                # extract training metrics
                model_string = f.removesuffix(f".pkl")
                results_df = pd.read_csv(os.path.join(model_store, f"{model_string}_cv_results.csv"))
                print(results_df)
                best = results_df.iloc[0]
                of_interest = [col for col in best.columns if (("mean" in col or "std" in col) and "time" not in col)]
                best = best[of_interest]
                print(best.T)
                

                # calculate testing metrics
                estimator = model
                if hasattr(estimator, "predict_proba"):
                    y_prob = estimator.predict_proba(X)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                else:
                    y_prob = estimator.decision_function(X)
                    y_pred = (y_prob >= 0).astype(int)  

                r['test_balanced_accuracy'] = balanced_accuracy_score(y, y_pred)
                r['test_auc'] = roc_auc_score(y, y_prob)

                r["mode"] = mode
                r["model"] = model_name
                r = r | {k: v[0] for k, v in best.to_dict(orient="list").items()}
                rows.append(r)
        else:
            for f in os.listdir(model_store):
                r = {}
                check = f.replace("_separate_pca", "_separate_decomp") # this is bc I made some bad naming practices earlier
                # load search
                if not check.endswith(f"{mode}_search.pkl"):
                    continue
                search = load(open(os.path.join(model_store, f), 'rb'))

                # print model name and evaluate
                model_name = search.best_estimator_.__class__.__name__
                print(f"- {model_name}")
                print(f"  Best params: {search.best_params_}")

                # extract training metrics
                results_df = pd.DataFrame(search.cv_results_)
                print(results_df)
                best_mask = results_df["rank_test_balanced_accuracy"] == 1
                best = results_df[best_mask]
                of_interest = [col for col in best.columns if (("mean" in col or "std" in col) and "time" not in col)]
                best = best[of_interest]
                # best = best.rename(columns=lambda c: re.sub(r"_test_", "_val_", c))
                best = best.rename(columns=lambda c: re.sub(r"_test_", "_val_", c))
                print(best.T)
                

                # calculate testing metrics
                estimator = search.best_estimator_
                if hasattr(estimator, "predict_proba"):
                    y_prob = estimator.predict_proba(X)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                else:
                    y_prob = estimator.decision_function(X)
                    y_pred = (y_prob >= 0).astype(int)  

                r['test_balanced_accuracy'] = balanced_accuracy_score(y, y_pred)
                r['test_auc'] = roc_auc_score(y, y_prob)

                r["mode"] = mode
                r["model"] = model_name
                r = r | {k: v[0] for k, v in best.to_dict(orient="list").items()}
                rows.append(r)


        # concatenate rows in one df
        df = pd.DataFrame(rows).sort_values("mean_val_auc", ascending=False)
        
        # save as csv
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', "{:.4f}".format)
        print("\nAll results:")
        print(df[['model', 'mode','mean_train_auc', 'mean_val_auc', 'test_balanced_accuracy', 'test_auc']])
        df.to_csv(os.path.join(model_store, "results.csv"), index=False)