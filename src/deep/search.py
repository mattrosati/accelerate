"""Hyperparameter search entrypoint for recurrent deep models.

This wraps ``deep.train.run_training`` with Optuna so search trials reuse the
same grouped split, patient-balanced sampling, checkpointing, and W&B logging
paths as a normal single training run.
"""

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deep.checkpointing import save_json  # noqa: E402
from deep.metrics import optuna_direction, resolve_monitor  # noqa: E402
from deep.train import (  # noqa: E402
    build_parser,
    load_data_bundle,
    resolve_target_col,
    run_training,
)


def build_search_parser():
    """Build the argparse parser for Optuna-based deep-model search."""
    parser = ArgumentParser(
        parents=[build_parser(add_help=False)],
        conflict_handler="resolve",
    )
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument(
        "--search_name",
        type=str,
        default="",
        help="Name for the search study and W&B group; defaults to a timestamp.",
    )
    parser.add_argument(
        "--search_metric",
        type=str,
        choices=[
            "val_patient_auc",
            "val_auc",
            "val_loss",
            "val_mae",
            "val_rmse",
            "val_r2",
            "val_patient_mae",
            "val_patient_rmse",
            "val_patient_r2",
        ],
        default="val_patient_auc",
    )
    parser.add_argument(
        "--hidden_dim_choices",
        type=int,
        nargs="+",
        default=[64, 128, 256],
    )
    parser.add_argument(
        "--batch_size_choices",
        type=int,
        nargs="+",
        default=[64, 128, 256],
    )
    parser.add_argument(
        "--grad_clip_choices",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 5.0],
    )
    return parser


def suggest_trial_args(base_args, trial):
    """Copy base args and overwrite trainable hyperparameters for one trial."""
    trial_args = Namespace(**vars(deepcopy(base_args)))
    trial_args.hidden_dim = trial.suggest_categorical(
        "hidden_dim", trial_args.hidden_dim_choices
    )
    trial_args.num_layers = trial.suggest_int("num_layers", 1, 3)
    trial_args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    trial_args.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    trial_args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    trial_args.batch_size = trial.suggest_categorical(
        "batch_size", trial_args.batch_size_choices
    )
    trial_args.bidirectional = trial.suggest_categorical("bidirectional", [False, True])
    trial_args.grad_clip = trial.suggest_categorical(
        "grad_clip", trial_args.grad_clip_choices
    )
    trial_args.lr_scheduler_type = trial.suggest_categorical(
        "lr_scheduler_type", ["cosine", "linear", "constant"]
    )
    trial_args.monitor = base_args.search_metric
    trial_args.run_name = f"{base_args.search_name}_trial_{trial.number:03d}"
    if not trial_args.wandb_group:
        trial_args.wandb_group = base_args.search_name
    if not trial_args.wandb_job_type:
        trial_args.wandb_job_type = "hparam-search"
    return trial_args


def extract_metric(summary, metric_name):
    """Map a search metric name to the matching summary field."""
    if not metric_name.startswith("val_"):
        raise ValueError(f"Unsupported search metric: {metric_name}")
    return summary["val"][metric_name.removeprefix("val_")]


def main():
    """CLI entrypoint for Optuna-based deep-model hyperparameter search."""
    args = build_search_parser().parse_args()
    if not args.search_name:
        args.search_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
    args.search_metric = resolve_monitor(args.task, args.search_metric)

    search_store = os.path.join(args.train_dir, f"deep_search_{args.search_name}")
    os.makedirs(search_store, exist_ok=True)

    data_bundle = load_data_bundle(
        args.train_dir,
        args.data_mode,
        resolve_target_col(args),
    )
    direction = optuna_direction(args.search_metric)
    study = optuna.create_study(
        study_name=args.search_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    def objective(trial):
        trial_args = suggest_trial_args(args, trial)
        summary = run_training(trial_args, data_bundle=data_bundle, print_summary=False)
        metric_value = extract_metric(summary, args.search_metric)
        trial.set_user_attr("run_name", summary["run_name"])
        trial.set_user_attr("model_store", summary["model_store"])
        trial.set_user_attr("best_path", summary["best_path"])
        trial.set_user_attr("best_epoch", summary["best_epoch"])
        trial.set_user_attr("metric_value", metric_value)
        return metric_value

    optimize_kwargs = {
        "n_trials": args.n_trials,
        "catch": (Exception,),
    }
    if args.timeout > 0:
        optimize_kwargs["timeout"] = args.timeout

    study.optimize(objective, **optimize_kwargs)

    trials_df = study.trials_dataframe()
    trials_path = os.path.join(search_store, "trials.csv")
    trials_df.to_csv(trials_path, index=False)

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    summary = {
        "search_name": args.search_name,
        "search_metric": args.search_metric,
        "direction": direction,
        "num_trials": len(study.trials),
        "num_completed_trials": len(completed_trials),
        "trials_path": trials_path,
    }
    if completed_trials:
        best_trial = study.best_trial
        summary |= {
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_run_name": best_trial.user_attrs.get("run_name"),
            "best_model_store": best_trial.user_attrs.get("model_store"),
            "best_checkpoint": best_trial.user_attrs.get("best_path"),
        }
    else:
        summary["warning"] = "No completed trials were available."
    save_json(os.path.join(search_store, "summary.json"), summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
