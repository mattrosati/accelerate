"""CLI training loop for recurrent sequence models.

The module exposes reusable helpers so a single training run and a
hyperparameter search can share the same data loading, grouped split, metrics,
checkpointing, and W&B logging paths.
"""

import json
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from transformers import EarlyStoppingCallback
from transformers.trainer_callback import ProgressCallback

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deep.callbacks import QuietProgressCallback  # noqa: E402
from deep.checkpointing import (  # noqa: E402
    log_wandb_artifacts,
    save_json,
    save_trainer_checkpoint,
    update_wandb_summary,
)
from deep.data import (  # noqa: E402
    build_hf_dataset,
    load_split_arrays,
    make_grouped_split,
)
from deep.metrics import (  # noqa: E402
    GroupAwareMetricsComputer,
    build_history_dataframe,
    extract_best_epoch,
    metric_direction,
    predict_split_metrics,
    resolve_monitor,
    training_metric_name,
)
from deep.models import GRUClassifier, LSTMClassifier  # noqa: E402
from deep.trainer import (  # noqa: E402
    PatientBalancedTrainer,
    build_training_arguments,
)


def build_parser(add_help=True):
    """Build the argparse parser for one deep-learning training run."""
    parser = ArgumentParser(add_help=add_help)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "gru"],
        required=True,
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        choices=["raw"],
        default="raw",
        help="Deep recurrent models currently operate on raw multivariate windows.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="",
        help="Label column to predict. Defaults to `in?` for classification and `MAPopt_Yale_affected_beta` for regression (frac_out also implemented).",
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument(
        "--patient_balance",
        type=str,
        choices=["none", "sampler"],
        default="sampler",
        help="Sampler mode to equalize per-patient contribution during training.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string like cpu, cuda, cuda:0, or auto.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Max number of epochs without validation improvement. Defaults to no early stopping.",
    )
    parser.add_argument(
        "--monitor",
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
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Trainer scheduler type passed through to Hugging Face TrainingArguments.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Optional warmup ratio for the Trainer-managed scheduler.",
    )
    parser.add_argument("--wandb_project", type=str, default="accelerate-deep")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_job_type", type=str, default="")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    return parser


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def seed_everything(seed):
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime_device(requested):
    """Apply the requested runtime device configuration for Trainer."""
    if requested == "auto":
        return False

    device = torch.device(requested)
    if device.type == "cpu":
        return True
    if device.type != "cuda":
        raise ValueError(f"Unsupported device type for Trainer: {requested}")
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but no GPU is available: {requested}")

    torch.cuda.set_device(device)
    return False


def resolve_target_col(args):
    """Resolve the effective target column from the task and CLI args."""
    if args.target_col:
        return args.target_col
    return "in?" if args.task == "classification" else "MAPopt_Yale_affected_beta"


def load_data_bundle(train_dir, data_mode, target_col):
    """Load the repository train/test splits once for reuse across trials."""
    train_X, train_y, train_groups, _, channels = load_split_arrays(
        train_dir, "train", data_mode=data_mode, target_col=target_col
    )
    test_X, test_y, test_groups, _, _ = load_split_arrays(
        train_dir, "test", data_mode=data_mode, target_col=target_col
    )
    return {
        "train_X": train_X,
        "train_y": train_y,
        "train_groups": train_groups,
        "test_X": test_X,
        "test_y": test_y,
        "test_groups": test_groups,
        "channels": channels,
    }


# ---------------------------------------------------------------------------
# run_training decomposition helpers
# ---------------------------------------------------------------------------


def _make_model(args, input_dim, pos_weight=None):
    """Instantiate the requested recurrent predictor from CLI args."""
    kwargs = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "task": args.task,
        "pos_weight": pos_weight,
    }
    if args.model == "lstm":
        return LSTMClassifier(**kwargs)
    if args.model == "gru":
        return GRUClassifier(**kwargs)
    raise ValueError(f"Unsupported model: {args.model}")


def _get_regression_target_stats(y_train):
    """Return train-fold normalization stats for regression targets."""
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std <= 0.0:
        std = 1.0
    return {"mean": mean, "std": std}


def _prepare_targets(args, y_tr, y_val, test_y):
    """Prepare model-space targets and any task-specific training metadata."""
    if args.task == "regression":
        target_stats = _get_regression_target_stats(y_tr)
        return (
            (y_tr - target_stats["mean"]) / target_stats["std"],
            (y_val - target_stats["mean"]) / target_stats["std"],
            (test_y - target_stats["mean"]) / target_stats["std"],
            target_stats,
            None,
        )

    pos_count = y_tr.sum()
    neg_count = len(y_tr) - pos_count
    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            "Training fold must contain both positive and negative examples."
        )
    return y_tr, y_val, test_y, None, (neg_count / pos_count)


def _make_splits(args, data_bundle):
    """Create grouped train/val/test arrays from the loaded data bundle."""
    train_X = data_bundle["train_X"]
    train_y = data_bundle["train_y"]
    train_groups = data_bundle["train_groups"]
    test_X = data_bundle["test_X"]
    test_y = data_bundle["test_y"]
    test_groups = data_bundle["test_groups"]
    channels = data_bundle["channels"]

    train_idx, val_idx = make_grouped_split(
        train_y,
        train_groups,
        n_splits=args.n_splits,
        seed=args.seed,
        fold_idx=args.fold_idx,
        task=args.task,
    )

    X_tr, X_val = train_X[train_idx], train_X[val_idx]
    y_tr, y_val = train_y[train_idx], train_y[val_idx]
    groups_tr, groups_val = train_groups[train_idx], train_groups[val_idx]

    if len(y_tr) == 0 or len(y_val) == 0:
        raise ValueError("Grouped split produced an empty training or validation fold.")

    return {
        "X_tr": X_tr,
        "X_val": X_val,
        "y_tr": y_tr,
        "y_val": y_val,
        "groups_tr": groups_tr,
        "groups_val": groups_val,
        "test_X": test_X,
        "test_y": test_y,
        "test_groups": test_groups,
        "channels": channels,
        "input_dim": train_X.shape[2],
    }


def _prepare_datasets(args, splits):
    """Build HF datasets and resolve task-specific target transforms."""
    y_tr_model, y_val_model, test_y_model, target_stats, pos_weight = _prepare_targets(
        args,
        splits["y_tr"],
        splits["y_val"],
        splits["test_y"],
    )
    train_ds = build_hf_dataset(splits["X_tr"], y_tr_model)
    val_ds = build_hf_dataset(splits["X_val"], y_val_model)
    test_ds = build_hf_dataset(splits["test_X"], test_y_model)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "target_stats": target_stats,
        "pos_weight": pos_weight,
    }


def _maybe_init_wandb(args, run_name, model_store, splits, datasets):
    """Initialize W&B unless logging has been explicitly disabled."""
    if args.wandb_mode == "disabled":
        return None

    wandb_kwargs = {
        "project": args.wandb_project,
        "name": f"{args.model}_{run_name}",
        "config": {
            **vars(args),
            "channels": splits["channels"],
            "train_windows": int(len(datasets["train_ds"])),
            "val_windows": int(len(datasets["val_ds"])),
            "test_windows": int(len(datasets["test_ds"])),
            "train_patients": int(np.unique(splits["groups_tr"]).shape[0]),
            "val_patients": int(np.unique(splits["groups_val"]).shape[0]),
            "test_patients": int(np.unique(splits["test_groups"]).shape[0]),
        },
        "mode": args.wandb_mode,
        "dir": model_store,
        "reinit": "finish_previous",
    }
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    if args.wandb_group:
        wandb_kwargs["group"] = args.wandb_group
    if args.wandb_job_type:
        wandb_kwargs["job_type"] = args.wandb_job_type
    if args.wandb_tags:
        wandb_kwargs["tags"] = args.wandb_tags
    return wandb.init(**wandb_kwargs)


def _log_split_diagnostics(args, splits):
    """Print train/val distribution stats before training starts."""
    n_train = len(splits["y_tr"])
    n_val = len(splits["y_val"])
    n_train_patients = int(np.unique(splits["groups_tr"]).shape[0])
    n_val_patients = int(np.unique(splits["groups_val"]).shape[0])
    print(f"--- Split diagnostics ---")
    print(f"Train: {n_train} windows from {n_train_patients} patients")
    print(f"Val:   {n_val} windows from {n_val_patients} patients")
    if args.task == "classification":
        train_pos = splits["y_tr"].sum()
        val_pos = splits["y_val"].sum()
        print(f"Train class balance: {train_pos/n_train:.3f} positive")
        print(f"Val   class balance: {val_pos/n_val:.3f} positive")
    else:
        print(
            f"Train target: mean={splits['y_tr'].mean():.4f}, std={splits['y_tr'].std():.4f}"
        )
        print(
            f"Val   target: mean={splits['y_val'].mean():.4f}, std={splits['y_val'].std():.4f}"
        )
    print(f"-------------------------")


def _build_trainer(args, splits, datasets, model_store, run_name, wandb_run, no_cuda):
    """Instantiate the model, metrics computer, and PatientBalancedTrainer."""
    model = _make_model(
        args, input_dim=splits["input_dim"], pos_weight=datasets["pos_weight"]
    )
    monitor_name = resolve_monitor(args.task, args.monitor, y_val=splits["y_val"])

    metrics_computer = GroupAwareMetricsComputer(
        args.task,
        target_stats=datasets["target_stats"],
        patient_balance=args.patient_balance != "none",
    )
    metrics_computer.set_groups(splits["groups_val"])

    training_args = build_training_arguments(
        args,
        model_store,
        run_name,
        monitor_name,
        report_to=["wandb"] if wandb_run is not None else [],
        no_cuda=no_cuda,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    trainer = PatientBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train_ds"],
        eval_dataset=datasets["val_ds"],
        compute_metrics=metrics_computer,
        patient_balance=args.patient_balance,
        train_groups=splits["groups_tr"],
        val_groups=splits["groups_val"],
        train_metrics_dataset=datasets["train_ds"],
        metrics_computer=metrics_computer,
        wandb_run=wandb_run,
        callbacks=callbacks,
    )
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(QuietProgressCallback())

    return trainer, metrics_computer, monitor_name


def _collect_and_save_results(
    trainer,
    args,
    splits,
    datasets,
    metrics_computer,
    model_store,
    run_name,
    monitor_name,
    target_col,
    wandb_run,
):
    """Extract metrics, save checkpoint/history/summary, log W&B artifacts."""
    maximize = metric_direction(training_metric_name(monitor_name))
    best_epoch, best_metric = extract_best_epoch(
        trainer.state.log_history,
        training_metric_name(monitor_name),
        maximize,
    )
    if np.isnan(best_metric) and trainer.state.best_metric is not None:
        best_metric = float(trainer.state.best_metric)

    best_path = os.path.join(model_store, f"{args.model}_best.pt")
    save_trainer_checkpoint(
        best_path,
        trainer,
        args,
        splits["channels"],
        best_epoch,
        best_metric,
        monitor_name,
        target_col,
        datasets["target_stats"],
    )

    history_df = build_history_dataframe(trainer.state.log_history)
    history_path = os.path.join(model_store, f"{args.model}_history.csv")
    history_df.to_csv(history_path, index=False)

    train_metrics = predict_split_metrics(
        trainer, datasets["train_ds"], splits["groups_tr"], "train", metrics_computer
    )
    val_metrics = predict_split_metrics(
        trainer, datasets["val_ds"], splits["groups_val"], "val", metrics_computer
    )
    test_metrics = predict_split_metrics(
        trainer, datasets["test_ds"], splits["test_groups"], "test", metrics_computer
    )

    summary = {
        "run_name": run_name,
        "model_store": model_store,
        "best_path": best_path,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "monitor": monitor_name,
        "task": args.task,
        "target_col": target_col,
        "target_stats": datasets["target_stats"],
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    summary_path = os.path.join(model_store, f"{args.model}_summary.json")
    save_json(summary_path, summary)

    if wandb_run is not None:
        update_wandb_summary(
            wandb_run,
            args,
            best_epoch,
            best_metric,
            train_metrics,
            val_metrics,
            test_metrics,
        )
        log_wandb_artifacts(
            wandb_run,
            args,
            run_name,
            best_path,
            history_path,
            summary_path,
            monitor_name,
            best_epoch,
            target_col,
        )

    return summary


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_training(args, data_bundle=None, print_summary=True):
    """Run one recurrent-model training job and return its summary payload."""
    if not isinstance(args, Namespace):
        raise TypeError("run_training expects an argparse.Namespace.")

    seed_everything(args.seed)
    no_cuda = configure_runtime_device(args.device)

    target_col = resolve_target_col(args)
    args.target_col = target_col
    if data_bundle is None:
        data_bundle = load_data_bundle(args.train_dir, args.data_mode, target_col)

    splits = _make_splits(args, data_bundle)
    datasets = _prepare_datasets(args, splits)

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H:%M")
    model_store = os.path.join(args.train_dir, f"deep_models_{run_name}")
    os.makedirs(model_store, exist_ok=True)

    wandb_run = _maybe_init_wandb(args, run_name, model_store, splits, datasets)
    trainer, metrics_computer, monitor_name = _build_trainer(
        args,
        splits,
        datasets,
        model_store,
        run_name,
        wandb_run,
        no_cuda,
    )
    _log_split_diagnostics(args, splits)
    trainer.train()

    summary = _collect_and_save_results(
        trainer,
        args,
        splits,
        datasets,
        metrics_computer,
        model_store,
        run_name,
        monitor_name,
        target_col,
        wandb_run,
    )

    if print_summary:
        print(f"Saved deep model artifacts to {model_store}")
        print(json.dumps(summary, indent=2, sort_keys=True))

    return summary


def main():
    """CLI entrypoint for a single recurrent-model training run."""
    args = build_parser().parse_args()
    if args.patient_balance == "none":
        args.monitor = (
            args.monitor.replace("patient_", "")
            if "patient_" in args.monitor
            else args.monitor
        )

    run_training(args)


if __name__ == "__main__":
    main()
