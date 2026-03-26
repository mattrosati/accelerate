"""CLI training loop for recurrent sequence models.

The module exposes reusable helpers so a single training run and a
hyperparameter search can share the same data loading, grouped split, metrics,
checkpointing, and W&B logging paths.
"""

import json
import os
import random
import re
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from transformers import EarlyStoppingCallback

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deep.data import (  # noqa: E402
    build_hf_dataset,
    load_split_arrays,
    make_grouped_split,
)
from deep.models import GRUClassifier, LSTMClassifier  # noqa: E402
from deep.trainer import (  # noqa: E402
    GroupAwareMetricsComputer,
    PatientBalancedTrainer,
    build_history_dataframe,
    build_training_arguments,
    extract_best_epoch,
    make_monitor_name,
    metric_direction,
    predict_split_metrics,
    save_trainer_checkpoint,
    update_wandb_summary,
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
        choices=["raw", "design", "whiten"],
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
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
        default="linear",
        help="Trainer scheduler type passed through to Hugging Face TrainingArguments.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
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


def make_model(args, input_dim, pos_weight=None):
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


def resolve_target_col(args):
    """Resolve the effective target column from the task and CLI args."""
    if args.target_col:
        return args.target_col
    return "in?" if args.task == "classification" else "MAPopt_Yale_affected_beta"


def get_regression_target_stats(y_train):
    """Return train-fold normalization stats for regression targets."""
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std <= 0.0:
        std = 1.0
    return {"mean": mean, "std": std}


def prepare_targets(args, y_tr, y_val, test_y):
    """Prepare model-space targets and any task-specific training metadata."""
    if args.task == "regression":
        target_stats = get_regression_target_stats(y_tr)
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
        raise ValueError("Training fold must contain both positive and negative examples.")
    return y_tr, y_val, test_y, None, (neg_count / pos_count)


def build_split_datasets(X_tr, y_tr, X_val, y_val, test_X, test_y):
    """Build Trainer-ready datasets for train/validation/test splits."""
    train_ds = build_hf_dataset(X_tr, y_tr)
    val_ds = build_hf_dataset(X_val, y_val)
    test_ds = build_hf_dataset(test_X, test_y)
    return train_ds, val_ds, test_ds


def save_json(path, payload):
    """Write a JSON payload with stable formatting for later review."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


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


def sanitize_wandb_artifact_name(value):
    """Return an artifact-safe name for W&B logging."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    sanitized = sanitized.strip("-.")
    return sanitized or "artifact"


def maybe_init_wandb(
    args,
    run_name,
    model_store,
    channels,
    groups_tr,
    groups_val,
    test_groups,
    train_ds,
    val_ds,
    test_ds,
):
    """Initialize W&B unless logging has been explicitly disabled."""
    if args.wandb_mode == "disabled":
        return None

    wandb_kwargs = {
        "project": args.wandb_project,
        "name": f"{args.model}_{run_name}",
        "config": {
            **vars(args),
            "channels": channels,
            "train_windows": int(len(train_ds)),
            "val_windows": int(len(val_ds)),
            "test_windows": int(len(test_ds)),
            "train_patients": int(np.unique(groups_tr).shape[0]),
            "val_patients": int(np.unique(groups_val).shape[0]),
            "test_patients": int(np.unique(test_groups).shape[0]),
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

    y_tr_model, y_val_model, test_y_model, regression_target_stats, pos_weight = prepare_targets(
        args,
        y_tr,
        y_val,
        test_y,
    )
    train_ds, val_ds, test_ds = build_split_datasets(
        X_tr,
        y_tr_model,
        X_val,
        y_val_model,
        test_X,
        test_y_model,
    )

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H:%M")
    model_store = os.path.join(args.train_dir, f"deep_models_{run_name}")
    os.makedirs(model_store, exist_ok=True)

    model = make_model(args, input_dim=train_X.shape[2], pos_weight=pos_weight)
    wandb_run = maybe_init_wandb(
        args,
        run_name,
        model_store,
        channels,
        groups_tr,
        groups_val,
        test_groups,
        train_ds,
        val_ds,
        test_ds,
    )

    monitor_name = make_monitor_name(args, y_val)
    metrics_computer = GroupAwareMetricsComputer(
        args.task,
        target_stats=regression_target_stats,
    )
    metrics_computer.set_groups(groups_val)

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
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=metrics_computer,
        patient_balance=args.patient_balance,
        train_groups=groups_tr,
        val_groups=groups_val,
        train_metrics_dataset=train_ds,
        metrics_computer=metrics_computer,
        wandb_run=wandb_run,
        callbacks=callbacks,
    )
    trainer.train()

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
        channels,
        best_epoch,
        best_metric,
        monitor_name,
        target_col,
        regression_target_stats,
    )

    history_df = build_history_dataframe(trainer.state.log_history)
    history_path = os.path.join(model_store, f"{args.model}_history.csv")
    history_df.to_csv(history_path, index=False)

    train_metrics = predict_split_metrics(
        trainer, train_ds, groups_tr, "train", metrics_computer
    )
    val_metrics = predict_split_metrics(
        trainer, val_ds, groups_val, "val", metrics_computer
    )
    test_metrics = predict_split_metrics(
        trainer, test_ds, test_groups, "test", metrics_computer
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
        "target_stats": regression_target_stats,
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
        artifact = wandb.Artifact(
            name=sanitize_wandb_artifact_name(f"{args.model}_{run_name}"),
            type="model",
            metadata={
                "train_dir": args.train_dir,
                "monitor": monitor_name,
                "best_epoch": best_epoch,
                "task": args.task,
                "target_col": target_col,
            },
        )
        artifact.add_file(best_path)
        artifact.add_file(history_path)
        artifact.add_file(summary_path)
        wandb.log_artifact(artifact)
        wandb.finish()

    if print_summary:
        print(f"Saved deep model artifacts to {model_store}")
        print(json.dumps(summary, indent=2, sort_keys=True))

    return summary


def main():
    """CLI entrypoint for a single recurrent-model training run."""
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
