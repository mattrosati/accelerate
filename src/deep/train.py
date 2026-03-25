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
import warnings
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deep.data import (  # noqa: E402
    build_hf_dataset,
    build_weighted_sampler,
    load_split_arrays,
    make_grouped_split,
    make_patient_weights,
)
from deep.models import GRUClassifier, LSTMClassifier  # noqa: E402


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


def logits_to_predictions(logits):
    """Convert logits to probabilities and hard predictions."""
    logits = np.asarray(logits, dtype=np.float64)
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def metric_or_default(value, maximize):
    """Map ``NaN`` monitor values to deterministic worst-case sentinels."""
    if np.isnan(value):
        return -np.inf if maximize else np.inf
    return value


def compute_metrics(loss, y_true, y_prob, y_pred, groups):
    """Compute window-level and patient-balanced binary classification metrics."""
    if y_true.size == 0:
        return {
            "loss": float(loss),
            "auc": np.nan,
            "balanced_accuracy": np.nan,
            "patient_balanced_accuracy": np.nan,
            "patient_auc": np.nan,
        }

    metrics = {
        "loss": float(loss),
        "auc": np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if np.unique(y_true).shape[0] > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)

    patient_weights = make_patient_weights(groups)
    metrics["patient_balanced_accuracy"] = np.nan
    if patient_weights.size > 0:
        metrics["patient_balanced_accuracy"] = balanced_accuracy_score(
            y_true, y_pred, sample_weight=patient_weights
        )
    metrics["patient_auc"] = np.nan
    if patient_weights.size > 0 and np.unique(y_true).shape[0] > 1:
        metrics["patient_auc"] = roc_auc_score(
            y_true, y_prob, sample_weight=patient_weights
        )

    return metrics


def compute_regression_metrics(loss, y_true, y_pred, groups):
    """Compute window-level and patient-balanced regression metrics."""
    if y_true.size == 0:
        return {
            "loss": float(loss),
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "patient_mae": np.nan,
            "patient_rmse": np.nan,
            "patient_r2": np.nan,
        }

    patient_weights = make_patient_weights(groups)
    metrics = {
        "loss": float(loss),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": np.nan,
        "patient_mae": np.nan,
        "patient_rmse": np.nan,
        "patient_r2": np.nan,
    }
    if y_true.shape[0] > 1 and np.unique(y_true).shape[0] > 1:
        metrics["r2"] = r2_score(y_true, y_pred)
    if patient_weights.size > 0:
        metrics["patient_mae"] = mean_absolute_error(
            y_true, y_pred, sample_weight=patient_weights
        )
        metrics["patient_rmse"] = float(
            np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=patient_weights))
        )
        if y_true.shape[0] > 1 and np.unique(y_true).shape[0] > 1:
            metrics["patient_r2"] = r2_score(
                y_true, y_pred, sample_weight=patient_weights
            )

    return metrics


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
        "reinit": True,
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


class GroupAwareMetricsComputer:
    """Compute metrics for Trainer while preserving patient grouping."""

    def __init__(self, task, target_stats=None):
        self.task = task
        self.target_stats = target_stats
        self.groups = np.array([], dtype=str)

    def set_groups(self, groups):
        """Swap in the group labels for the next evaluation call."""
        self.groups = np.asarray(groups).astype(str)

    def __call__(self, eval_pred):
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        if predictions.ndim > 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        labels = np.asarray(eval_pred.label_ids)
        if labels.ndim > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)

        if self.task == "classification":
            y_true = labels.astype(int)
            y_prob, y_pred = logits_to_predictions(predictions)
            metrics = compute_metrics(np.nan, y_true, y_prob, y_pred, self.groups)
        else:
            y_true = labels.astype(np.float64)
            y_pred = predictions.astype(np.float64)
            if self.target_stats is not None:
                y_true = y_true * self.target_stats["std"] + self.target_stats["mean"]
                y_pred = y_pred * self.target_stats["std"] + self.target_stats["mean"]
            metrics = compute_regression_metrics(np.nan, y_true, y_pred, self.groups)

        metrics.pop("loss", None)
        return metrics


class PatientBalancedTrainer(Trainer):
    """Trainer subclass that preserves inverse-frequency patient sampling."""

    def __init__(self, *args, patient_balance="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.patient_balance = patient_balance

    def _get_train_sampler(self):
        if self.patient_balance != "sampler" or self.train_dataset is None:
            return super()._get_train_sampler()
        return build_weighted_sampler(self.train_dataset["groups"])


def make_monitor_name(args, y_val):
    """Resolve the effective validation metric used for early stopping."""
    monitor_name = args.monitor
    if args.task == "regression" and monitor_name in {"val_auc", "val_patient_auc"}:
        warnings.warn(
            f"Switching monitor from {monitor_name} to val_patient_rmse for regression.",
            stacklevel=2,
        )
        monitor_name = "val_patient_rmse"
    if (
        args.task == "classification"
        and np.unique(y_val).shape[0] < 2
        and monitor_name in {"val_auc", "val_patient_auc"}
    ):
        warnings.warn(
            f"Validation fold has a single class; switching monitor from {monitor_name} "
            "to val_loss.",
            stacklevel=2,
        )
        monitor_name = "val_loss"
    return monitor_name


def training_metric_name(monitor_name):
    """Map repository-style validation metrics to Trainer metric names."""
    if not monitor_name.startswith("val_"):
        raise ValueError(f"Unsupported monitor name: {monitor_name}")
    return f"eval_{monitor_name.removeprefix('val_')}"


def metric_direction(metric_name):
    """Return whether a metric should be maximized."""
    minimize_metrics = {
        "eval_loss",
        "eval_mae",
        "eval_rmse",
        "eval_patient_mae",
        "eval_patient_rmse",
    }
    return metric_name not in minimize_metrics


def extract_best_epoch(log_history, metric_name, maximize):
    """Recover the best epoch from Trainer log history."""
    candidates = [
        entry
        for entry in log_history
        if metric_name in entry and "epoch" in entry
    ]
    if not candidates:
        return -1, np.nan

    best_entry = max(
        candidates,
        key=lambda entry: metric_or_default(entry[metric_name], maximize=maximize),
    )
    if not maximize:
        best_entry = min(
            candidates,
            key=lambda entry: metric_or_default(entry[metric_name], maximize=maximize),
        )

    return int(round(best_entry["epoch"])), float(best_entry[metric_name])


def build_history_dataframe(log_history):
    """Convert Trainer log history into a compact epoch-by-epoch dataframe."""
    rows = {}
    ignored_eval_fields = {
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "eval_jit_compilation_time",
    }
    for entry in log_history:
        if "epoch" not in entry:
            continue
        epoch = int(round(entry["epoch"]))
        row = rows.setdefault(epoch, {"epoch": epoch})
        if "loss" in entry and "eval_loss" not in entry:
            row["train_loss"] = float(entry["loss"])
        if "learning_rate" in entry:
            row["lr"] = float(entry["learning_rate"])
        for key, value in entry.items():
            if key.startswith("eval_") and key not in ignored_eval_fields:
                row[f"val_{key.removeprefix('eval_')}"] = float(value)

    if not rows:
        return pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "lr"])
    return pd.DataFrame(rows.values()).sort_values("epoch").reset_index(drop=True)


def normalize_metric_values(metrics):
    """Convert Trainer metric payloads into plain Python scalars."""
    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def extract_prefixed_metrics(metrics, prefix):
    """Strip a Trainer metric prefix from a metric dictionary."""
    prefix = f"{prefix}_"
    ignored_suffixes = {
        "runtime",
        "samples_per_second",
        "steps_per_second",
        "jit_compilation_time",
    }
    normalized = {}
    for key, value in normalize_metric_values(metrics).items():
        if not key.startswith(prefix):
            continue
        suffix = key.removeprefix(prefix)
        if suffix in ignored_suffixes:
            continue
        normalized[suffix] = value
    return normalized


def build_training_arguments(args, model_store, run_name, monitor_name, report_to, no_cuda):
    """Create TrainingArguments configured for epoch-level evaluation."""
    return TrainingArguments(
        output_dir=model_store,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size * 2, 1),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=training_metric_name(monitor_name),
        greater_is_better=metric_direction(training_metric_name(monitor_name)),
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=max(args.grad_clip, 0.0),
        no_cuda=no_cuda,
        report_to=report_to,
        run_name=f"{args.model}_{run_name}",
        logging_dir=os.path.join(model_store, "logs"),
        remove_unused_columns=True,
    )


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

    regression_target_stats = None
    if args.task == "regression":
        regression_target_stats = get_regression_target_stats(y_tr)
        y_tr_model = (y_tr - regression_target_stats["mean"]) / regression_target_stats[
            "std"
        ]
        y_val_model = (
            y_val - regression_target_stats["mean"]
        ) / regression_target_stats["std"]
        test_y_model = (
            test_y - regression_target_stats["mean"]
        ) / regression_target_stats["std"]
        pos_weight = None
    else:
        y_tr_model = y_tr
        y_val_model = y_val
        test_y_model = test_y
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        if pos_count == 0 or neg_count == 0:
            raise ValueError(
                "Training fold must contain both positive and negative examples."
            )
        pos_weight = neg_count / pos_count

    train_ds = build_hf_dataset(X_tr, y_tr_model, groups_tr)
    val_ds = build_hf_dataset(X_val, y_val_model, groups_val)
    test_ds = build_hf_dataset(test_X, test_y_model, test_groups)

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
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config": vars(args),
            "channels": channels,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "monitor": monitor_name,
            "task": args.task,
            "target_col": target_col,
            "target_stats": regression_target_stats,
            "trainer_best_checkpoint": trainer.state.best_model_checkpoint,
        },
        best_path,
    )

    history_df = build_history_dataframe(trainer.state.log_history)
    history_path = os.path.join(model_store, f"{args.model}_history.csv")
    history_df.to_csv(history_path, index=False)

    metrics_computer.set_groups(groups_tr)
    train_output = trainer.predict(train_ds, metric_key_prefix="train")
    train_metrics = extract_prefixed_metrics(train_output.metrics, "train")

    metrics_computer.set_groups(groups_val)
    val_output = trainer.predict(val_ds, metric_key_prefix="val")
    val_metrics = extract_prefixed_metrics(val_output.metrics, "val")

    metrics_computer.set_groups(test_groups)
    test_output = trainer.predict(test_ds, metric_key_prefix="test")
    test_metrics = extract_prefixed_metrics(test_output.metrics, "test")

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
        wandb.summary.update(
            {
                "best_epoch": best_epoch,
                "best_metric": best_metric,
            }
        )
        if args.task == "classification":
            wandb.summary.update(
                {
                    "final_train_auc": train_metrics["auc"],
                    "final_val_auc": val_metrics["auc"],
                    "final_val_patient_auc": val_metrics["patient_auc"],
                    "final_test_auc": test_metrics["auc"],
                    "final_test_patient_auc": test_metrics["patient_auc"],
                }
            )
        else:
            wandb.summary.update(
                {
                    "final_train_rmse": train_metrics["rmse"],
                    "final_val_rmse": val_metrics["rmse"],
                    "final_val_patient_rmse": val_metrics["patient_rmse"],
                    "final_test_rmse": test_metrics["rmse"],
                    "final_test_patient_rmse": test_metrics["patient_rmse"],
                }
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
