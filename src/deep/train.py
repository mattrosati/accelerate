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
from torch import nn
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deep.data import (  # noqa: E402
    SequenceDataset,
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
        help="Label column to predict. Defaults to `in?` for classification and `frac_out` for regression.",
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
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
    parser.add_argument("--early_stopping_patience", type=int, default=10)
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


def make_model(args, input_dim):
    """Instantiate the requested recurrent predictor from CLI args."""
    if args.model == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
    if args.model == "gru":
        return GRUClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def get_device(requested):
    """Resolve the torch device from a CLI string."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


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
    return "in?" if args.task == "classification" else "frac_out"


def get_regression_target_stats(y_train):
    """Return train-fold normalization stats for regression targets."""
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std <= 0.0:
        std = 1.0
    return {"mean": mean, "std": std}


def evaluate(model, dataloader, criterion, device, groups, task, target_stats=None):
    """Evaluate a model over one dataloader and return aggregate metrics."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.shape[0]
            total_examples += X_batch.shape[0]
            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())

    avg_loss = total_loss / max(total_examples, 1)
    if total_examples == 0:
        if task == "classification":
            return compute_metrics(
                avg_loss, np.array([], dtype=int), np.array([]), np.array([]), groups
            )
        return compute_regression_metrics(avg_loss, np.array([]), np.array([]), groups)

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_targets).numpy()

    if task == "classification":
        y_true = y_true.astype(int)
        y_prob, y_pred = logits_to_predictions(logits)
        return compute_metrics(avg_loss, y_true, y_prob, y_pred, groups)

    y_pred = logits
    if target_stats is not None:
        y_pred = y_pred * target_stats["std"] + target_stats["mean"]
        y_true = y_true * target_stats["std"] + target_stats["mean"]
    return compute_regression_metrics(avg_loss, y_true, y_pred, groups)


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
    """Run one optimization epoch and return mean training loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * X_batch.shape[0]
        total_examples += X_batch.shape[0]

    return total_loss / max(total_examples, 1)


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


def _make_dataloaders(
    args, X_tr, y_tr, groups_tr, X_val, y_val, groups_val, test_X, test_y, test_groups
):
    """Build train/validation/test dataloaders for one grouped split."""
    train_ds = SequenceDataset(X_tr, y_tr, groups_tr)
    val_ds = SequenceDataset(X_val, y_val, groups_val)
    test_ds = SequenceDataset(test_X, test_y, test_groups)

    train_sampler = None
    shuffle = True
    if args.patient_balance == "sampler":
        train_sampler = build_weighted_sampler(groups_tr)
        shuffle = False

    # Training may use replacement sampling, but train metrics should be
    # computed on the full underlying training fold rather than sampled batches.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_batch_size = (
        min(args.batch_size * 2, len(val_ds)) if len(val_ds) > 0 else args.batch_size
    )
    train_eval_batch_size = min(args.batch_size * 2, len(train_ds))
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=max(train_eval_batch_size, 1),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(eval_batch_size, 1),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_eval_batch_size = min(args.batch_size * 2, len(test_ds))
    test_loader = DataLoader(
        test_ds,
        batch_size=max(test_eval_batch_size, 1),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        train_eval_loader,
        val_loader,
        test_loader,
    )


def run_training(args, data_bundle=None, print_summary=True):
    """Run one recurrent-model training job and return its summary payload."""
    if not isinstance(args, Namespace):
        raise TypeError("run_training expects an argparse.Namespace.")

    seed_everything(args.seed)
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
        y_tr_model = (y_tr - regression_target_stats["mean"]) / regression_target_stats["std"]
        y_val_model = (y_val - regression_target_stats["mean"]) / regression_target_stats["std"]
        test_y_model = (
            test_y - regression_target_stats["mean"]
        ) / regression_target_stats["std"]
    else:
        y_tr_model = y_tr
        y_val_model = y_val
        test_y_model = test_y

    (
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        train_eval_loader,
        val_loader,
        test_loader,
    ) = _make_dataloaders(
        args,
        X_tr,
        y_tr_model,
        groups_tr,
        X_val,
        y_val_model,
        groups_val,
        test_X,
        test_y_model,
        test_groups,
    )

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H:%M")
    model_store = os.path.join(args.train_dir, f"deep_models_{run_name}")
    os.makedirs(model_store, exist_ok=True)

    device = get_device(args.device)
    model = make_model(args, input_dim=train_X.shape[2]).to(device)

    if args.task == "classification":
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        if pos_count == 0 or neg_count == 0:
            raise ValueError(
                "Training fold must contain both positive and negative examples."
            )
        pos_weight = torch.tensor(
            [neg_count / pos_count], device=device, dtype=torch.float32
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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

    minimize_metrics = {
        "val_loss",
        "val_mae",
        "val_rmse",
        "val_patient_mae",
        "val_patient_rmse",
    }
    history = []
    best_metric = np.inf if monitor_name in minimize_metrics else -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    best_path = os.path.join(model_store, f"{args.model}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
        )
        train_metrics = evaluate(
            model,
            train_eval_loader,
            criterion,
            device,
            groups_tr,
            args.task,
            target_stats=regression_target_stats,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            groups_val,
            args.task,
            target_stats=regression_target_stats,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if args.task == "classification":
            row |= {
                "train_auc": train_metrics["auc"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_patient_auc": train_metrics["patient_auc"],
                "train_patient_balanced_accuracy": train_metrics[
                    "patient_balanced_accuracy"
                ],
                "val_auc": val_metrics["auc"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_patient_auc": val_metrics["patient_auc"],
                "val_patient_balanced_accuracy": val_metrics[
                    "patient_balanced_accuracy"
                ],
            }
        else:
            row |= {
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "train_patient_mae": train_metrics["patient_mae"],
                "train_patient_rmse": train_metrics["patient_rmse"],
                "train_patient_r2": train_metrics["patient_r2"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_patient_mae": val_metrics["patient_mae"],
                "val_patient_rmse": val_metrics["patient_rmse"],
                "val_patient_r2": val_metrics["patient_r2"],
            }
        history.append(row)
        if wandb_run is not None:
            wandb.log(row, step=epoch)

        maximize = monitor_name not in minimize_metrics
        current_metric = metric_or_default(row[monitor_name], maximize=maximize)
        improved = (
            current_metric < best_metric
            if monitor_name in minimize_metrics
            else current_metric > best_metric
        )
        if improved or best_epoch == -1:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            # The checkpoint stores enough metadata to reload the exact trained
            # model configuration without relying on the W&B run state.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "channels": channels,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "monitor": monitor_name,
                    "task": args.task,
                    "target_col": target_col,
                    "target_stats": regression_target_stats,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_metrics = evaluate(
        model,
        train_eval_loader,
        criterion,
        device,
        groups_tr,
        args.task,
        target_stats=regression_target_stats,
    )
    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        groups_val,
        args.task,
        target_stats=regression_target_stats,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        test_groups,
        args.task,
        target_stats=regression_target_stats,
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

    history_path = os.path.join(model_store, f"{args.model}_history.csv")
    summary_path = os.path.join(model_store, f"{args.model}_summary.json")
    pd.DataFrame(history).to_csv(history_path, index=False)
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
