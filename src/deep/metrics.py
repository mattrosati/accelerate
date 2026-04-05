"""Metric computation, naming conventions, and direction helpers.

All metric-related logic for the deep learning pipeline lives here so that
trainer.py, train.py, and search.py share a single source of truth.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from patient_utils import make_patient_weights  # noqa: E402

# ---------------------------------------------------------------------------
# Metric naming and direction
# ---------------------------------------------------------------------------

VALID_MONITOR_NAMES = (
    "val_patient_auc",
    "val_auc",
    "val_loss",
    "val_mae",
    "val_rmse",
    "val_r2",
    "val_patient_mae",
    "val_patient_rmse",
    "val_patient_r2",
    "val_balanced_accuracy",
    "val_patient_balanced_accuracy",
)

_MINIMIZE_BARE = frozenset({"loss", "mae", "rmse", "patient_mae", "patient_rmse"})


def is_maximize(metric_name):
    """Whether a metric should be maximized.  Works with any prefix."""
    bare = metric_name.removeprefix("val_").removeprefix("eval_")
    return bare not in _MINIMIZE_BARE


def metric_direction(metric_name):
    """Return whether a metric should be maximized (legacy interface)."""
    return is_maximize(metric_name)


def optuna_direction(metric_name):
    """Return ``'maximize'`` or ``'minimize'`` for Optuna."""
    return "maximize" if is_maximize(metric_name) else "minimize"


def training_metric_name(monitor_name):
    """Map repository-style validation metrics to Trainer metric names."""
    if not monitor_name.startswith("val_"):
        raise ValueError(f"Unsupported monitor name: {monitor_name}")
    return f"eval_{monitor_name.removeprefix('val_')}"


def resolve_monitor(task, metric_name, y_val=None):
    """Pick a monitor metric compatible with the training task.

    When *y_val* is ``None`` the single-class fallback check is skipped,
    which is useful for the search entrypoint that doesn't hold y_val at
    metric-resolution time.
    """
    if task == "regression" and metric_name in {"val_auc", "val_patient_auc"}:
        warnings.warn(
            f"Switching monitor from {metric_name} to val_patient_rmse for regression.",
            stacklevel=2,
        )
        return "val_patient_rmse"
    if task == "multiclass" and metric_name in {"val_auc", "val_patient_auc"}:
        warnings.warn(
            f"Switching monitor from {metric_name} to val_patient_balanced_accuracy for multiclass.",
            stacklevel=2,
        )
        return "val_patient_balanced_accuracy"
    if (
        y_val is not None
        and task in ("classification", "multiclass")
        and np.unique(y_val).shape[0] < 2
        and metric_name in {"val_auc", "val_patient_auc"}
    ):
        warnings.warn(
            f"Validation fold has a single class; switching monitor from {metric_name} "
            "to val_loss.",
            stacklevel=2,
        )
        return "val_loss"
    return metric_name


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitMetricsConfig:
    """Describe one split that should be scored outside the built-in eval loop."""

    name: str
    dataset: object = None
    groups: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Logit / prediction helpers
# ---------------------------------------------------------------------------


def logits_to_predictions(logits):
    """Convert logits to probabilities and hard predictions.

    For binary (1-D logits): sigmoid → threshold 0.5.
    For multiclass (2-D logits with C>1): softmax → argmax.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 2 and logits.shape[1] > 1:
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        preds = probs.argmax(axis=1)
        return probs, preds
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def metric_or_default(value, maximize):
    """Map ``NaN`` monitor values to deterministic worst-case sentinels."""
    if np.isnan(value):
        return -np.inf if maximize else np.inf
    return value


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(loss, y_true, y_prob, y_pred, groups, patient_balance=True):
    """Compute window-level and patient-balanced binary classification metrics."""
    if y_true.size == 0:
        metrics = {
            "loss": float(loss),
            "auc": np.nan,
            "balanced_accuracy": np.nan,
        }
        if patient_balance:
            metrics["patient_balanced_accuracy"] = np.nan
            metrics["patient_auc"] = np.nan
        return metrics

    metrics = {
        "loss": float(loss),
        "auc": np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if np.unique(y_true).shape[0] > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)

    if patient_balance:
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


def compute_multiclass_metrics(loss, y_true, y_prob, y_pred, groups, patient_balance=True):
    """Compute window-level and patient-balanced multiclass classification metrics.

    ``y_prob`` is a ``(N, C)`` probability matrix from softmax.
    """
    if y_true.size == 0:
        metrics = {
            "loss": float(loss),
            "auc": np.nan,
            "balanced_accuracy": np.nan,
        }
        if patient_balance:
            metrics["patient_balanced_accuracy"] = np.nan
            metrics["patient_auc"] = np.nan
        return metrics

    metrics = {
        "loss": float(loss),
        "auc": np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    n_classes = len(np.unique(y_true))
    if n_classes > 1 and y_prob.ndim == 2 and y_prob.shape[1] > 1:
        try:
            metrics["auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["auc"] = np.nan

    if patient_balance:
        patient_weights = make_patient_weights(groups)
        metrics["patient_balanced_accuracy"] = np.nan
        if patient_weights.size > 0:
            metrics["patient_balanced_accuracy"] = balanced_accuracy_score(
                y_true, y_pred, sample_weight=patient_weights
            )
        metrics["patient_auc"] = np.nan
        if patient_weights.size > 0 and n_classes > 1 and y_prob.ndim == 2:
            try:
                metrics["patient_auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro",
                    sample_weight=patient_weights,
                )
            except ValueError:
                metrics["patient_auc"] = np.nan

    return metrics


def compute_regression_metrics(loss, y_true, y_pred, groups, patient_balance=True):
    """Compute window-level and patient-balanced regression metrics."""
    if y_true.size == 0:
        metrics = {
            "loss": float(loss),
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }
        if patient_balance:
            metrics["patient_mae"] = np.nan
            metrics["patient_rmse"] = np.nan
            metrics["patient_r2"] = np.nan
        return metrics

    metrics = {
        "loss": float(loss),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": np.nan,
    }
    if y_true.shape[0] > 1 and np.unique(y_true).shape[0] > 1:
        metrics["r2"] = r2_score(y_true, y_pred)

    if patient_balance:
        patient_weights = make_patient_weights(groups)
        metrics["patient_mae"] = np.nan
        metrics["patient_rmse"] = np.nan
        metrics["patient_r2"] = np.nan
        if patient_weights.size > 0:
            metrics["patient_mae"] = mean_absolute_error(
                y_true, y_pred, sample_weight=patient_weights
            )
            metrics["patient_rmse"] = float(
                np.sqrt(
                    mean_squared_error(y_true, y_pred, sample_weight=patient_weights)
                )
            )
            if y_true.shape[0] > 1 and np.unique(y_true).shape[0] > 1:
                metrics["patient_r2"] = r2_score(
                    y_true, y_pred, sample_weight=patient_weights
                )

    return metrics


# ---------------------------------------------------------------------------
# GroupAwareMetricsComputer
# ---------------------------------------------------------------------------


class GroupAwareMetricsComputer:
    """Compute metrics for Trainer while preserving patient grouping."""

    def __init__(self, task, target_stats=None, patient_balance=True):
        self.task = task
        self.target_stats = target_stats
        self.patient_balance = patient_balance
        self.groups = np.array([], dtype=str)
        self.last_outputs = {}

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

        if self.task in ("classification", "multiclass"):
            y_true = labels.astype(int)
            y_prob, y_pred = logits_to_predictions(predictions)
            self.last_outputs = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }
            if self.task == "multiclass":
                metrics = compute_multiclass_metrics(
                    np.nan, y_true, y_prob, y_pred, self.groups, self.patient_balance
                )
            else:
                metrics = compute_metrics(
                    np.nan, y_true, y_prob, y_pred, self.groups, self.patient_balance
                )
        else:
            y_true = labels.astype(np.float64)
            y_pred = predictions.astype(np.float64)
            if self.target_stats is not None:
                y_true = y_true * self.target_stats["std"] + self.target_stats["mean"]
                y_pred = y_pred * self.target_stats["std"] + self.target_stats["mean"]
            self.last_outputs = {
                "y_true": y_true,
                "y_pred": y_pred,
            }
            metrics = compute_regression_metrics(
                np.nan, y_true, y_pred, self.groups, self.patient_balance
            )

        metrics.pop("loss", None)
        return metrics

    def get_last_outputs(self):
        """Return the most recent labels/predictions seen by ``compute_metrics``."""
        return self.last_outputs


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------


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


def predict_split_metrics(trainer, dataset, groups, prefix, metrics_computer):
    """Run Trainer prediction for one split and return normalized metrics."""
    metrics_computer.set_groups(groups)
    prediction_output = trainer.predict(dataset, metric_key_prefix=prefix)
    return extract_prefixed_metrics(prediction_output.metrics, prefix)


# ---------------------------------------------------------------------------
# History / best-epoch extraction
# ---------------------------------------------------------------------------


def extract_best_epoch(log_history, metric_name, maximize):
    """Recover the best epoch from Trainer log history."""
    candidates = [
        entry for entry in log_history if metric_name in entry and "epoch" in entry
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
            if key.startswith("train_"):
                row[key] = float(value)
        for key, value in entry.items():
            if key.startswith("eval_") and key not in ignored_eval_fields:
                row[f"val_{key.removeprefix('eval_')}"] = float(value)

    if not rows:
        return pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "lr"])
    return pd.DataFrame(rows.values()).sort_values("epoch").reset_index(drop=True)
