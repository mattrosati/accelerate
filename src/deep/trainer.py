"""Trainer and metric helpers for recurrent deep models."""

import os
import warnings
from dataclasses import dataclass
from inspect import signature

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
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import ProgressCallback

from deep.data import build_weighted_sampler, make_patient_weights

TRAINING_ARGUMENT_PARAMS = signature(TrainingArguments.__init__).parameters


@dataclass(frozen=True)
class SplitMetricsConfig:
    """Describe one split that should be scored outside the built-in eval loop."""

    name: str
    dataset: object = None
    groups: np.ndarray | None = None


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


class GroupAwareMetricsComputer:
    """Compute metrics for Trainer while preserving patient grouping."""

    def __init__(self, task, target_stats=None):
        self.task = task
        self.target_stats = target_stats
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

        if self.task == "classification":
            y_true = labels.astype(int)
            y_prob, y_pred = logits_to_predictions(predictions)
            self.last_outputs = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }
            metrics = compute_metrics(np.nan, y_true, y_prob, y_pred, self.groups)
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
            metrics = compute_regression_metrics(np.nan, y_true, y_pred, self.groups)

        metrics.pop("loss", None)
        return metrics

    def get_last_outputs(self):
        """Return the most recent labels/predictions seen by `compute_metrics`."""
        return self.last_outputs


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


class PatientBalancedTrainer(Trainer):
    """Trainer subclass that preserves inverse-frequency patient sampling."""

    def __init__(
        self,
        *args,
        patient_balance="none",
        train_groups=None,
        val_groups=None,
        train_metrics_dataset=None,
        metrics_computer=None,
        wandb_run=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patient_balance = patient_balance
        self.val_groups = None if val_groups is None else np.asarray(val_groups)
        self.metrics_computer = metrics_computer
        self.wandb_run = wandb_run
        self.train_metrics_split = SplitMetricsConfig(
            name="train",
            dataset=train_metrics_dataset,
            groups=None if train_groups is None else np.asarray(train_groups),
        )

    def _get_train_sampler(self, train_dataset=None):
        if self.patient_balance != "sampler":
            return super()._get_train_sampler(train_dataset)
        if self.train_metrics_split.groups is None:
            return super()._get_train_sampler(train_dataset)
        return build_weighted_sampler(self.train_metrics_split.groups)

    def _set_metric_groups(self, groups):
        if self.metrics_computer is not None:
            self.metrics_computer.set_groups(groups)

    def _make_epoch_log_payload(self, metrics):
        payload = normalize_metric_values(metrics)
        if self.state.epoch is not None:
            payload["epoch"] = float(self.state.epoch)
        return payload

    def _append_metrics_to_history(self, metrics):
        payload = self._make_epoch_log_payload(metrics)
        history_entry = dict(payload)
        history_entry["step"] = self.state.global_step
        self.state.log_history.append(history_entry)
        return payload

    def _predict_split_metrics(self, split_config):
        if (
            split_config.dataset is None
            or split_config.groups is None
            or self.metrics_computer is None
        ):
            return None
        return predict_split_metrics(
            self,
            split_config.dataset,
            split_config.groups,
            split_config.name,
            self.metrics_computer,
        )

    def _record_split_metrics(self, split_config):
        metrics = self._predict_split_metrics(split_config)
        if not metrics:
            return
        payload = self._append_metrics_to_history(metrics)
        self._log_to_wandb(payload)

    def _log_to_wandb(self, payload):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload)

    def _log_validation_confusion_matrix(self):
        if self.metrics_computer is None:
            return
        if getattr(self.metrics_computer, "task", None) != "classification":
            return

        outputs = self.metrics_computer.get_last_outputs()
        if not outputs:
            return

        self._log_to_wandb(
            {
                "val_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=outputs["y_true"].tolist(),
                    preds=outputs["y_pred"].tolist(),
                    class_names=["0", "1"],
                )
            }
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        self._set_metric_groups(self.val_groups)
        try:
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self._set_metric_groups(self.val_groups)

        if metric_key_prefix == "eval":
            self._log_validation_confusion_matrix()
            self._record_split_metrics(self.train_metrics_split)

        return metrics


class QuietProgressCallback(ProgressCallback):
    """Keep tqdm progress bars while suppressing the default metric dict prints."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control


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


def build_training_arguments(
    args, model_store, run_name, monitor_name, report_to, no_cuda
):
    """Create TrainingArguments configured for epoch-level evaluation."""
    training_kwargs = {
        "output_dir": model_store,
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": max(args.batch_size * 2, 1),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": training_metric_name(monitor_name),
        "greater_is_better": metric_direction(training_metric_name(monitor_name)),
        "dataloader_num_workers": args.num_workers,
        "seed": args.seed,
        "data_seed": args.seed,
        "max_grad_norm": max(args.grad_clip, 0.0),
        "report_to": report_to,
        "run_name": f"{args.model}_{run_name}",
        "logging_dir": os.path.join(model_store, "logs"),
        "remove_unused_columns": True,
    }
    if "evaluation_strategy" in TRAINING_ARGUMENT_PARAMS:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"
    if "no_cuda" in TRAINING_ARGUMENT_PARAMS:
        training_kwargs["no_cuda"] = no_cuda
    elif "use_cpu" in TRAINING_ARGUMENT_PARAMS:
        training_kwargs["use_cpu"] = no_cuda

    return TrainingArguments(**training_kwargs)


def save_trainer_checkpoint(
    best_path,
    trainer,
    args,
    channels,
    best_epoch,
    best_metric,
    monitor_name,
    target_col,
    target_stats,
):
    """Persist the best trained model and metadata in the repository format."""
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
            "target_stats": target_stats,
            "trainer_best_checkpoint": trainer.state.best_model_checkpoint,
        },
        best_path,
    )


def update_wandb_summary(
    wandb_run, args, best_epoch, best_metric, train_metrics, val_metrics, test_metrics
):
    """Write the final training summary to W&B."""
    if wandb_run is None:
        return

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
        return

    wandb.summary.update(
        {
            "final_train_rmse": train_metrics["rmse"],
            "final_val_rmse": val_metrics["rmse"],
            "final_val_patient_rmse": val_metrics["patient_rmse"],
            "final_test_rmse": test_metrics["rmse"],
            "final_test_patient_rmse": test_metrics["patient_rmse"],
        }
    )
