"""Patient-balanced Trainer subclass and TrainingArguments builder."""

import os
from inspect import signature

import numpy as np
import wandb
from transformers import Trainer, TrainingArguments

from deep.data import build_weighted_sampler
from deep.metrics import (
    SplitMetricsConfig,
    metric_direction,
    normalize_metric_values,
    predict_split_metrics,
    training_metric_name,
)

TRAINING_ARGUMENT_PARAMS = signature(TrainingArguments.__init__).parameters


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
