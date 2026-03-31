"""Persistence helpers: checkpoint saving, JSON I/O, and W&B artifact logging."""

import json
import re

import torch
import wandb


def save_json(path, payload):
    """Write a JSON payload with stable formatting for later review."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def sanitize_wandb_artifact_name(value):
    """Return an artifact-safe name for W&B logging."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    sanitized = sanitized.strip("-.")
    return sanitized or "artifact"


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
        summary = {
            "final_train_auc": train_metrics["auc"],
            "final_val_auc": val_metrics["auc"],
            "final_test_auc": test_metrics["auc"],
        }
        if "patient_auc" in val_metrics:
            summary["final_val_patient_auc"] = val_metrics["patient_auc"]
            summary["final_test_patient_auc"] = test_metrics["patient_auc"]
        wandb.summary.update(summary)
        return

    summary = {
        "final_train_rmse": train_metrics["rmse"],
        "final_val_rmse": val_metrics["rmse"],
        "final_test_rmse": test_metrics["rmse"],
    }
    if "patient_rmse" in val_metrics:
        summary["final_val_patient_rmse"] = val_metrics["patient_rmse"]
        summary["final_test_patient_rmse"] = test_metrics["patient_rmse"]
    wandb.summary.update(summary)


def log_wandb_artifacts(
    wandb_run,
    args,
    run_name,
    best_path,
    history_path,
    summary_path,
    monitor_name,
    best_epoch,
    target_col,
):
    """Upload model artifacts and finalize the W&B run."""
    if wandb_run is None:
        return

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
