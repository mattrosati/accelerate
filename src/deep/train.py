import json
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(args, input_dim):
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
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def predict_scores(model, dataloader, device):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy().astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return probs, preds, targets


def compute_metrics(loss, y_true, y_prob, y_pred, groups):
    metrics = {
        "loss": float(loss),
        "auc": np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if np.unique(y_true).shape[0] > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)

    patient_weights = make_patient_weights(groups)
    metrics["patient_balanced_accuracy"] = balanced_accuracy_score(
        y_true, y_pred, sample_weight=patient_weights
    )
    metrics["patient_auc"] = np.nan
    if np.unique(y_true).shape[0] > 1:
        metrics["patient_auc"] = roc_auc_score(
            y_true, y_prob, sample_weight=patient_weights
        )

    return metrics


def evaluate(model, dataloader, criterion, device, groups):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.shape[0]
            total_examples += X_batch.shape[0]

    y_prob, y_pred, y_true = predict_scores(model, dataloader, device)
    avg_loss = total_loss / max(total_examples, 1)
    return compute_metrics(avg_loss, y_true, y_prob, y_pred, groups)


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip):
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = ArgumentParser()
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
        choices=["val_patient_auc", "val_auc", "val_loss"],
        default="val_patient_auc",
    )
    parser.add_argument("--wandb_project", type=str, default="accelerate-deep")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    train_X, train_y, train_groups, train_labels, channels = load_split_arrays(
        args.train_dir, "train", data_mode=args.data_mode
    )
    test_X, test_y, test_groups, _, _ = load_split_arrays(
        args.train_dir, "test", data_mode=args.data_mode
    )

    train_idx, val_idx = make_grouped_split(
        train_y,
        train_groups,
        n_splits=args.n_splits,
        seed=args.seed,
        fold_idx=args.fold_idx,
    )

    X_tr, X_val = train_X[train_idx], train_X[val_idx]
    y_tr, y_val = train_y[train_idx], train_y[val_idx]
    groups_tr, groups_val = train_groups[train_idx], train_groups[val_idx]

    train_ds = SequenceDataset(X_tr, y_tr, groups_tr)
    val_ds = SequenceDataset(X_val, y_val, groups_val)
    test_ds = SequenceDataset(test_X, test_y, test_groups)

    train_sampler = None
    shuffle = True
    if args.patient_balance == "sampler":
        train_sampler = build_weighted_sampler(groups_tr)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_batch_size = min(args.batch_size * 2, len(val_ds)) if len(val_ds) > 0 else args.batch_size
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

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H:%M")
    model_store = os.path.join(args.train_dir, f"deep_models_{run_name}")
    os.makedirs(model_store, exist_ok=True)

    device = get_device(args.device)
    model = make_model(args, input_dim=train_X.shape[2]).to(device)

    pos_count = y_tr.sum()
    neg_count = len(y_tr) - pos_count
    if pos_count == 0:
        raise ValueError("Training fold contains no positive examples.")
    pos_weight = torch.tensor([neg_count / pos_count], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
        "mode": "disabled" if args.wandb_mode == "disabled" else args.wandb_mode,
        "dir": model_store,
    }
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    wandb.init(**wandb_kwargs)

    history = []
    best_metric = -np.inf if args.monitor != "val_loss" else np.inf
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
        train_metrics = evaluate(model, train_eval_loader, criterion, device, groups_tr)
        val_metrics = evaluate(model, val_loader, criterion, device, groups_val)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_metrics["auc"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_patient_auc": train_metrics["patient_auc"],
            "train_patient_balanced_accuracy": train_metrics[
                "patient_balanced_accuracy"
            ],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_patient_auc": val_metrics["patient_auc"],
            "val_patient_balanced_accuracy": val_metrics[
                "patient_balanced_accuracy"
            ],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        wandb.log(row, step=epoch)

        current_metric = row[args.monitor]
        improved = current_metric < best_metric if args.monitor == "val_loss" else current_metric > best_metric
        if improved or best_epoch == -1:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "channels": channels,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_metrics = evaluate(model, train_eval_loader, criterion, device, groups_tr)
    val_metrics = evaluate(model, val_loader, criterion, device, groups_val)
    test_metrics = evaluate(model, test_loader, criterion, device, test_groups)

    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "monitor": args.monitor,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    pd.DataFrame(history).to_csv(
        os.path.join(model_store, f"{args.model}_history.csv"), index=False
    )
    save_json(os.path.join(model_store, f"{args.model}_summary.json"), summary)

    wandb.summary.update(
        {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "final_train_auc": train_metrics["auc"],
            "final_val_auc": val_metrics["auc"],
            "final_val_patient_auc": val_metrics["patient_auc"],
            "final_test_auc": test_metrics["auc"],
            "final_test_patient_auc": test_metrics["patient_auc"],
        }
    )
    if args.wandb_mode != "disabled":
        artifact = wandb.Artifact(
            name=f"{args.model}_{run_name}",
            type="model",
            metadata={
                "train_dir": args.train_dir,
                "monitor": args.monitor,
                "best_epoch": best_epoch,
            },
        )
        artifact.add_file(best_path)
        artifact.add_file(os.path.join(model_store, f"{args.model}_history.csv"))
        artifact.add_file(os.path.join(model_store, f"{args.model}_summary.json"))
        wandb.log_artifact(artifact)
    wandb.finish()

    print(f"Saved deep model artifacts to {model_store}")
    print(json.dumps(summary, indent=2, sort_keys=True))
