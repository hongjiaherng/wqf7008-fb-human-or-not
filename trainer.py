from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import numpy as np


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds  = (all_probs >= threshold).astype(int)

    return {
        "loss":      total_loss / len(loader.dataset),
        "auc":       roc_auc_score(all_labels, all_probs),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
    }


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float | None = None,
    patience: int = 10,
    device: torch.device | None = None,
    verbose: bool = True,
    model_name: str = "model",
    fold_tag: str = "fold_1",
) -> dict:
    """
    Train any BaseClassifier model, saving the best checkpoint to
    ../weights/{model_name}-{fold_tag}.pt.

    Args:
        model:        MLP, CNN, or any nn.Module with compatible forward().
        train_loader: DataLoader for training set.
        val_loader:   DataLoader for validation set.
        epochs:       Maximum training epochs.
        lr:           Learning rate.
        weight_decay: L2 regularization.
        pos_weight:   Scalar weight for positive class in BCEWithLogitsLoss.
                      Pass (n_negatives / n_positives) to handle imbalance.
        patience:     Early stopping patience (epochs without val loss improvement).
        device:       torch.device. Auto-detects if None.
        verbose:      Print per-epoch metrics.
        model_name:   Model identifier used in the checkpoint filename.
        fold_tag:     Fold identifier used in the checkpoint filename,
                      e.g. 'fold_1' or 'final'.

    Returns:
        history dict with train/val metrics per epoch, and best val metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")

    model = model.to(device)

    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    weights_dir = Path("../weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / f"{model_name}-{fold_tag}.pt"

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    best_val_loss = float("inf")
    best_metrics  = {}
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics  = val_metrics
            no_improve    = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    history["best"] = best_metrics
    return history


# ── Plot Model Loss ────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
 
 
def plot_loss_curves(model_name: str, all_histories: list[tuple[str, dict]]) -> None:
    """
    Plot train and validation loss curves for all folds and the final run.
 
    Args:
        model_name:    Model identifier, used in the title and saved filename.
        all_histories: List of (tag, history) tuples collected during CV,
                       e.g. [("fold_1", hist), ..., ("final", hist)].
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name.upper()} — Loss vs Epochs", fontsize=14)
 
    for tag, hist in all_histories:
        epochs_range = range(1, len(hist["train_loss"]) + 1)
        is_final = tag == "final"
        kwargs = dict(
            linestyle = "--" if is_final else "-",
            linewidth = 2.0  if is_final else 1.0,
            alpha     = 1.0  if is_final else 0.7,
            label     = tag,
        )
        axes[0].plot(epochs_range, hist["train_loss"], **kwargs)
        axes[1].plot(epochs_range, hist["val_loss"],   **kwargs)
 
    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(f"../weights/{model_name}_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.show()