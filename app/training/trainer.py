"""
Training loop for metric learning with online hard triplet mining.
Designed to run on Google Colab GPU.
"""
import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from app.training.triplet_loss import batch_hard_triplet_loss
from app.training.models import MetricModel


def evaluate(model: MetricModel, loader: DataLoader, device: torch.device) -> dict:
    """
    Compute validation loss and Recall@1 (nearest-neighbour accuracy).
    """
    model.eval()
    all_embs, all_labels = [], []

    with torch.no_grad():
        val_loss, val_steps = 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            embs = model(imgs)
            loss, _ = batch_hard_triplet_loss(embs, labels)
            val_loss += loss.item()
            val_steps += 1
            all_embs.append(embs.cpu())
            all_labels.append(labels.cpu())

    all_embs = torch.cat(all_embs)     # (N, D)
    all_labels = torch.cat(all_labels) # (N,)

    # Recall@1: for each embedding, find nearest neighbour (excluding self)
    sim = all_embs @ all_embs.T  # (N, N) cosine similarity (L2-normalized)
    sim.fill_diagonal_(-1e9)
    nn_labels = all_labels[sim.argmax(dim=1)]
    recall_1 = (nn_labels == all_labels).float().mean().item()

    return {
        "val_loss": val_loss / max(val_steps, 1),
        "recall@1": recall_1,
    }


def train(
    model_name: str,
    dataset_dir: str,
    save_dir: str,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    margin: float = 0.3,
    device_str: str = "cuda",
):
    from app.training.dataset import build_splits, CarDataset
    from app.training.dataset import (
        TRAIN_TRANSFORMS, VAL_TRANSFORMS,
        TRAIN_TRANSFORMS_518, VAL_TRANSFORMS_518,
    )
    from app.training.models import MODEL_CONFIGS

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {device}")
    print(f"{'='*60}")

    # Dataset
    train_paths, val_paths = build_splits(dataset_dir, val_ratio=0.2)
    use_518 = (model_name == "dinov2")
    t_tf = TRAIN_TRANSFORMS_518 if use_518 else TRAIN_TRANSFORMS
    v_tf = VAL_TRANSFORMS_518 if use_518 else VAL_TRANSFORMS

    train_ds = CarDataset(train_paths, transform=t_tf)
    val_ds   = CarDataset(val_paths,   transform=v_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images | Classes: {train_ds.num_classes}")

    # Model
    model = MetricModel(model_name).to(device)
    print(f"Trainable params: {model.count_trainable():,}")

    optimizer = AdamW(model.trainable_params(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    history = []
    best_recall = 0.0
    best_path = os.path.join(save_dir, f"{model_name}_best.pth")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss, epoch_violated = 0.0, 0.0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embs = model(imgs)
            loss, metrics = batch_hard_triplet_loss(embs, labels, margin=margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), max_norm=1.0)
            optimizer.step()

            epoch_loss += metrics["loss"]
            epoch_violated += metrics["fraction_violated"]

        scheduler.step()
        n_batches = len(train_loader)
        train_loss = epoch_loss / n_batches
        violated   = epoch_violated / n_batches

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["val_loss"], 4),
            "recall@1": round(val_metrics["recall@1"], 4),
            "fraction_violated": round(violated, 3),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "time_s": round(elapsed, 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_metrics['val_loss']:.4f} | "
            f"R@1: {val_metrics['recall@1']:.3f} | "
            f"Violated: {violated:.2f} | "
            f"{elapsed:.0f}s"
        )

        # Save best checkpoint
        if val_metrics["recall@1"] > best_recall:
            best_recall = val_metrics["recall@1"]
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "history": history,
            }, best_path)
            print(f"  → Saved best model (R@1={best_recall:.3f})")

    # Save training history
    hist_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Recall@1: {best_recall:.3f} | Saved to {best_path}")
    return history, best_recall
