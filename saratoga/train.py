"""
train.py
========
Training loop for the RacePredictor model.

Trains on 2025 Saratoga data to predict Win / Place / Show probabilities.
Saves the best model checkpoint to model_best.pt.

Usage:
  python train.py
  python train.py --epochs 100 --lr 0.001
  python train.py --epochs 50  --batch 16

Evaluation metrics:
  - Top-1 Win accuracy  : did the highest win-prob horse actually win?
  - Top-2 Place accuracy: was the winner in the top 2 predicted?
  - Top-3 Show accuracy : was the winner in the top 3 predicted?
  - Loss                : masked cross-entropy (lower = better)
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from features import load_races, FEATURE_COLS, MAX_HORSES
from model    import build_model, masked_cross_entropy

DEVICE     = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("model_best.pt")


# ── Dataset ────────────────────────────────────────────────────────────────────

class RaceDataset(Dataset):
    def __init__(self, X, y, masks):
        self.X     = torch.tensor(X,     dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx]


# ── Metrics ────────────────────────────────────────────────────────────────────

def top_k_accuracy(probs, targets, mask, k: int, target_idx: int = 0) -> float:
    """
    What % of races did the model place the true winner in its top-k predictions?

    target_idx: 0=win, 1=place, 2=show
    """
    correct = 0
    total   = 0

    for i in range(len(probs)):
        m      = mask[i].bool()
        p      = probs[i, :, target_idx]
        t      = targets[i, :, target_idx]

        # Skip if no true winner in this race
        if t[m].sum() == 0:
            continue

        # True winner index
        true_winner = t[m].argmax().item()

        # Top-k predicted indices (among real horses only)
        real_probs = p[m]
        topk = real_probs.topk(min(k, m.sum().item())).indices.tolist()

        if true_winner in topk:
            correct += 1
        total += 1

    return correct / max(total, 1)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets, all_masks = [], [], []

    with torch.no_grad():
        for X, y, mask in loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            probs = model(X, mask)
            loss  = masked_cross_entropy(probs, y, mask)
            total_loss += loss.item()
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())
            all_masks.append(mask.cpu())

    probs   = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    masks   = torch.cat(all_masks)

    return {
        "loss":         total_loss / len(loader),
        "win_top1":     top_k_accuracy(probs, targets, masks, k=1, target_idx=0),
        "win_top3":     top_k_accuracy(probs, targets, masks, k=3, target_idx=0),
        "place_top2":   top_k_accuracy(probs, targets, masks, k=2, target_idx=1),
        "show_top3":    top_k_accuracy(probs, targets, masks, k=3, target_idx=2),
    }


# ── Training loop ──────────────────────────────────────────────────────────────

def train(epochs: int = 80, lr: float = 5e-4, batch_size: int = 32,
          val_split: float = 0.2, patience: int = 20) -> None:

    print(f"Device: {DEVICE}")
    print(f"Loading data...")
    X, y, masks, meta = load_races()

    dataset  = RaceDataset(X, y, masks)
    n_val    = int(len(dataset) * val_split)
    n_train  = len(dataset) - n_val

    # Chronological split — train on earlier races, validate on later ones
    # This avoids data leakage from future races into training
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds   = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"Train races: {n_train} | Val races: {n_val}")

    model     = build_model(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch    = 0

    print(f"\nTraining for {epochs} epochs...\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  "
          f"{'Win@1':>6}  {'Win@3':>6}  {'Place@2':>8}  {'Show@3':>7}")
    print("─" * 70)

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_b, y_b, mask_b in train_loader:
            X_b, y_b, mask_b = X_b.to(DEVICE), y_b.to(DEVICE), mask_b.to(DEVICE)
            optimizer.zero_grad()
            probs = model(X_b, mask_b)
            loss  = masked_cross_entropy(probs, y_b, mask_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, DEVICE)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {metrics['loss']:>9.4f}  "
              f"{metrics['win_top1']:>6.1%}  {metrics['win_top3']:>6.1%}  "
              f"{metrics['place_top2']:>8.1%}  {metrics['show_top3']:>7.1%}")

        # Save best model
        if metrics["loss"] < best_val_loss:
            best_val_loss     = metrics["loss"]
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val_loss,
                "metrics":     metrics,
            }, MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest model: epoch {best_epoch} | val loss: {best_val_loss:.4f}")
    print(f"Saved to {MODEL_PATH}")


# ── Predict (inference) ────────────────────────────────────────────────────────

def predict_race(entries: list, race_info: dict) -> list:
    """
    Predict Win/Place/Show probabilities for a single upcoming race.

    Args:
        entries   : list of dicts — one per horse, keys match FEATURE_COLS
        race_info : dict with race-level features (distance, surface, etc.)

    Returns:
        list of dicts sorted by win probability:
        [{"horse": name, "win_pct": 0.32, "place_pct": 0.48, "show_pct": 0.61}, ...]

    Example:
        entries = [
            {"horse_name": "Gratefully", "post_position": 6,
             "morning_line_decimal": 12.0, "hrn_speed_figure": 105, ...},
            ...
        ]
        race_info = {"distance_furlongs": 5.5, "surface_code": 1, ...}
    """
    from features import FEATURE_COLS, _normalize
    import pandas as pd

    if not MODEL_PATH.exists():
        raise FileNotFoundError("No model found — run: python train.py")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model      = build_model("cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Build feature dataframe
    rows = []
    for e in entries:
        row = {**race_info, **e}
        rows.append(row)
    df = pd.DataFrame(rows)

    # Normalize
    df = _normalize(df)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    n      = len(df)
    X_race = df[FEATURE_COLS].values.astype(np.float32)

    # Pad to MAX_HORSES
    if n < MAX_HORSES:
        pad    = MAX_HORSES - n
        X_race = np.vstack([X_race, np.zeros((pad, len(FEATURE_COLS)), dtype=np.float32)])
    mask = np.array([True] * n + [False] * (MAX_HORSES - n))

    X_t    = torch.tensor(X_race[np.newaxis], dtype=torch.float32)
    mask_t = torch.tensor(mask[np.newaxis],   dtype=torch.bool)

    with torch.no_grad():
        probs = model(X_t, mask_t)[0].numpy()  # (MAX_HORSES, 3)

    results = []
    for i, entry in enumerate(entries):
        results.append({
            "horse":     entry.get("horse_name", f"Horse {i+1}"),
            "win_pct":   float(probs[i, 0]),
            "place_pct": float(probs[i, 1]),
            "show_pct":  float(probs[i, 2]),
        })

    return sorted(results, key=lambda x: x["win_pct"], reverse=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RacePredictor model")
    parser.add_argument("--epochs", type=int,   default=80)
    parser.add_argument("--lr",     type=float, default=5e-4)
    parser.add_argument("--batch",  type=int,   default=32)
    args = parser.parse_args()
    train(args.epochs, args.lr, args.batch)
