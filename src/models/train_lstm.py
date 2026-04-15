"""
Training loop for GaitLSTM / GaitBiLSTM.

Features:
  - Weighted CrossEntropyLoss to handle class imbalance
  - Adam optimiser with ReduceLROnPlateau scheduler
  - Early stopping (patience=20)
  - Returns best model weights (lowest val loss)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    epochs:        int   = 200
    batch_size:    int   = 64
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    patience:      int   = 30          # early stopping
    lr_patience:   int   = 10          # ReduceLROnPlateau patience
    lr_factor:     float = 0.5
    monitor:       str   = "val_acc"   # "val_loss" or "val_acc"
    device:        str   = field(default_factory=_default_device)
    verbose:       bool  = True
    log_every:     int   = 10          # print every N epochs


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    train_acc:  list[float] = field(default_factory=list)
    val_acc:    list[float] = field(default_factory=list)
    best_epoch: int = 0


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool) -> DataLoader:
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                      shuffle=shuffle)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: np.ndarray | None = None,
    cfg: TrainConfig | None = None,
) -> tuple[nn.Module, TrainHistory]:
    """Train model with early stopping; return best checkpoint + history.

    Args:
        model:         GaitLSTM / GaitBiLSTM instance
        X_train:       (N_train, 101, 4) float32
        y_train:       (N_train,) int
        X_val:         (N_val, 101, 4) float32
        y_val:         (N_val,) int
        class_weights: (3,) array — passed to CrossEntropyLoss weight
        cfg:           TrainConfig (uses defaults if None)

    Returns:
        best_model, history
    """
    if cfg is None:
        cfg = TrainConfig()

    device = torch.device(cfg.device)
    model  = model.to(device)

    # Loss with class weights
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        w = None
    criterion = nn.CrossEntropyLoss(weight=w)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=cfg.lr_factor,
        patience=cfg.lr_patience,
    )

    train_loader = _make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   cfg.batch_size, shuffle=False)

    history     = TrainHistory()
    best_val    = -float("inf")   # we always maximise (acc or negative loss)
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        t_loss /= t_total
        t_acc   = t_correct / t_total

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = criterion(logits, yb)
                v_loss    += loss.item() * len(yb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total   += len(yb)

        v_loss /= v_total
        v_acc   = v_correct / v_total

        scheduler.step(v_loss)

        history.train_loss.append(t_loss)
        history.val_loss.append(v_loss)
        history.train_acc.append(t_acc)
        history.val_acc.append(v_acc)

        if cfg.verbose and epoch % cfg.log_every == 0:
            lr = optimiser.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{cfg.epochs} | "
                  f"train loss={t_loss:.4f} acc={t_acc:.3f} | "
                  f"val loss={v_loss:.4f} acc={v_acc:.3f} | lr={lr:.2e}")

        # ── Early stopping (monitor val_acc or val_loss) ──
        monitor_val = v_acc if cfg.monitor == "val_acc" else -v_loss
        if monitor_val > best_val + 1e-5:
            best_val     = monitor_val
            best_weights = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                if cfg.verbose:
                    print(f"  Early stop at epoch {epoch} "
                          f"(best epoch={history.best_epoch})")
                break

    model.load_state_dict(best_weights)
    return model, history


def predict(model: nn.Module, X: np.ndarray,
            device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, y_proba) for input array X."""
    model.eval()
    dev   = torch.device(device)
    model = model.to(dev)
    Xt    = torch.tensor(X, dtype=torch.float32).to(dev)

    with torch.no_grad():
        logits = model(Xt)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        pred   = np.argmax(proba, axis=1)

    return pred, proba