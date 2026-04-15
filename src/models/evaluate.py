"""
Evaluation utilities and comparative report.

Generates:
  - Per-model: confusion matrix, per-class metrics (precision, recall, F1)
  - Comparative table: all models side-by-side
  - Learning curves for LSTM / Bi-LSTM
  - Feature importance bar chart (XGBoost, RF)
  - ROC curves (one-vs-rest)

All figures saved to data/output/figures/models/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)

OUT_DIR = Path("data/output/figures/models")
LABELS  = ["NM", "KOA", "PD"]
COLORS  = {
    "NM": "#2ecc71", "KOA": "#e74c3c", "PD": "#3498db",
    "EL": "#f39c12", "MD": "#e74c3c",  "SV": "#8e44ad",
}


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray | None = None,
                    label_names: list[str] | None = None) -> dict:
    """Return a dict with accuracy, per-class F1, macro F1, AUC (if proba given).

    Works for both binary and multi-class problems.
    """
    y_pred  = np.asarray(y_pred).ravel().astype(int)
    y_true  = np.asarray(y_true).ravel().astype(int)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if label_names is None:
        from src.models.dataset import LABEL_GROUP, LABEL_STAGE
        # Try to infer label names from class indices
        if max(classes) <= 2 and len(classes) <= 3:
            label_names = [LABEL_GROUP.get(c, str(c)) for c in range(3)]
        else:
            label_names = [str(c) for c in classes]

    acc   = accuracy_score(y_true, y_pred)
    f1_m  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, zero_division=0,
                     labels=list(range(len(label_names))))

    result: dict = {
        "accuracy":    round(acc,  4),
        "macro_f1":    round(f1_m, 4),
        "weighted_f1": round(f1_w, 4),
    }
    for i, name in enumerate(label_names):
        result[f"f1_{name}"] = round(float(f1_pc[i]) if i < len(f1_pc) else 0.0, 4)

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                # Binary: use positive class probability
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                    average="macro")
            result["roc_auc_macro"] = round(auc, 4)
        except Exception:
            result["roc_auc_macro"] = float("nan")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-model plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str,
                          label_names: list[str] | None = None) -> Path:
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if label_names is None:
        label_names = LABELS[:len(classes)]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in zip(
        axes,
        [cm,        cm_norm],
        ["d",       ".2f"],
        ["Contagens", "Proporção (por classe real)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=ax, cbar=True,
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(title)

    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    return _save(fig, f"cm_{model_name.replace(' ', '_')}.png")


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray,
                    model_name: str,
                    label_names: list[str] | None = None) -> Path:
    classes = sorted(np.unique(y_true))
    if label_names is None:
        label_names = LABELS[:len(classes)]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, label in enumerate(label_names):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        try:
            auc = roc_auc_score(y_bin, y_proba[:, i])
        except Exception:
            auc = float("nan")
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(label, "#555555"),
                label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (one-vs-rest) — {model_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return _save(fig, f"roc_{model_name.replace(' ', '_')}.png")


def plot_learning_curves(history, model_name: str) -> Path:
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.train_loss, label="Train")
    axes[0].plot(epochs, history.val_loss,   label="Val")
    axes[0].axvline(history.best_epoch, color="gray", lw=1, ls="--",
                    label=f"Best (ep {history.best_epoch})")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.train_acc, label="Train")
    axes[1].plot(epochs, history.val_acc,   label="Val")
    axes[1].axvline(history.best_epoch, color="gray", lw=1, ls="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Learning curves — {model_name}", fontsize=11)
    plt.tight_layout()
    return _save(fig, f"learning_{model_name.replace(' ', '_')}.png")


def plot_feature_importance(importance_df, model_name: str, top_n: int = 20) -> Path:
    if importance_df.empty:
        return None
    df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_n} Features — {model_name}")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _save(fig, f"feat_imp_{model_name.replace(' ', '_')}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Comparative summary
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison_table(results: dict[str, dict]) -> Path:
    """Bar chart comparing all models on key metrics."""
    import pandas as pd

    metric_keys = ["accuracy", "macro_f1", "f1_NM", "f1_KOA", "f1_PD", "roc_auc_macro"]
    metric_labels = ["Accuracy", "Macro F1", "F1 NM", "F1 KOA", "F1 PD", "ROC AUC"]

    models = list(results.keys())
    data   = {mk: [results[m].get(mk, 0) for m in models] for mk in metric_keys}

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(18, 4), sharey=False)
    bar_colors = plt.cm.tab10.colors

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        vals = data[mk]
        bars = ax.bar(models, vals, color=bar_colors[:len(models)], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )
        ax.set_ylim(0, 1.1)
        ax.set_title(ml, fontsize=9)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=7)
        ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "model_comparison.png")


def print_summary_table(results: dict[str, dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(results).T
    cols = ["accuracy", "macro_f1", "weighted_f1",
            "f1_NM", "f1_KOA", "f1_PD", "roc_auc_macro"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(df[cols].to_string(float_format="{:.4f}".format))
    print("=" * 70)