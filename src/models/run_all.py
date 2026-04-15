"""
Full training and evaluation pipeline.

Trains and evaluates 5 models on the same subject-level train/test split:
  1. XGBoost        (tabular features)
  2. Random Forest  (tabular features)
  3. SVM            (tabular features)
  4. LSTM           (normalised cycle sequences)
  5. Bi-LSTM        (normalised cycle sequences)

All models share the same test subjects — no leakage.

Usage:
    venv/bin/python src/models/run_all.py [--epochs 150] [--seed 42]

Output:
    data/output/figures/models/   ← all figures
    data/output/models/           ← saved model checkpoints + results JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.dataset import (
    load_tabular_dataset, load_sequence_dataset,
    subject_split, compute_class_weights,
    GROUP_LABEL, LABEL_GROUP,
)
from src.models.tabular import (
    build_xgboost, build_random_forest, build_svm,
    fit_predict, get_feature_importance,
)
from src.models.lstm_model import build_lstm, build_bilstm
from src.models.train_lstm import train_model, predict, TrainConfig
from src.models.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_learning_curves, plot_feature_importance,
    plot_comparison_table, print_summary_table,
)

MODELS_OUT = Path("data/output/models")


def banner(text: str) -> None:
    print(f"\n{'─'*60}\n  {text}\n{'─'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Tabular pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_tabular(seed: int = 42) -> dict[str, dict]:
    banner("TABULAR MODELS (XGBoost, Random Forest, SVM)")

    tab = load_tabular_dataset()
    print(f"  Dataset: {tab.X.shape[0]} files, {tab.X.shape[1]} features")
    print(f"  Classes: {dict(zip(*np.unique(tab.y, return_counts=True)))}")

    tr_idx, te_idx = subject_split(tab.y, tab.groups, test_size=0.25,
                                   random_state=seed)
    X_tr, X_te = tab.X.values[tr_idx], tab.X.values[te_idx]
    y_tr, y_te = tab.y[tr_idx],        tab.y[te_idx]

    print(f"  Train subjects: {len(np.unique(tab.groups[tr_idx]))} | "
          f"Test subjects: {len(np.unique(tab.groups[te_idx]))}")
    print(f"  Train: {dict(zip(*np.unique(y_tr, return_counts=True)))} | "
          f"Test: {dict(zip(*np.unique(y_te, return_counts=True)))}")

    configs = {
        "XGBoost":       build_xgboost(random_state=seed),
        "Random Forest": build_random_forest(random_state=seed),
        "SVM":           build_svm(),
    }

    results = {}
    for name, model in configs.items():
        t0 = time.time()
        print(f"\n  [{name}] training...")
        y_pred, y_proba = fit_predict(model, X_tr, y_tr, X_te)
        elapsed = time.time() - t0

        metrics = compute_metrics(y_te, y_pred, y_proba)
        metrics["train_time_s"] = round(elapsed, 2)
        results[name] = metrics

        print(f"  [{name}] acc={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"roc_auc={metrics.get('roc_auc_macro', float('nan')):.4f}  "
              f"({elapsed:.1f}s)")

        plot_confusion_matrix(y_te, y_pred, name)
        plot_roc_curves(y_te, y_proba, name)

        imp = get_feature_importance(model, tab.feature_names)
        if not imp.empty:
            plot_feature_importance(imp, name)
            top5 = imp.head(5)["feature"].tolist()
            print(f"  [{name}] top-5 features: {top5}")

    # Save train/test indices for reproducibility
    return results, tr_idx, te_idx, tab


# ──────────────────────────────────────────────────────────────────────────────
# Sequence (LSTM) pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_sequence(tr_idx_tab, te_idx_tab, tab,
                 epochs: int = 150, seed: int = 42) -> dict[str, dict]:
    """Train LSTM and Bi-LSTM using the same test subjects as the tabular split."""
    banner("SEQUENCE MODELS (LSTM, Bi-LSTM)")

    seq = load_sequence_dataset()
    print(f"  Dataset: {seq.X.shape[0]} cycles  shape={seq.X.shape}")
    print(f"  Classes: {dict(zip(*np.unique(seq.y, return_counts=True)))}")

    # Reuse the same test subjects identified by the tabular split
    test_subjects  = set(tab.groups[te_idx_tab])
    train_subjects = set(tab.groups[tr_idx_tab])

    tr_mask = np.isin(seq.groups, list(train_subjects))
    te_mask = np.isin(seq.groups, list(test_subjects))

    # Subjects that appear in seq but not in tab (can happen if all files
    # for a subject were skipped) → assign to train
    neither = ~(tr_mask | te_mask)
    if neither.any():
        tr_mask |= neither

    X_tr, X_te = seq.X[tr_mask], seq.X[te_mask]
    y_tr, y_te = seq.y[tr_mask], seq.y[te_mask]

    print(f"  Train cycles: {len(X_tr)}  "
          f"{dict(zip(*np.unique(y_tr, return_counts=True)))}")
    print(f"  Test  cycles: {len(X_te)}  "
          f"{dict(zip(*np.unique(y_te, return_counts=True)))}")

    cw = compute_class_weights(y_tr)
    print(f"  Class weights: NM={cw[0]:.2f}  KOA={cw[1]:.2f}  PD={cw[2]:.2f}")

    # Val split from train (10%) for early stopping — still subject-level
    rng         = np.random.RandomState(seed)
    train_subjs = np.unique(seq.groups[tr_mask])
    val_subjs   = set(rng.choice(train_subjs,
                                  size=max(2, int(len(train_subjs) * 0.20)),
                                  replace=False))

    val_mask2  = np.isin(seq.groups[tr_mask], list(val_subjs))
    tr_mask2   = ~val_mask2

    X_tr2, X_val = X_tr[tr_mask2], X_tr[val_mask2]
    y_tr2, y_val = y_tr[tr_mask2], y_tr[val_mask2]
    print(f"  → inner train: {len(X_tr2)} | val: {len(X_val)}")

    cfg = TrainConfig(epochs=epochs, batch_size=64, lr=1e-3,
                      patience=30, monitor="val_acc",
                      verbose=True, log_every=10)

    torch.manual_seed(seed)
    results = {}

    model_builders = {
        "LSTM":    build_lstm,
        "Bi-LSTM": build_bilstm,
    }

    for name, builder in model_builders.items():
        print(f"\n  [{name}] training...")
        t0    = time.time()
        model = builder(input_size=4, hidden_size=64, num_layers=2,
                        num_classes=3, dropout=0.3)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [{name}] parameters: {n_params:,}")

        best_model, history = train_model(
            model, X_tr2, y_tr2, X_val, y_val,
            class_weights=cw, cfg=cfg,
        )
        elapsed = time.time() - t0

        y_pred, y_proba = predict(best_model, X_te)
        metrics = compute_metrics(y_te, y_pred, y_proba)
        metrics["train_time_s"] = round(elapsed, 2)
        metrics["best_epoch"]   = history.best_epoch
        results[name] = metrics

        print(f"  [{name}] acc={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  "
              f"roc_auc={metrics.get('roc_auc_macro', float('nan')):.4f}  "
              f"({elapsed:.1f}s)")

        plot_confusion_matrix(y_te, y_pred, name)
        plot_roc_curves(y_te, y_proba, name)
        plot_learning_curves(history, name)

        # Save checkpoint
        MODELS_OUT.mkdir(parents=True, exist_ok=True)
        torch.save(best_model.state_dict(),
                   MODELS_OUT / f"{name.replace('-', '_').lower()}_best.pt")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Tabular ──
    tab_results, tr_idx, te_idx, tab = run_tabular(seed=args.seed)

    # ── Sequence ──
    seq_results = run_sequence(tr_idx, te_idx, tab,
                                epochs=args.epochs, seed=args.seed)

    # ── Comparison ──
    all_results = {**tab_results, **seq_results}
    banner("FINAL COMPARISON")
    print_summary_table(all_results)
    plot_comparison_table(all_results)

    # Save JSON
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    out_json = MODELS_OUT / "results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_json}")
    print(f"  Figures in data/output/figures/models/")


if __name__ == "__main__":
    main()