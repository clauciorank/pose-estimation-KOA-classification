"""
Focused classification pipeline — two tasks:

  Task A: NM vs KOA (binary)
      Distinguishes healthy controls from knee osteoarthritis patients.

  Task B: KOA severity staging — EL / MD / SV
      Classifies disease severity within KOA patients.
      Note: different patients share subject IDs across stages (001_EL ≠ 001_MD),
      so subject-level split is mandatory to prevent identity leakage.

Both tasks use 5-fold stratified subject-level cross-validation to produce
reliable mean ± std estimates given the small dataset size.

Usage:
    venv/bin/python src/models/run_focused.py [--epochs 200] [--seed 42] [--folds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.dataset import (
    load_tabular_dataset, load_sequence_dataset,
    compute_class_weights,
    NM_KOA_LABEL, STAGE_LABEL, LABEL_STAGE,
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
)

MODELS_OUT  = Path("data/output/models")
FIGURES_OUT = Path("data/output/figures/models")


def banner(text: str) -> None:
    print(f"\n{'═'*60}\n  {text}\n{'═'*60}")


def section(text: str) -> None:
    print(f"\n{'─'*55}\n  {text}\n{'─'*55}")


# ──────────────────────────────────────────────────────────────────────────────
# CV helpers
# ──────────────────────────────────────────────────────────────────────────────

def _summarize_folds(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold metrics into mean ± std."""
    summary = {}
    for key in fold_metrics[0]:
        vals = [f[key] for f in fold_metrics if not np.isnan(f.get(key, np.nan))]
        if vals:
            summary[key]            = round(float(np.mean(vals)), 4)
            summary[f"{key}_std"]   = round(float(np.std(vals)),  4)
        else:
            summary[key]            = float("nan")
            summary[f"{key}_std"]   = float("nan")
    return summary


def _agg_confusion_matrix(cms: list[np.ndarray]) -> np.ndarray:
    return np.sum(cms, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Tabular CV
# ──────────────────────────────────────────────────────────────────────────────

def run_tabular_cv(
    tab, label_names: list[str], tag: str, seed: int, n_splits: int = 5
) -> dict[str, dict]:
    """5-fold stratified subject-level CV for XGBoost, RF, SVM."""
    n_classes = len(label_names)
    model_names = ["XGBoost", "Random Forest", "SVM"]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = {n: [] for n in model_names}
    fold_cms     = {n: [] for n in model_names}
    fold_imp     = {n: [] for n in ["XGBoost", "Random Forest"]}

    print(f"  {n_splits}-fold CV | {tab.X.shape[0]} files | {len(np.unique(tab.groups))} subjects")

    for fold, (tr, te) in enumerate(
        sgkf.split(tab.X, tab.y, groups=tab.groups), start=1
    ):
        X_tr, X_te = tab.X.values[tr], tab.X.values[te]
        y_tr, y_te = tab.y[tr],        tab.y[te]
        n_tr_subj  = len(np.unique(tab.groups[tr]))
        n_te_subj  = len(np.unique(tab.groups[te]))

        configs = {
            "XGBoost":       build_xgboost(n_classes=n_classes, random_state=seed),
            "Random Forest": build_random_forest(random_state=seed),
            "SVM":           build_svm(),
        }
        fold_line = []
        for name, model in configs.items():
            y_pred, y_proba = fit_predict(model, X_tr, y_tr, X_te)
            m = compute_metrics(y_te, y_pred, y_proba, label_names)
            fold_metrics[name].append(m)

            cm = confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))
            fold_cms[name].append(cm)

            if name in fold_imp:
                imp = get_feature_importance(model, tab.feature_names)
                if not imp.empty:
                    fold_imp[name].append(imp.set_index("feature")["importance"])

            fold_line.append(f"{name[:3]}={m['accuracy']:.3f}")

        print(f"    fold {fold}/{n_splits}  "
              f"(train {n_tr_subj} subj / test {n_te_subj} subj)  " +
              "  ".join(fold_line))

    # Summarize
    results = {}
    for name in model_names:
        results[name] = _summarize_folds(fold_metrics[name])
        _print_cv_line(name, results[name])

        # Aggregated CM across folds
        agg_cm = _agg_confusion_matrix(fold_cms[name])
        _plot_agg_cm(agg_cm, f"{tag}_{name}_cv", label_names)

    # Feature importance: mean across folds
    for name in ["XGBoost", "Random Forest"]:
        if fold_imp[name]:
            imp_mean = (
                import_pandas().concat(fold_imp[name], axis=1)
                .fillna(0).mean(axis=1)
                .sort_values(ascending=False)
                .reset_index()
            )
            imp_mean.columns = ["feature", "importance"]
            plot_feature_importance(imp_mean, f"{tag}_{name}_cv")
            print(f"  [{name}] top-5 (mean): {imp_mean.head(5)['feature'].tolist()}")

    return results


def import_pandas():
    import pandas as pd
    return pd


# ──────────────────────────────────────────────────────────────────────────────
# LSTM CV
# ──────────────────────────────────────────────────────────────────────────────

def run_lstm_cv(
    seq, label_names: list[str], tag: str,
    epochs: int, seed: int, n_splits: int = 5,
) -> dict[str, dict]:
    """5-fold subject-level CV for LSTM and Bi-LSTM."""
    n_classes = len(label_names)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Derive one label per cycle for grouping
    y_seq   = seq.y
    grp_seq = seq.groups

    # Build a subject-level label array for stratification
    unique_subjs = np.unique(grp_seq)
    subj_label   = {s: int(np.bincount(y_seq[grp_seq == s]).argmax()) for s in unique_subjs}
    subj_arr     = np.array(unique_subjs)
    subj_lbl_arr = np.array([subj_label[s] for s in unique_subjs])

    fold_metrics = {"LSTM": [], "Bi-LSTM": []}
    fold_cms     = {"LSTM": [], "Bi-LSTM": []}

    print(f"  {n_splits}-fold CV | {seq.X.shape[0]} cycles | {len(unique_subjs)} subjects")

    cfg = TrainConfig(
        epochs=epochs, batch_size=32, lr=1e-3,
        patience=30, monitor="val_acc",
        verbose=False, log_every=epochs + 1,
    )

    for fold, (tr_subj_idx, te_subj_idx) in enumerate(
        sgkf.split(subj_arr, subj_lbl_arr, groups=subj_arr), start=1
    ):
        tr_subjs = set(subj_arr[tr_subj_idx])
        te_subjs = set(subj_arr[te_subj_idx])

        tr_mask = np.isin(grp_seq, list(tr_subjs))
        te_mask = np.isin(grp_seq, list(te_subjs))

        X_tr, X_te = seq.X[tr_mask], seq.X[te_mask]
        y_tr, y_te = y_seq[tr_mask], y_seq[te_mask]

        # Val split: 20% of train subjects
        rng = np.random.RandomState(seed + fold)
        tr_subjs_arr = np.array(list(tr_subjs))
        n_val = max(2, int(len(tr_subjs_arr) * 0.20))
        val_subjs = set(rng.choice(tr_subjs_arr, size=n_val, replace=False))
        val_mask2 = np.isin(seq.groups[tr_mask], list(val_subjs))
        tr_mask2  = ~val_mask2

        X_tr2, X_val = X_tr[tr_mask2], X_tr[val_mask2]
        y_tr2, y_val = y_tr[tr_mask2], y_tr[val_mask2]

        cw = compute_class_weights(y_tr2)

        torch.manual_seed(seed + fold)
        fold_line = []
        for name, builder in [("LSTM", build_lstm), ("Bi-LSTM", build_bilstm)]:
            model = builder(input_size=4, hidden_size=64, num_layers=2,
                            num_classes=n_classes, dropout=0.3)
            best_model, _ = train_model(
                model, X_tr2, y_tr2, X_val, y_val,
                class_weights=cw, cfg=cfg,
            )
            y_pred, y_proba = predict(best_model, X_te)
            m = compute_metrics(y_te, y_pred, y_proba, label_names)
            fold_metrics[name].append(m)

            cm = confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))
            fold_cms[name].append(cm)
            fold_line.append(f"{name}={m['accuracy']:.3f}")

        print(f"    fold {fold}/{n_splits}  "
              f"(train {len(tr_subjs)} / test {len(te_subjs)} subj)  " +
              "  ".join(fold_line))

    results = {}
    for name in ["LSTM", "Bi-LSTM"]:
        results[name] = _summarize_folds(fold_metrics[name])
        _print_cv_line(name, results[name])
        agg_cm = _agg_confusion_matrix(fold_cms[name])
        _plot_agg_cm(agg_cm, f"{tag}_{name}_cv", label_names)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Printing and plotting
# ──────────────────────────────────────────────────────────────────────────────

def _print_cv_line(name: str, m: dict) -> None:
    acc  = m.get("accuracy", float("nan"))
    f1   = m.get("macro_f1", float("nan"))
    auc  = m.get("roc_auc_macro", float("nan"))
    accs = m.get("accuracy_std", float("nan"))
    f1s  = m.get("macro_f1_std", float("nan"))
    print(f"  [{name}] acc={acc:.4f}±{accs:.4f}  "
          f"macro_f1={f1:.4f}±{f1s:.4f}  roc_auc={auc:.4f}")


def _plot_agg_cm(cm: np.ndarray, name: str, label_names: list[str]) -> None:
    """Plot aggregated (summed) confusion matrix across CV folds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in zip(
        axes,
        [cm,        cm_norm],
        ["d",       ".2f"],
        ["Contagens (todos os folds)", "Proporção (por classe real)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=ax, cbar=True,
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(title)

    fig.suptitle(f"Confusion Matrix (CV agregada) — {name}", fontsize=12)
    plt.tight_layout()

    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_OUT / f"cm_{name.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_cv_summary(results: dict[str, dict], title: str = "CV SUMMARY") -> None:
    import pandas as pd
    rows = {}
    for model, m in results.items():
        rows[model] = {
            "accuracy":    f"{m.get('accuracy', float('nan')):.4f} ± {m.get('accuracy_std', float('nan')):.4f}",
            "macro_f1":    f"{m.get('macro_f1',   float('nan')):.4f} ± {m.get('macro_f1_std',   float('nan')):.4f}",
            "roc_auc":     f"{m.get('roc_auc_macro', float('nan')):.4f} ± {m.get('roc_auc_macro_std', float('nan')):.4f}",
        }
    df = pd.DataFrame(rows).T
    print(f"\n{'='*65}")
    print(f"  {title}")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65)


def plot_cv_comparison(results: dict[str, dict], tag: str) -> None:
    """Bar chart with error bars (mean ± std across CV folds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "macro_f1", "roc_auc_macro"]
    metric_labels = ["Accuracy", "Macro F1", "ROC AUC"]
    models = list(results.keys())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), sharey=False)
    x = np.arange(len(models))

    for ax, mk, ml in zip(axes, metrics, metric_labels):
        means = [results[m].get(mk, 0)          for m in models]
        stds  = [results[m].get(f"{mk}_std", 0) for m in models]
        bars = ax.bar(x, means, color=[colors[i % 10] for i in range(len(models))],
                      edgecolor="white", alpha=0.85)
        ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                    capsize=4, lw=1.5)
        for bar, val, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )
        ax.set_ylim(0, 1.15)
        ax.set_title(ml, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(f"Model Comparison — 5-fold CV ({tag})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_OUT / f"cv_comparison_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CV comparison figure: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Task A — NM vs KOA
# ──────────────────────────────────────────────────────────────────────────────

def task_nm_koa(epochs: int, seed: int, n_splits: int) -> dict[str, dict]:
    banner("TASK A — NM vs KOA (binary)")
    LABELS = ["NM", "KOA"]
    TAG    = "nmkoa"

    section("Tabular models")
    tab = load_tabular_dataset(filter_groups=["NM", "KOA"], label_mode="nm_koa")
    print(f"  {tab.X.shape[0]} files | {tab.X.shape[1]} features | "
          f"{len(np.unique(tab.groups))} subjects")
    tab_results = run_tabular_cv(tab, LABELS, TAG, seed, n_splits)

    section("Sequence models (LSTM / Bi-LSTM)")
    seq = load_sequence_dataset(filter_groups=["NM", "KOA"], label_mode="nm_koa")
    print(f"  {seq.X.shape[0]} cycles | shape={seq.X.shape}")
    seq_results = run_lstm_cv(seq, LABELS, TAG, epochs, seed, n_splits)

    all_results = {**tab_results, **seq_results}
    print_cv_summary(all_results, f"TASK A — NM vs KOA ({n_splits}-fold CV)")
    plot_cv_comparison(all_results, TAG)

    _save_results(all_results, f"results_{TAG}_cv.json")
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Task B — KOA staging (EL / MD / SV)
# ──────────────────────────────────────────────────────────────────────────────

def task_koa_staging(epochs: int, seed: int, n_splits: int) -> dict[str, dict]:
    banner("TASK B — KOA Severity Staging (EL / MD / SV)")
    LABELS = ["EL", "MD", "SV"]
    TAG    = "koastage"

    section("Tabular models")
    tab = load_tabular_dataset(filter_groups=["KOA"], label_mode="stage")
    print(f"  {tab.X.shape[0]} files | {tab.X.shape[1]} features | "
          f"{len(np.unique(tab.groups))} subjects")
    print("  Note: different patients share IDs across stages — split is by subject_key")
    tab_results = run_tabular_cv(tab, LABELS, TAG, seed, n_splits)

    section("Sequence models (LSTM / Bi-LSTM)")
    seq = load_sequence_dataset(filter_groups=["KOA"], label_mode="stage")
    print(f"  {seq.X.shape[0]} cycles | shape={seq.X.shape}")
    seq_results = run_lstm_cv(seq, LABELS, TAG, epochs, seed, n_splits)

    all_results = {**tab_results, **seq_results}
    print_cv_summary(all_results, f"TASK B — KOA Staging ({n_splits}-fold CV)")
    plot_cv_comparison(all_results, TAG)

    _save_results(all_results, f"results_{TAG}_cv.json")
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Cross-task summary
# ──────────────────────────────────────────────────────────────────────────────

def plot_cross_task_summary(res_nmkoa: dict, res_staging: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "macro_f1", "roc_auc_macro"]
    metric_labels = ["Accuracy", "Macro F1", "ROC AUC"]
    tasks = {"NM vs KOA": res_nmkoa, "KOA Staging": res_staging}
    model_names = list(next(iter(tasks.values())).keys())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(model_names))
    width = 0.35

    for ax, mk, ml in zip(axes, metrics, metric_labels):
        for i, (task_name, res) in enumerate(tasks.items()):
            means  = [res[m].get(mk, 0)          for m in model_names]
            stds   = [res[m].get(f"{mk}_std", 0) for m in model_names]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, means, width,
                          label=task_name,
                          color=[colors[j % 10] for j in range(len(model_names))],
                          alpha=0.85 if i == 0 else 0.5,
                          edgecolor="white")
            ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                        color="black", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(ml, fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].legend(["NM vs KOA", "KOA Staging"], fontsize=8)
    axes[0].set_ylabel("Score (mean ± std)")
    fig.suptitle("Cross-task Model Comparison (5-fold CV)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_OUT / "cross_task_comparison_cv.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Cross-task figure: {path}")


def _save_results(results: dict, filename: str) -> None:
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    path = MODELS_OUT / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--folds",  type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    res_nmkoa   = task_nm_koa(args.epochs, args.seed, args.folds)
    res_staging = task_koa_staging(args.epochs, args.seed, args.folds)

    banner("CROSS-TASK SUMMARY")
    plot_cross_task_summary(res_nmkoa, res_staging)

    print("\nAll done.")
    print(f"Figures → {FIGURES_OUT}")
    print(f"Models  → {MODELS_OUT}")


if __name__ == "__main__":
    main()