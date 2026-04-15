"""
Ensemble pipeline: Bi-LSTM embeddings + tabular kinematic features.

For each CV fold:
  1. Train a Bi-LSTM on train cycles (same architecture as run_focused.py)
  2. Extract the pre-classifier embedding (last hidden state, 128-dim) for
     every cycle without touching the existing model code
  3. For each subject, average embeddings across all their cycles
     and average tabular features across their files
  4. Concatenate [128-dim LSTM | 51-dim tabular] → 179-dim feature vector
  5. Train XGBoost, RF, SVM on the combined vector
  6. Evaluate and compare against standalone results from run_focused.py

Produces:
  data/output/models/results_nmkoa_ensemble_cv.json
  data/output/models/results_koastage_ensemble_cv.json
  data/output/figures/models/ensemble_comparison_nmkoa.png
  data/output/figures/models/ensemble_comparison_koastage.png

Usage:
    venv/bin/python src/models/run_ensemble.py [--epochs 200] [--seed 42] [--folds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.dataset import (
    load_tabular_dataset, load_sequence_dataset, compute_class_weights,
)
from src.models.tabular import (
    build_xgboost, build_random_forest, build_svm, fit_predict,
)
from src.models.lstm_model import build_bilstm, build_lstm
from src.models.train_lstm import train_model, TrainConfig
from src.models.evaluate import compute_metrics
from src.models.run_focused import (
    _summarize_folds, _print_cv_line, _plot_agg_cm, print_cv_summary,
)

MODELS_OUT  = Path("data/output/models")
FIGURES_OUT = Path("data/output/figures/models")


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (no changes to GaitLSTM)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_bilstm_embeddings(
    model, X: np.ndarray, device: str | None = None, batch_size: int = 256
) -> np.ndarray:
    """Extract pre-classifier embedding from a trained GaitLSTM (LSTM or Bi-LSTM).

    Accesses model.lstm and model.dropout directly so GaitLSTM is unmodified.
    In eval mode dropout is a no-op — embeddings are deterministic.

    Returns: (N, emb_dim) float32 array  [64 for LSTM, 128 for Bi-LSTM]
    """
    if device is None:
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(
                X[start : start + batch_size], dtype=torch.float32
            ).to(dev)
            out, _   = model.lstm(xb)       # (B, 101, 128)
            last     = out[:, -1, :]        # (B, 128) last timestep
            last     = model.dropout(last)  # no-op in eval
            parts.append(last.cpu().numpy())

    return np.vstack(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Subject-level feature aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _build_subject_tab(tab) -> pd.DataFrame:
    """Average tabular features across files for each subject.

    Returns DataFrame indexed by subject_key with shape (n_subjects, n_features).
    """
    df = tab.X.copy()
    df["__key__"] = tab.groups
    return df.groupby("__key__")[tab.feature_names].mean()


def _cycles_to_subject_features(
    seq_groups: np.ndarray,
    lstm_embeddings: np.ndarray,
    subj_tab: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """For each cycle, build [LSTM embedding | tabular features].

    Args:
        seq_groups:       (N_cycles,) subject keys
        lstm_embeddings:  (N_cycles, emb_dim) LSTM embeddings
        subj_tab:         DataFrame indexed by subject_key

    Returns:
        X_combined: (N_cycles, emb_dim + n_tab_features)
        missing:    mask of cycles whose subject was not in subj_tab (should be empty)
    """
    tab_rows = []
    missing  = []
    for g in seq_groups:
        if g in subj_tab.index:
            tab_rows.append(subj_tab.loc[g].values)
            missing.append(False)
        else:
            tab_rows.append(np.zeros(subj_tab.shape[1]))
            missing.append(True)

    tab_arr = np.array(tab_rows, dtype=np.float32)
    return np.hstack([lstm_embeddings, tab_arr]), np.array(missing)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble CV
# ──────────────────────────────────────────────────────────────────────────────

def run_ensemble_cv(
    tab,
    seq,
    label_names: list[str],
    tag: str,
    epochs: int,
    seed: int,
    n_splits: int = 5,
    encoder: str = "bilstm",
) -> dict[str, dict]:
    """5-fold subject-level CV: LSTM/Bi-LSTM encoder + tabular features → XGB/RF/SVM."""

    n_classes = len(label_names)
    subj_tab  = _build_subject_tab(tab)

    # Subject-level stratification arrays for CV
    unique_subjs  = np.unique(seq.groups)
    subj_lbl      = {
        s: int(np.bincount(seq.y[seq.groups == s]).argmax())
        for s in unique_subjs
    }
    subj_arr      = np.array(unique_subjs)
    subj_lbl_arr  = np.array([subj_lbl[s] for s in unique_subjs])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cfg  = TrainConfig(
        epochs=epochs, batch_size=32, lr=1e-3,
        patience=30, monitor="val_acc",
        verbose=False, log_every=epochs + 1,
    )

    clf_names    = ["Ensemble XGB", "Ensemble RF", "Ensemble SVM"]
    fold_metrics = {n: [] for n in clf_names}
    fold_cms     = {n: [] for n in clf_names}

    # Embedding dim: LSTM = 64, Bi-LSTM = 64*2 = 128
    emb_dim = 64 if encoder == "lstm" else 128
    enc_label = "LSTM" if encoder == "lstm" else "Bi-LSTM"
    print(f"  {n_splits}-fold CV | {seq.X.shape[0]} cycles | "
          f"{len(unique_subjs)} subjects | encoder={enc_label} | "
          f"combined dim = {emb_dim} (LSTM emb) + {len(tab.feature_names)} (tabular)")

    from sklearn.metrics import confusion_matrix

    for fold, (tr_si, te_si) in enumerate(
        sgkf.split(subj_arr, subj_lbl_arr, groups=subj_arr), start=1
    ):
        tr_subjs = set(subj_arr[tr_si])
        te_subjs = set(subj_arr[te_si])

        tr_mask = np.isin(seq.groups, list(tr_subjs))
        te_mask = np.isin(seq.groups, list(te_subjs))

        X_seq_tr, X_seq_te = seq.X[tr_mask], seq.X[te_mask]
        y_tr,     y_te     = seq.y[tr_mask], seq.y[te_mask]

        # ── Val split for LSTM early stopping ────────────────────────────────
        rng        = np.random.RandomState(seed + fold)
        tr_s_arr   = np.array(list(tr_subjs))
        n_val      = max(2, int(len(tr_s_arr) * 0.20))
        val_subjs  = set(rng.choice(tr_s_arr, size=n_val, replace=False))
        val_m2     = np.isin(seq.groups[tr_mask], list(val_subjs))
        tr_m2      = ~val_m2

        X_tr2, X_val = X_seq_tr[tr_m2], X_seq_tr[val_m2]
        y_tr2, y_val = y_tr[tr_m2],     y_tr[val_m2]

        cw = compute_class_weights(y_tr2)

        # ── Train encoder ─────────────────────────────────────────────────────
        torch.manual_seed(seed + fold)
        if encoder == "lstm":
            model = build_lstm(
                input_size=4, hidden_size=64, num_layers=2,
                num_classes=n_classes, dropout=0.3,
            )
        else:
            model = build_bilstm(
                input_size=4, hidden_size=64, num_layers=2,
                num_classes=n_classes, dropout=0.3,
            )
        best_model, _ = train_model(
            model, X_tr2, y_tr2, X_val, y_val,
            class_weights=cw, cfg=cfg,
        )

        # ── Extract embeddings ────────────────────────────────────────────────
        emb_tr = _extract_bilstm_embeddings(best_model, X_seq_tr)
        emb_te = _extract_bilstm_embeddings(best_model, X_seq_te)

        # ── Build combined feature matrices ───────────────────────────────────
        X_tr_comb, miss_tr = _cycles_to_subject_features(
            seq.groups[tr_mask], emb_tr, subj_tab
        )
        X_te_comb, miss_te = _cycles_to_subject_features(
            seq.groups[te_mask], emb_te, subj_tab
        )

        if miss_tr.any() or miss_te.any():
            print(f"    ⚠ fold {fold}: {miss_tr.sum()} train / "
                  f"{miss_te.sum()} test cycles had no tabular entry — zeroed")

        # ── Train classifiers ─────────────────────────────────────────────────
        configs = {
            "Ensemble XGB": build_xgboost(n_classes=n_classes, random_state=seed),
            "Ensemble RF":  build_random_forest(random_state=seed),
            "Ensemble SVM": build_svm(),
        }
        fold_line = []
        for name, clf in configs.items():
            y_pred, y_proba = fit_predict(clf, X_tr_comb, y_tr, X_te_comb)
            m  = compute_metrics(y_te, y_pred, y_proba, label_names)
            fold_metrics[name].append(m)

            cm = confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))
            fold_cms[name].append(cm)

            fold_line.append(f"{name.split()[-1]}={m['accuracy']:.3f}")

        print(f"    fold {fold}/{n_splits}  "
              f"(train {len(tr_subjs)} / test {len(te_subjs)} subj)  " +
              "  ".join(fold_line))

    # ── Summarize ─────────────────────────────────────────────────────────────
    results = {}
    for name in clf_names:
        results[name] = _summarize_folds(fold_metrics[name])
        _print_cv_line(name, results[name])
        _plot_agg_cm(
            np.sum(fold_cms[name], axis=0),
            f"{tag}_{name.replace(' ', '_')}_cv",
            label_names,
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_ensemble_comparison(
    standalone: dict[str, dict],
    ensemble:   dict[str, dict],
    tag: str,
    title: str,
) -> None:
    """Side-by-side bar chart: standalone models vs ensemble models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics       = ["accuracy", "macro_f1", "roc_auc_macro"]
    metric_labels = ["Accuracy", "Macro F1", "ROC AUC"]

    # Separate standalone into tabular and sequence groups
    tabular_names  = ["XGBoost", "Random Forest", "SVM"]
    sequence_names = ["LSTM", "Bi-LSTM"]
    ensemble_names = ["Ensemble XGB", "Ensemble RF", "Ensemble SVM"]

    groups = [
        ("Tabular only",  tabular_names,  "steelblue"),
        ("Sequence only", sequence_names, "darkorange"),
        ("Ensemble",      ensemble_names, "seagreen"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, mk, ml in zip(axes, metrics, metric_labels):
        x_pos  = 0
        xticks = []
        xlabels = []

        for group_label, names, color in groups:
            all_results = {**standalone, **ensemble}
            for name in names:
                if name not in all_results:
                    continue
                m   = all_results[name]
                val = m.get(mk, 0)
                std = m.get(f"{mk}_std", 0)

                bar = ax.bar(x_pos, val, color=color, alpha=0.8, edgecolor="white")
                ax.errorbar(x_pos, val, yerr=std, fmt="none",
                            color="black", capsize=3, lw=1.5)
                ax.text(x_pos, val + std + 0.01, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=6)

                xticks.append(x_pos)
                xlabels.append(name.replace("Ensemble ", ""))
                x_pos += 1

            x_pos += 0.5  # gap between groups

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.set_title(ml, fontsize=10)
        ax.grid(True, alpha=0.25, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="steelblue",  alpha=0.8, label="Tabular only"),
        Patch(color="darkorange", alpha=0.8, label="Sequence only (LSTM)"),
        Patch(color="seagreen",   alpha=0.8, label="Ensemble (LSTM + tabular)"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="lower right")
    axes[0].set_ylabel("Score (mean ± std CV)")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_OUT / f"ensemble_comparison_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison figure: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Task runners
# ──────────────────────────────────────────────────────────────────────────────

def _load_standalone(filename: str) -> dict:
    path = MODELS_OUT / filename
    if not path.exists():
        print(f"  ⚠ Standalone results not found at {path} — run run_focused.py first")
        return {}
    with open(path) as f:
        return json.load(f)


def banner(text: str) -> None:
    print(f"\n{'═'*60}\n  {text}\n{'═'*60}")


def section(text: str) -> None:
    print(f"\n{'─'*55}\n  {text}\n{'─'*55}")


def task_nm_koa(epochs: int, seed: int, n_splits: int, encoder: str = "bilstm") -> None:
    enc_label = "LSTM" if encoder == "lstm" else "Bi-LSTM"
    banner(f"TASK A — NM vs KOA  |  Ensemble CV  |  encoder={enc_label}")
    label_names = ["NM", "KOA"]
    tag         = "nmkoa"

    tab = load_tabular_dataset(filter_groups=["NM", "KOA"], label_mode="nm_koa")
    seq = load_sequence_dataset(filter_groups=["NM", "KOA"], label_mode="nm_koa")

    section(f"Ensemble models ({enc_label} + tabular)")
    ens_results = run_ensemble_cv(tab, seq, label_names, tag, epochs, seed, n_splits, encoder=encoder)

    print_cv_summary(ens_results, f"TASK A Ensemble ({n_splits}-fold CV)")

    standalone = _load_standalone("results_nmkoa_cv.json")

    if standalone:
        section("Full comparison")
        all_r = {**standalone, **ens_results}
        print_cv_summary(all_r, "TASK A — All models")
        plot_ensemble_comparison(standalone, ens_results, tag,
                                 f"NM vs KOA — Standalone vs Ensemble ({enc_label}, 5-fold CV)")

    enc_suffix = f"_{encoder}" if encoder != "bilstm" else ""
    out_json = MODELS_OUT / f"results_{tag}_ensemble{enc_suffix}_cv.json"
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(ens_results, f, indent=2)
    print(f"  Results → {out_json}")


def task_koa_staging(epochs: int, seed: int, n_splits: int, encoder: str = "bilstm") -> None:
    enc_label = "LSTM" if encoder == "lstm" else "Bi-LSTM"
    banner(f"TASK B — KOA Staging  |  Ensemble CV  |  encoder={enc_label}")
    label_names = ["EL", "MD", "SV"]
    tag         = "koastage"

    tab = load_tabular_dataset(filter_groups=["KOA"], label_mode="stage")
    seq = load_sequence_dataset(filter_groups=["KOA"], label_mode="stage")

    section(f"Ensemble models ({enc_label} + tabular)")
    ens_results = run_ensemble_cv(tab, seq, label_names, tag, epochs, seed, n_splits, encoder=encoder)

    print_cv_summary(ens_results, f"TASK B Ensemble ({n_splits}-fold CV)")

    standalone = _load_standalone("results_koastage_cv.json")

    if standalone:
        section("Full comparison")
        all_r = {**standalone, **ens_results}
        print_cv_summary(all_r, "TASK B — All models")
        plot_ensemble_comparison(standalone, ens_results, tag,
                                 f"KOA Staging — Standalone vs Ensemble ({enc_label}, 5-fold CV)")

    enc_suffix = f"_{encoder}" if encoder != "bilstm" else ""
    out_json = MODELS_OUT / f"results_{tag}_ensemble{enc_suffix}_cv.json"
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(ens_results, f, indent=2)
    print(f"  Results → {out_json}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--folds",   type=int, default=5)
    parser.add_argument("--encoder", choices=["lstm", "bilstm"], default="bilstm",
                        help="Encoder type for the ensemble (default: bilstm)")
    parser.add_argument("--task",    choices=["a", "b", "both"], default="both",
                        help="Which task to run: a=NM/KOA, b=KOA staging, both=all (default: both)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.task in ("a", "both"):
        task_nm_koa(args.epochs, args.seed, args.folds, encoder=args.encoder)
    if args.task in ("b", "both"):
        task_koa_staging(args.epochs, args.seed, args.folds, encoder=args.encoder)

    print("\nAll done.")
    print(f"Figures → {FIGURES_OUT}")
    print(f"Models  → {MODELS_OUT}")


if __name__ == "__main__":
    main()