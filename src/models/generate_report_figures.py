"""
Generate additional figures for the final report:
  1. OOF (out-of-fold) ROC curves for all models — both tasks
  2. Ensemble XGB feature importance (tabular vs LSTM modalities)
  3. Per-fold accuracy strip plots (variance visualisation)

Usage:
    venv/bin/python src/models/generate_report_figures.py [--epochs 200] [--seed 42]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

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
from src.models.run_focused import _summarize_folds
from src.models.run_ensemble import _extract_bilstm_embeddings, _build_subject_tab, _cycles_to_subject_features

FIGURES_OUT = Path("data/output/figures/models")
MODELS_OUT  = Path("data/output/models")

COLORS_GROUP = {
    "NM":  "#2ecc71",
    "KOA": "#e74c3c",
    "EL":  "#f39c12",
    "MD":  "#e74c3c",
    "SV":  "#8e44ad",
}


# ──────────────────────────────────────────────────────────────────────────────
# OOF prediction collector
# ──────────────────────────────────────────────────────────────────────────────

def collect_oof_tabular(tab, label_names, n_splits=5, seed=42):
    """Return dict of model → (y_true_oof, y_proba_oof, fold_ids)."""
    n_classes = len(label_names)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = {
        "XGBoost":       {"y_true": [], "y_proba": [], "fold": []},
        "Random Forest": {"y_true": [], "y_proba": [], "fold": []},
        "SVM":           {"y_true": [], "y_proba": [], "fold": []},
    }

    for fold, (tr, te) in enumerate(sgkf.split(tab.X, tab.y, groups=tab.groups), 1):
        configs = {
            "XGBoost":       build_xgboost(n_classes=n_classes, random_state=seed),
            "Random Forest": build_random_forest(random_state=seed),
            "SVM":           build_svm(),
        }
        for name, model in configs.items():
            _, y_proba = fit_predict(model, tab.X.values[tr], tab.y[tr],
                                      tab.X.values[te])
            oof[name]["y_true"].extend(tab.y[te].tolist())
            oof[name]["y_proba"].extend(y_proba.tolist())
            oof[name]["fold"].extend([fold] * len(te))

    return {
        name: {
            "y_true":  np.array(d["y_true"]),
            "y_proba": np.array(d["y_proba"]),
            "fold":    np.array(d["fold"]),
        }
        for name, d in oof.items()
    }


def collect_oof_lstm(seq, label_names, n_splits=5, seed=42, epochs=200):
    """OOF predictions for LSTM and Bi-LSTM."""
    n_classes    = len(label_names)
    unique_subjs = np.unique(seq.groups)
    subj_lbl     = {s: int(np.bincount(seq.y[seq.groups == s]).argmax()) for s in unique_subjs}
    subj_arr     = np.array(unique_subjs)
    subj_lbl_arr = np.array([subj_lbl[s] for s in unique_subjs])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cfg  = TrainConfig(epochs=epochs, batch_size=32, lr=1e-3, patience=30,
                       monitor="val_acc", verbose=False, log_every=epochs+1)

    oof = {
        "LSTM":    {"y_true": [], "y_proba": [], "fold": []},
        "Bi-LSTM": {"y_true": [], "y_proba": [], "fold": []},
    }

    for fold, (tr_si, te_si) in enumerate(
        sgkf.split(subj_arr, subj_lbl_arr, groups=subj_arr), 1
    ):
        tr_subjs = set(subj_arr[tr_si])
        te_subjs = set(subj_arr[te_si])
        tr_mask  = np.isin(seq.groups, list(tr_subjs))
        te_mask  = np.isin(seq.groups, list(te_subjs))

        X_tr, X_te = seq.X[tr_mask], seq.X[te_mask]
        y_tr, y_te = seq.y[tr_mask], seq.y[te_mask]

        rng       = np.random.RandomState(seed + fold)
        tr_s_arr  = np.array(list(tr_subjs))
        n_val     = max(2, int(len(tr_s_arr) * 0.20))
        val_subjs = set(rng.choice(tr_s_arr, size=n_val, replace=False))
        val_m2    = np.isin(seq.groups[tr_mask], list(val_subjs))
        tr_m2     = ~val_m2

        cw = compute_class_weights(y_tr[tr_m2])

        torch.manual_seed(seed + fold)
        for name, builder in [("LSTM", build_lstm), ("Bi-LSTM", build_bilstm)]:
            model = builder(input_size=4, hidden_size=64, num_layers=2,
                            num_classes=n_classes, dropout=0.3)
            best, _ = train_model(model, X_tr[tr_m2], y_tr[tr_m2],
                                   X_tr[val_m2], y_tr[val_m2],
                                   class_weights=cw, cfg=cfg)
            best.eval()
            dev = next(best.parameters()).device   # respects GPU if training used cuda
            with torch.no_grad():
                logits = best(torch.tensor(X_te, dtype=torch.float32).to(dev))
                proba  = torch.softmax(logits, dim=1).cpu().numpy()
            oof[name]["y_true"].extend(y_te.tolist())
            oof[name]["y_proba"].extend(proba.tolist())
            oof[name]["fold"].extend([fold] * len(y_te))
        print(f"  fold {fold}/{n_splits} done")

    return {
        name: {
            "y_true":  np.array(d["y_true"]),
            "y_proba": np.array(d["y_proba"]),
        }
        for name, d in oof.items()
    }


def collect_oof_ensemble(tab, seq, label_names, n_splits=5, seed=42, epochs=200):
    """OOF predictions for Ensemble XGB (Bi-LSTM + tabular)."""
    n_classes = len(label_names)
    subj_tab  = _build_subject_tab(tab)

    unique_subjs = np.unique(seq.groups)
    subj_lbl     = {s: int(np.bincount(seq.y[seq.groups == s]).argmax()) for s in unique_subjs}
    subj_arr     = np.array(unique_subjs)
    subj_lbl_arr = np.array([subj_lbl[s] for s in unique_subjs])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cfg  = TrainConfig(epochs=epochs, batch_size=32, lr=1e-3, patience=30,
                       monitor="val_acc", verbose=False, log_every=epochs+1)

    oof = {"y_true": [], "y_proba": [], "fold": []}

    for fold, (tr_si, te_si) in enumerate(
        sgkf.split(subj_arr, subj_lbl_arr, groups=subj_arr), 1
    ):
        tr_subjs = set(subj_arr[tr_si])
        te_subjs = set(subj_arr[te_si])
        tr_mask  = np.isin(seq.groups, list(tr_subjs))
        te_mask  = np.isin(seq.groups, list(te_subjs))

        X_seq_tr, X_seq_te = seq.X[tr_mask], seq.X[te_mask]
        y_tr, y_te         = seq.y[tr_mask], seq.y[te_mask]

        rng       = np.random.RandomState(seed + fold)
        tr_s_arr  = np.array(list(tr_subjs))
        n_val     = max(2, int(len(tr_s_arr) * 0.20))
        val_subjs = set(rng.choice(tr_s_arr, size=n_val, replace=False))
        val_m2    = np.isin(seq.groups[tr_mask], list(val_subjs))
        tr_m2     = ~val_m2

        cw = compute_class_weights(y_tr[tr_m2])

        torch.manual_seed(seed + fold)
        model = build_bilstm(input_size=4, hidden_size=64, num_layers=2,
                              num_classes=n_classes, dropout=0.3)
        best, _ = train_model(model, X_seq_tr[tr_m2], y_tr[tr_m2],
                               X_seq_tr[val_m2], y_tr[val_m2],
                               class_weights=cw, cfg=cfg)

        emb_tr = _extract_bilstm_embeddings(best, X_seq_tr)
        emb_te = _extract_bilstm_embeddings(best, X_seq_te)
        X_tr_c, _ = _cycles_to_subject_features(seq.groups[tr_mask], emb_tr, subj_tab)
        X_te_c, _ = _cycles_to_subject_features(seq.groups[te_mask], emb_te, subj_tab)

        xgb = build_xgboost(n_classes=n_classes, random_state=seed)
        _, y_proba = fit_predict(xgb, X_tr_c, y_tr, X_te_c)
        oof["y_true"].extend(y_te.tolist())
        oof["y_proba"].extend(y_proba.tolist())
        oof["fold"].extend([fold] * len(y_te))
        print(f"  fold {fold}/{n_splits} done")

    return {
        "y_true":  np.array(oof["y_true"]),
        "y_proba": np.array(oof["y_proba"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ROC plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_oof_roc(oof_dict: dict, label_names: list[str],
                 title: str, filename: str) -> None:
    """One figure per model group with OOF ROC curves (one-vs-rest)."""
    n_classes = len(label_names)
    colors    = [COLORS_GROUP.get(l, "#555") for l in label_names]
    ls_cycle  = ["-", "--", "-.", ":"]

    model_names = list(oof_dict.keys())
    n_models    = len(model_names)
    fig, axes   = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        d      = oof_dict[name]
        y_true = d["y_true"]
        y_prob = d["y_proba"]

        for i, label in enumerate(label_names):
            y_bin = (y_true == i).astype(int)
            try:
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
                auc = roc_auc_score(y_bin, y_prob[:, i])
                ax.plot(fpr, tpr, lw=2, color=colors[i],
                        linestyle=ls_cycle[i % len(ls_cycle)],
                        label=f"{label}  AUC={auc:.3f}")
            except Exception:
                pass

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble feature importance
# ──────────────────────────────────────────────────────────────────────────────

def plot_ensemble_feature_importance(
    tab, seq, label_names: list[str], tag: str,
    epochs: int = 200, seed: int = 42, top_n: int = 25,
) -> None:
    """Train Ensemble XGB on the full dataset and plot feature importance.

    Features are named: 'tab__<feature_name>' for tabular
    and 'lstm__<dim_index>' for LSTM embedding dimensions.
    Adds a modality-level bar showing aggregate tabular vs LSTM importance.
    """
    n_classes = len(label_names)
    subj_tab  = _build_subject_tab(tab)

    cfg = TrainConfig(epochs=epochs, batch_size=32, lr=1e-3, patience=30,
                      monitor="val_acc", verbose=False, log_every=epochs+1)

    # Val split: 20% of subjects (for LSTM early stopping)
    unique_subjs = np.unique(seq.groups)
    rng          = np.random.RandomState(seed)
    n_val        = max(2, int(len(unique_subjs) * 0.20))
    val_subjs    = set(rng.choice(unique_subjs, size=n_val, replace=False))
    tr_mask      = ~np.isin(seq.groups, list(val_subjs))
    val_mask     = np.isin(seq.groups, list(val_subjs))

    cw = compute_class_weights(seq.y[tr_mask])

    torch.manual_seed(seed)
    model = build_bilstm(input_size=4, hidden_size=64, num_layers=2,
                          num_classes=n_classes, dropout=0.3)
    best, _ = train_model(model,
                           seq.X[tr_mask], seq.y[tr_mask],
                           seq.X[val_mask], seq.y[val_mask],
                           class_weights=cw, cfg=cfg)

    emb_all = _extract_bilstm_embeddings(best, seq.X)
    X_all, _ = _cycles_to_subject_features(seq.groups, emb_all, subj_tab)

    emb_dim   = emb_all.shape[1]       # 128
    tab_dim   = len(tab.feature_names) # 51
    feat_names = (
        [f"lstm__{i}"   for i in range(emb_dim)] +
        [f"tab__{n}"    for n in tab.feature_names]
    )

    # Feature importance on full dataset (visualization only — not used for CV metrics).
    # Pipeline.fit also fits the imputer on all data here, which is intentional for
    # this visualization-only context (no train/test split exists).
    xgb_pipe = build_xgboost(n_classes=n_classes, random_state=seed)
    xgb_pipe.fit(X_all, seq.y)
    imp = pd.DataFrame({"feature": feat_names,
                         "importance": xgb_pipe.named_steps["xgb"].feature_importances_})
    imp["modality"] = imp["feature"].apply(lambda x: "LSTM" if x.startswith("lstm__") else "Tabular")
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Figure 1: top-N individual features ──────────────────────────────────
    top = imp.head(top_n).copy()
    top["label"] = top["feature"].str.replace("tab__", "").str.replace("lstm__", "LSTM dim ")
    colors_bar = ["#3498db" if m == "LSTM" else "#e74c3c" for m in top["modality"]]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    ax.barh(top["label"][::-1], top["importance"][::-1], color=colors_bar[::-1])
    ax.set_xlabel("XGBoost Feature Importance (gain)")
    ax.set_title(f"Top-{top_n} Ensemble Features — {tag.upper()}", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="Tabular (kinematic)"),
        Patch(color="#3498db", label="LSTM embedding"),
    ], fontsize=9)
    plt.tight_layout()
    _save(fig, f"ensemble_feat_imp_{tag}_top{top_n}.png")

    # ── Figure 2: modality aggregate ─────────────────────────────────────────
    mod_imp = imp.groupby("modality")["importance"].sum().reset_index()
    mod_imp["pct"] = mod_imp["importance"] / mod_imp["importance"].sum() * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(mod_imp["modality"], mod_imp["pct"],
                  color=["#3498db", "#e74c3c"], edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, mod_imp["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Aggregate importance (%)")
    ax.set_title(f"Modality contribution — {tag.upper()}", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _save(fig, f"ensemble_modality_imp_{tag}.png")

    print(f"  Modality importance: {mod_imp.set_index('modality')['pct'].to_dict()}")

    # Save top features CSV for report
    imp.to_csv(MODELS_OUT / f"ensemble_feat_imp_{tag}.csv", index=False)
    print(f"  Feature importance CSV → {MODELS_OUT}/ensemble_feat_imp_{tag}.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Per-fold accuracy strip plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_fold_variance(results_a: dict, results_b: dict) -> None:
    """Show per-model accuracy distribution across folds (reconstructed from std)."""
    import json

    # Load per-fold data from JSON (mean ± std only stored, so reconstruct concept)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_order = ["XGBoost", "Random Forest", "SVM", "LSTM", "Bi-LSTM",
                   "Ensemble XGB", "Ensemble RF", "Ensemble SVM"]
    all_results = {"NM vs KOA": results_a, "KOA Staging": results_b}

    for ax, (task, results) in zip(axes, all_results.items()):
        models  = [m for m in model_order if m in results]
        means   = [results[m].get("accuracy", 0)     for m in models]
        stds    = [results[m].get("accuracy_std", 0) for m in models]

        colors = []
        for m in models:
            if "Ensemble" in m:  colors.append("#2ecc71")
            elif m in ("LSTM", "Bi-LSTM"): colors.append("#e67e22")
            else: colors.append("#3498db")

        x = np.arange(len(models))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, edgecolor="white",
               capsize=5, error_kw={"lw": 2})
        for xi, (m, s) in zip(x, zip(means, stds)):
            ax.text(xi, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Accuracy (mean ± std)")
        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y")

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#3498db", alpha=0.8, label="Tabular"),
            Patch(color="#e67e22", alpha=0.8, label="LSTM only"),
            Patch(color="#2ecc71", alpha=0.8, label="Ensemble"),
        ], fontsize=8)

    fig.suptitle("Model Accuracy: Mean ± Std across 5-fold CV", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "all_models_accuracy_cv.png")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> None:
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


import json


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate report figures for gait analysis classification results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --fast-only   Only regenerate figures from existing JSON results (seconds).
                Produces: all_models_accuracy_cv.png, cross_task_comparison_cv.png
                Does NOT retrain any model.

  (default)     Full run: re-trains all models for OOF ROC curves + feature
                importance. Takes 1-3h on CPU (GPU recommended).
                Produces: roc_oof_*.png, ensemble_feat_imp_*.png + all fast figures.
""",
    )
    parser.add_argument("--epochs",    type=int,  default=200)
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--fast-only", action="store_true",
                        help="Only regenerate figures that require no model training.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load existing CV results (always available) ───────────────────────────
    res_nmkoa   = {**_load_json(MODELS_OUT / "results_nmkoa_cv.json"),
                   **_load_json(MODELS_OUT / "results_nmkoa_ensemble_cv.json")}
    res_staging = {**_load_json(MODELS_OUT / "results_koastage_cv.json"),
                   **_load_json(MODELS_OUT / "results_koastage_ensemble_cv.json")}

    # ── Fast: overall accuracy bar chart (JSON only, no training) ────────────
    print("\n[1/4] Accuracy overview figure (fast)...")
    plot_fold_variance(res_nmkoa, res_staging)

    if args.fast_only:
        print("\n--fast-only: skipping OOF ROC and feature importance (require model training).")
        print(f"Figures → {FIGURES_OUT}")
        return

    # ── Slow: OOF ROC — Task A ───────────────────────────────────────────────
    print("\n[2/4] OOF ROC curves — Task A (NM vs KOA)...  [re-trains models, slow on CPU]")
    tab_a = load_tabular_dataset(filter_groups=["NM", "KOA"], label_mode="nm_koa")
    seq_a = load_sequence_dataset(filter_groups=["NM", "KOA"], label_mode="nm_koa")
    labels_a = ["NM", "KOA"]

    oof_tab_a  = collect_oof_tabular(tab_a, labels_a, seed=args.seed)
    print("  tabular OOF done; collecting LSTM...")
    oof_lstm_a = collect_oof_lstm(seq_a, labels_a, seed=args.seed, epochs=args.epochs)
    print("  LSTM OOF done; collecting ensemble...")
    oof_ens_a  = collect_oof_ensemble(tab_a, seq_a, labels_a, seed=args.seed, epochs=args.epochs)

    plot_oof_roc({**oof_tab_a, **oof_lstm_a},
                 labels_a, "ROC Curves OOF — NM vs KOA", "roc_oof_nmkoa_standalone.png")
    plot_oof_roc({"Ensemble XGB": oof_ens_a},
                 labels_a, "ROC Curve OOF — Ensemble XGB (NM vs KOA)", "roc_oof_nmkoa_ensemble.png")

    # ── Slow: OOF ROC — Task B ───────────────────────────────────────────────
    print("\n[3/4] OOF ROC curves — Task B (KOA Staging)...")
    tab_b = load_tabular_dataset(filter_groups=["KOA"], label_mode="stage")
    seq_b = load_sequence_dataset(filter_groups=["KOA"], label_mode="stage")
    labels_b = ["EL", "MD", "SV"]

    oof_tab_b  = collect_oof_tabular(tab_b, labels_b, seed=args.seed)
    print("  tabular OOF done; collecting LSTM...")
    oof_lstm_b = collect_oof_lstm(seq_b, labels_b, seed=args.seed, epochs=args.epochs)
    print("  LSTM OOF done; collecting ensemble...")
    oof_ens_b  = collect_oof_ensemble(tab_b, seq_b, labels_b, seed=args.seed, epochs=args.epochs)

    plot_oof_roc({**oof_tab_b, **oof_lstm_b},
                 labels_b, "ROC Curves OOF — KOA Staging", "roc_oof_koastage_standalone.png")
    plot_oof_roc({"Ensemble XGB": oof_ens_b},
                 labels_b, "ROC Curve OOF — Ensemble XGB (KOA Staging)", "roc_oof_koastage_ensemble.png")

    # ── Slow: Ensemble feature importance ────────────────────────────────────
    print("\n[4/4] Ensemble feature importance...")
    print("  Task A (NM vs KOA)...")
    plot_ensemble_feature_importance(tab_a, seq_a, labels_a, "nmkoa",
                                      epochs=args.epochs, seed=args.seed)
    print("  Task B (KOA Staging)...")
    plot_ensemble_feature_importance(tab_b, seq_b, labels_b, "koastage",
                                      epochs=args.epochs, seed=args.seed)

    print("\nAll figures done.")
    print(f"Figures → {FIGURES_OUT}")


if __name__ == "__main__":
    main()
