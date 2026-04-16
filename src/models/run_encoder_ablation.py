"""
Ablation: LSTM vs Bi-LSTM como encoder do ensemble.

Para cada tipo de encoder e cada fold de CV:
  1. Treina o encoder (LSTM ou Bi-LSTM) nos ciclos de treino do fold
  2. Extrai o embedding do último timestep (64-dim para LSTM, 128-dim para Bi-LSTM)
  3. Concatena com features tabulares → vetor combinado
  4. Treina XGBoost classificador no vetor combinado
  5. Avalia no fold de teste

Comparação direta e justa:
  - Mesmos folds (mesma semente e splits)
  - Mesmo XGBoost classificador
  - Mesma arquitetura base (hidden_size=64, 2 camadas, dropout=0.3)
  - Única diferença: bidirectional=False (LSTM) vs bidirectional=True (Bi-LSTM)
  - Dimensão do embedding: 64-dim (LSTM) vs 128-dim (Bi-LSTM)
  - Vetor combinado:       115-dim (LSTM) vs 179-dim (Bi-LSTM)

Produz:
  data/output/models/results_encoder_ablation.json
  data/output/figures/models/encoder_ablation_nmkoa.png
  data/output/figures/models/encoder_ablation_koastage.png

Uso:
    venv/bin/python src/models/run_encoder_ablation.py [--epochs 200] [--seed 42] [--folds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.dataset import (
    load_tabular_dataset,
    load_sequence_dataset,
    compute_class_weights,
)
from src.models.lstm_model import build_lstm, build_bilstm
from src.models.tabular import build_xgboost, fit_predict
from src.models.train_lstm import train_model, TrainConfig
from src.models.evaluate import compute_metrics
from src.models.run_focused import _summarize_folds, _print_cv_line
from src.models.run_ensemble import _build_subject_tab, _cycles_to_subject_features

MODELS_OUT  = Path("data/output/models")
FIGURES_OUT = Path("data/output/figures/models")


# ──────────────────────────────────────────────────────────────────────────────
# Extração de embedding (genérico para LSTM e Bi-LSTM)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_embedding(
    model, X: np.ndarray, device: str | None = None, batch_size: int = 256
) -> np.ndarray:
    if device is None:
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    """Extrai embedding do último timestep de um GaitLSTM (bidirecional ou não).

    Para LSTM:    retorna (N, 64)  — hidden_size × 1 direção
    Para Bi-LSTM: retorna (N, 128) — hidden_size × 2 direções

    Acessa model.lstm e model.dropout diretamente; GaitLSTM não é modificado.
    Em modo eval, dropout é no-op — embeddings são determinísticos.
    """
    model.eval()
    dev   = torch.device(device)
    model = model.to(dev)

    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb   = torch.tensor(
                X[start : start + batch_size], dtype=torch.float32
            ).to(dev)
            out, _ = model.lstm(xb)    # (B, 101, hidden * directions)
            last   = out[:, -1, :]     # último timestep: (B, emb_dim)
            last   = model.dropout(last)
            parts.append(last.cpu().numpy())

    return np.vstack(parts)


# ──────────────────────────────────────────────────────────────────────────────
# CV por tipo de encoder
# ──────────────────────────────────────────────────────────────────────────────

def run_encoder_cv(
    tab,
    seq,
    label_names: list[str],
    encoder_type: str,          # "lstm" ou "bilstm"
    tag: str,
    epochs: int,
    seed: int,
    n_splits: int = 5,
) -> dict:
    """5-fold CV com encoder fixo (LSTM ou Bi-LSTM) + XGBoost classificador.

    Retorna dict com métricas agregadas dos folds.
    """
    assert encoder_type in ("lstm", "bilstm"), "encoder_type deve ser 'lstm' ou 'bilstm'"

    bidirectional = encoder_type == "bilstm"
    directions    = 2 if bidirectional else 1
    emb_dim       = 64 * directions   # 64 (LSTM) ou 128 (Bi-LSTM)
    n_classes     = len(label_names)
    subj_tab      = _build_subject_tab(tab)
    n_tab_feat    = len(tab.feature_names)
    combined_dim  = emb_dim + n_tab_feat

    label = "Bi-LSTM" if bidirectional else "LSTM"
    print(f"    Encoder: {label} | emb={emb_dim}-dim | "
          f"combined={combined_dim}-dim ({emb_dim}+{n_tab_feat})")

    # Arrays de sujeito para stratificação
    unique_subjs  = np.unique(seq.groups)
    subj_lbl      = {
        s: int(np.bincount(seq.y[seq.groups == s]).argmax())
        for s in unique_subjs
    }
    subj_arr     = np.array(unique_subjs)
    subj_lbl_arr = np.array([subj_lbl[s] for s in unique_subjs])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cfg  = TrainConfig(
        epochs=epochs, batch_size=32, lr=1e-3,
        patience=30, monitor="val_acc",
        early_stop=False,
        verbose=False, log_every=epochs + 1,
    )

    clf_name     = f"Ensemble XGB ({label})"
    fold_metrics = []
    fold_cms     = []

    for fold, (tr_si, te_si) in enumerate(
        sgkf.split(subj_arr, subj_lbl_arr, groups=subj_arr), start=1
    ):
        tr_subjs = set(subj_arr[tr_si])
        te_subjs = set(subj_arr[te_si])

        tr_mask = np.isin(seq.groups, list(tr_subjs))
        te_mask = np.isin(seq.groups, list(te_subjs))

        X_seq_tr, X_seq_te = seq.X[tr_mask], seq.X[te_mask]
        y_tr,     y_te     = seq.y[tr_mask], seq.y[te_mask]

        cw = compute_class_weights(y_tr)

        # Treina encoder dentro do fold (sem acesso a dados de teste)
        torch.manual_seed(seed + fold)
        if bidirectional:
            model = build_bilstm(
                input_size=4, hidden_size=64, num_layers=2,
                num_classes=n_classes, dropout=0.3,
            )
        else:
            model = build_lstm(
                input_size=4, hidden_size=64, num_layers=2,
                num_classes=n_classes, dropout=0.3,
            )

        best_model, _ = train_model(
            model, X_seq_tr, y_tr, X_seq_tr, y_tr,
            class_weights=cw, cfg=cfg,
        )

        # Extrai embeddings do encoder treinado no fold
        emb_tr = _extract_embedding(best_model, X_seq_tr)
        emb_te = _extract_embedding(best_model, X_seq_te)

        # Concatena com features tabulares
        X_tr_comb, _ = _cycles_to_subject_features(
            seq.groups[tr_mask], emb_tr, subj_tab
        )
        X_te_comb, _ = _cycles_to_subject_features(
            seq.groups[te_mask], emb_te, subj_tab
        )

        # Treina e avalia XGBoost
        clf = build_xgboost(n_classes=n_classes, random_state=seed)
        y_pred, y_proba = fit_predict(clf, X_tr_comb, y_tr, X_te_comb)
        m = compute_metrics(y_te, y_pred, y_proba, label_names)
        fold_metrics.append(m)
        fold_cms.append(
            confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))
        )

        print(f"      fold {fold}/{n_splits}  "
              f"(train {len(tr_subjs)} / test {len(te_subjs)} subj)  "
              f"acc={m['accuracy']:.3f}  f1={m['macro_f1']:.3f}  "
              f"auc={m.get('roc_auc_macro', float('nan')):.3f}")

    summary = _summarize_folds(fold_metrics)
    return {
        "metrics": summary,
        "encoder": label,
        "emb_dim": emb_dim,
        "combined_dim": combined_dim,
        "fold_cms": [cm.tolist() for cm in fold_cms],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figura comparativa
# ──────────────────────────────────────────────────────────────────────────────

def plot_ablation(
    lstm_res: dict,
    bilstm_res: dict,
    tag: str,
    title: str,
) -> None:
    """Gráfico lado a lado: LSTM encoder vs Bi-LSTM encoder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    metrics       = ["accuracy", "macro_f1", "roc_auc_macro"]
    metric_labels = ["Accuracy", "Macro F1", "ROC AUC"]
    encoders      = [
        ("LSTM encoder\n(64-dim)", lstm_res,   "steelblue"),
        ("Bi-LSTM encoder\n(128-dim)", bilstm_res, "darkorange"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, mk, ml in zip(axes, metrics, metric_labels):
        for xi, (enc_label, res, color) in enumerate(encoders):
            m   = res["metrics"]
            val = m.get(mk, 0)
            std = m.get(f"{mk}_std", 0)

            ax.bar(xi, val, color=color, alpha=0.82, edgecolor="white", width=0.6)
            ax.errorbar(xi, val, yerr=std, fmt="none",
                        color="black", capsize=5, lw=2)
            ax.text(xi, val + std + 0.012, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [e[0] for e in encoders], fontsize=9
        )
        ax.set_ylim(0, 1.15)
        ax.set_title(ml, fontsize=11)
        ax.set_ylabel("Score (mean ± std, 5-fold CV)" if ax is axes[0] else "")
        ax.grid(True, alpha=0.25, axis="y")

    # Anotação com dimensionalidade
    for ax in axes:
        ax.annotate(
            f"LSTM: {lstm_res['combined_dim']}-dim  |  "
            f"Bi-LSTM: {bilstm_res['combined_dim']}-dim",
            xy=(0.5, -0.18), xycoords="axes fraction",
            ha="center", fontsize=7.5, color="gray",
        )

    legend_handles = [
        Patch(color="steelblue",  alpha=0.82,
              label=f"LSTM encoder ({lstm_res['emb_dim']}-dim emb)"),
        Patch(color="darkorange", alpha=0.82,
              label=f"Bi-LSTM encoder ({bilstm_res['emb_dim']}-dim emb)"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_OUT / f"encoder_ablation_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Tabela de resultados no terminal
# ──────────────────────────────────────────────────────────────────────────────

def print_ablation_table(lstm_res: dict, bilstm_res: dict, task_name: str) -> None:
    metrics = ["accuracy", "macro_f1", "roc_auc_macro"]
    labels  = ["Accuracy", "Macro F1", "ROC AUC"]

    print(f"\n{'─'*65}")
    print(f"  {task_name} — Encoder Ablation")
    print(f"{'─'*65}")
    header = f"  {'Encoder':<22} {'Emb dim':>8} {'Comb dim':>9}  "
    header += "  ".join(f"{l:>12}" for l in labels)
    print(header)
    print(f"{'─'*65}")

    for res in [lstm_res, bilstm_res]:
        m    = res["metrics"]
        row  = f"  {res['encoder']:<22} {res['emb_dim']:>8} {res['combined_dim']:>9}  "
        for mk in metrics:
            val = m.get(mk, float("nan"))
            std = m.get(f"{mk}_std", float("nan"))
            row += f"  {val:.3f}±{std:.3f}"
        print(row)

    print(f"{'─'*65}")

    # Delta (Bi-LSTM - LSTM)
    print("  Delta (Bi-LSTM − LSTM):", end="")
    for mk in metrics:
        d = bilstm_res["metrics"].get(mk, 0) - lstm_res["metrics"].get(mk, 0)
        sign = "+" if d >= 0 else ""
        print(f"  {sign}{d:+.3f}", end="")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Runners por tarefa
# ──────────────────────────────────────────────────────────────────────────────

def run_task(
    filter_groups: list[str],
    label_mode: str,
    label_names: list[str],
    tag: str,
    title: str,
    epochs: int,
    seed: int,
    n_splits: int,
) -> dict:
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")

    tab = load_tabular_dataset(filter_groups=filter_groups, label_mode=label_mode)
    seq = load_sequence_dataset(filter_groups=filter_groups, label_mode=label_mode)

    print(f"  Dataset: {seq.X.shape[0]} ciclos | "
          f"{len(np.unique(seq.groups))} sujeitos | "
          f"{len(label_names)} classes: {label_names}")

    print(f"\n  [LSTM encoder]")
    lstm_res = run_encoder_cv(
        tab, seq, label_names, "lstm", tag, epochs, seed, n_splits
    )

    print(f"\n  [Bi-LSTM encoder]")
    bilstm_res = run_encoder_cv(
        tab, seq, label_names, "bilstm", tag, epochs, seed, n_splits
    )

    print_ablation_table(lstm_res, bilstm_res, title)
    plot_ablation(lstm_res, bilstm_res, tag, title)

    return {"lstm": lstm_res, "bilstm": bilstm_res}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablação LSTM vs Bi-LSTM como encoder do ensemble"
    )
    parser.add_argument("--epochs", type=int, default=200,
                        help="Épocas máximas de treino do encoder (default: 200)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Semente aleatória (default: 42)")
    parser.add_argument("--folds",  type=int, default=5,
                        help="Número de folds CV (default: 5)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_results: dict = {}

    # Tarefa A — NM vs KOA
    all_results["nmkoa"] = run_task(
        filter_groups=["NM", "KOA"],
        label_mode="nm_koa",
        label_names=["NM", "KOA"],
        tag="nmkoa",
        title="TASK A — NM vs KOA  |  Encoder Ablation",
        epochs=args.epochs,
        seed=args.seed,
        n_splits=args.folds,
    )

    # Tarefa B — KOA Staging
    all_results["koastage"] = run_task(
        filter_groups=["KOA"],
        label_mode="stage",
        label_names=["EL", "MD", "SV"],
        tag="koastage",
        title="TASK B — KOA Staging  |  Encoder Ablation",
        epochs=args.epochs,
        seed=args.seed,
        n_splits=args.folds,
    )

    # Salva resultados (sem fold_cms para JSON limpo)
    output: dict = {}
    for task, res in all_results.items():
        output[task] = {}
        for enc, data in res.items():
            output[task][enc] = {
                "encoder":       data["encoder"],
                "emb_dim":       data["emb_dim"],
                "combined_dim":  data["combined_dim"],
                "metrics":       data["metrics"],
            }

    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_OUT / "results_encoder_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Resultados → {out_path}")

    # Sumário final
    print(f"\n{'═'*60}")
    print("  SUMÁRIO FINAL — Encoder Ablation")
    print(f"{'═'*60}")
    task_names = {
        "nmkoa":    "Task A (NM vs KOA)",
        "koastage": "Task B (KOA Staging)",
    }
    for task, res in output.items():
        print(f"\n  {task_names.get(task, task)}:")
        for enc, data in res.items():
            m = data["metrics"]
            print(f"    {data['encoder']:10s} ({data['emb_dim']:3d}-dim emb) "
                  f"| acc={m['accuracy']:.3f}±{m['accuracy_std']:.3f} "
                  f"| f1={m['macro_f1']:.3f}±{m['macro_f1_std']:.3f} "
                  f"| auc={m.get('roc_auc_macro', float('nan')):.3f}"
                  f"±{m.get('roc_auc_macro_std', float('nan')):.3f}")

    print(f"\n  Figuras → {FIGURES_OUT}/encoder_ablation_*.png")
    print("  Concluído.")


if __name__ == "__main__":
    main()
