# Pipeline de Classificação — Guia de Execução Completo

Todos os comandos executados na raiz do projeto:
  cd /home/claucio/Documents/analise_marcha

## Pré-requisito: GPU funcionando

  venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  # Deve retornar: True  NVIDIA GeForce RTX 3060
  # Se False: venv/bin/pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124

---

## Passo 1 — Modelos standalone  (~8-12 min com GPU)
## Tabular (XGBoost, RF, SVM) + LSTM/Bi-LSTM, 5-fold CV por sujeito

  venv/bin/python src/models/run_focused.py --epochs 200 --seed 42 --folds 5

  Saídas:
    data/output/models/results_nmkoa_cv.json
    data/output/models/results_koastage_cv.json
    data/output/figures/models/cm_nmkoa_*.png
    data/output/figures/models/cm_koastage_*.png
    data/output/figures/models/feat_imp_*.png
    data/output/figures/models/cv_comparison_*.png
    data/output/figures/models/cross_task_comparison_cv.png

---

## Passo 2 — Ensemble (Bi-LSTM + tabular)  (~10-15 min com GPU)
## Requer Passo 1

  venv/bin/python src/models/run_ensemble.py --epochs 200 --seed 42 --folds 5

  Saídas:
    data/output/models/results_nmkoa_ensemble_cv.json
    data/output/models/results_koastage_ensemble_cv.json
    data/output/figures/models/ensemble_comparison_nmkoa.png
    data/output/figures/models/ensemble_comparison_koastage.png

---

## Passo 3 — Figuras de visão geral  (< 5 seg, sem treino)
## Requer Passo 1 + 2

  venv/bin/python src/models/generate_report_figures.py --fast-only

  Saídas:
    data/output/figures/models/all_models_accuracy_cv.png

---

## Passo 4 — Ablação do encoder: LSTM vs Bi-LSTM  (~15-20 min com GPU)
## Opcional. Pode rodar em paralelo com Passo 5.

  venv/bin/python src/models/run_encoder_ablation.py --epochs 200 --seed 42 --folds 5

  Saídas:
    data/output/models/results_encoder_ablation.json
    data/output/figures/models/encoder_ablation_nmkoa.png
    data/output/figures/models/encoder_ablation_koastage.png

---

## Passo 5 — ROC OOF + feature importance do ensemble  (~20-40 min com GPU)
## Opcional. Re-treina todos os modelos do zero para coleta OOF.
## Pode rodar em paralelo com Passo 4.

  venv/bin/python src/models/generate_report_figures.py --epochs 200 --seed 42

  Saídas:
    data/output/figures/models/roc_oof_nmkoa_standalone.png
    data/output/figures/models/roc_oof_nmkoa_ensemble.png
    data/output/figures/models/roc_oof_koastage_standalone.png
    data/output/figures/models/roc_oof_koastage_ensemble.png
    data/output/figures/models/ensemble_feat_imp_nmkoa_top25.png
    data/output/figures/models/ensemble_modality_imp_nmkoa.png
    data/output/figures/models/ensemble_feat_imp_koastage_top25.png
    data/output/figures/models/ensemble_modality_imp_koastage.png

---

## Ordem recomendada (executar em sequência)

  Passo 1  →  Passo 2  →  Passo 3  →  Passo 4 e Passo 5 em paralelo

## Para refazer tudo do zero (limpar resultados anteriores):

  rm -f data/output/models/*.json
  rm -f data/output/figures/models/*.png
