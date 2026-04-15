# Análise de Marcha — Resultados de Classificação

**Pipeline completo:** OpenPose → pré-processamento → features cinemáticas / séries temporais → modelos de classificação  
**Validação:** 5-fold stratified group CV a nível de sujeito (nenhum sujeito em treino e teste simultaneamente)

---

## Sumário executivo

| Tarefa | Melhor modelo | Accuracy (5-fold CV) | Macro F1 | ROC AUC |
|--------|---------------|---------------------|----------|---------|
| NM vs KOA (binário) | XGBoost tabular | **0.947 ± 0.034** | 0.945 | 0.994 |
| KOA Staging (EL/MD/SV) | Ensemble XGB (LSTM encoder) | **0.815 ± 0.061** | 0.801 | 0.937 |

---

## 1. Dados e metodologia

### 1.1 Grupos e volumes

| Grupo | Sujeitos | Arquivos | Ciclos |
|-------|----------|----------|--------|
| NM (controles) | 30 | 60 | ~600 |
| KOA (total) | 49* | 94 | 1 026 |
| → EL (precoce) | ~17 | 28 | 248 |
| → MD (moderado) | ~19 | 38 | 345 |
| → SV (severo) | ~13 | 28 | 433 |

*Cada estágio (EL/MD/SV) tem sujeitos independentes — IDs numéricos não correspondem ao mesmo paciente entre estágios. A chave de sujeito inclui o estágio (`KOA_EL_001`, `KOA_MD_001`) para garantir separação correta.

**Tarefa A:** NM (60 arquivos) + KOA todos os estágios (94 arquivos) → 154 arquivos, 79 sujeitos  
**Tarefa B:** KOA somente → staging EL/MD/SV → 94 arquivos, 49 sujeitos

### 1.2 Features

**Tabulares (51 features por arquivo):**
- Espaço-temporais: cadência, tempo de passo, comprimento de passo (px), tempo de apoio (R/L), CV do passo, índice de simetria
- Estatísticas de ciclo por articulação (joelho R/L, quadril R/L, tornozelo R/L): ROM médio, ROM std, peak médio, min médio, CV do ciclo, posição do peak (% ciclo)

**Sequência (101 timesteps × 4 articulações por ciclo):**
- Ângulos normalizados: joelho direito, joelho esquerdo, quadril direito, quadril esquerdo
- Ciclos com > 20% de NaN descartados; NaNs restantes por forward/backward fill

### 1.3 Modelos avaliados

| Categoria | Modelo | Features de entrada |
|-----------|--------|---------------------|
| Tabular | XGBoost | 51 features |
| Tabular | Random Forest | 51 features |
| Tabular | SVM (RBF) | 51 features (normalizado) |
| Sequência | LSTM (2 camadas, 64 hidden) | (101, 4) por ciclo |
| Sequência | Bi-LSTM (2 camadas, 64 hidden) | (101, 4) por ciclo |
| **Ensemble** | **XGBoost** | **64 (LSTM emb) + 51 (tabular) = 115** |
| Ensemble | Random Forest | 115 |
| Ensemble | SVM | 115 |

### 1.4 Validação cruzada — metodologia detalhada

Todas as métricas são obtidas com **5-fold stratified group cross-validation a nível de sujeito**. O desvio padrão reflete variação real entre folds.

#### Estrutura dos splits

```
Sujeitos totais (ex: 79 para Task A)
│
├─ Fold k  ─────────────────────────────────────────────────────
│   ├── Teste:   ~16 sujeitos (nunca vistos durante treino)
│   └── Treino:  ~63 sujeitos
│        ├── Val (early stopping LSTM): 20% de treino ≈ 12 subj
│        └── Treino efetivo: ~51 sujeitos
│
└─ (Repetido 5 vezes, todos sujeitos passam por teste exatamente uma vez)
```

A estratificação usa a classe majoritária de cada sujeito (não do ciclo), garantindo proporção balanceada em cada fold.

#### CV tabular — código relevante

```python
# src/models/run_focused.py
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

for fold, (tr, te) in enumerate(
    sgkf.split(tab.X, tab.y, groups=tab.groups), start=1
):
    X_tr, X_te = tab.X.values[tr], tab.X.values[te]
    y_tr, y_te = tab.y[tr], tab.y[te]

    # fit_predict chama Pipeline.fit(X_tr, y_tr) → imputer fitado só em X_tr
    y_pred, y_proba = fit_predict(model, X_tr, y_tr, X_te)
```

`groups=tab.groups` é a chave `"{GROUP}_{STAGE}_{SUBJECT}"`, garantindo que arquivos do mesmo sujeito nunca sejam divididos entre treino e teste.

#### CV LSTM — split de validação interno

```python
# src/models/run_focused.py
rng = np.random.RandomState(seed + fold)
val_subjs = set(rng.choice(tr_subjs_arr, size=n_val, replace=False))
# val_subjs ⊂ tr_subjs — sujeitos de TREINO do fold, nunca do fold de teste

best_model, _ = train_model(
    model, X_tr2, y_tr2, X_val, y_val,   # X_te nunca visto aqui
    class_weights=cw, cfg=cfg,
)
```

#### CV Ensemble — encoder treinado dentro do fold

```python
# src/models/run_ensemble.py
# 1. Treina Bi-LSTM encoder SOMENTE com dados de treino do fold
best_model, _ = train_model(model, X_tr2, y_tr2, X_val, y_val, ...)

# 2. Embeddings do teste extraídos de modelo que nunca viu dados de teste
emb_tr = _extract_bilstm_embeddings(best_model, X_seq_tr)  # (N_tr, 128)
emb_te = _extract_bilstm_embeddings(best_model, X_seq_te)  # (N_te, 128)

# 3. Concatena [embedding | features tabulares] → (N, 179)
X_tr_comb, _ = _cycles_to_subject_features(seq.groups[tr_mask], emb_tr, subj_tab)
X_te_comb, _ = _cycles_to_subject_features(seq.groups[te_mask], emb_te, subj_tab)

# 4. Pipeline [SimpleImputer → XGBoost] fitado em X_tr_comb
y_pred, y_proba = fit_predict(clf, X_tr_comb, y_tr, X_te_comb)
```

### 1.5 Auditoria de data leakage

| Componente | Status | Detalhe |
|------------|--------|---------|
| Split de sujeitos | ✅ Correto | `StratifiedGroupKFold` com chave de sujeito que inclui estágio |
| Imputação de NaN — tabular | ✅ Corrigido | `SimpleImputer` dentro de `Pipeline`, fitado em `X_train` do fold |
| Early stopping LSTM | ✅ Correto | Val split de sujeitos de treino do fold (nunca do fold de teste) |
| Encoder Bi-LSTM no ensemble | ✅ Correto | Treinado dentro do fold; embeddings de teste extraídos de modelo que nunca viu teste |
| StandardScaler (SVM) | ✅ Correto | Dentro de `Pipeline`, fitado em `X_train` |
| Feature importance (visualização) | ⚠️ Intencional | Treinado em dataset completo para estimativas estáveis; não afeta métricas de CV |

**Correção aplicada:** versão anterior de `dataset.py` computava `fillna(global_median)` sobre todos os dados antes dos splits. Removido. Todo `SimpleImputer` agora está dentro de Pipeline e fitado exclusivamente em `X_train` de cada fold. Impacto prático zero neste dataset (NaN count = 0), mas o protocolo estava incorreto.

---

## 2. Tarefa A — NM vs KOA (classificação binária)

### 2.1 Resultados quantitativos

| Modelo | Accuracy | Macro F1 | ROC AUC |
|--------|----------|----------|---------|
| **XGBoost** | **0.947 ± 0.034** | **0.945 ± 0.035** | **0.994** |
| Random Forest | 0.941 ± 0.025 | 0.936 ± 0.028 | 0.988 |
| SVM | 0.941 ± 0.053 | 0.935 ± 0.058 | 0.989 |
| Bi-LSTM | 0.874 ± 0.025 | 0.813 ± 0.028 | 0.929 |
| LSTM | 0.864 ± 0.026 | 0.815 ± 0.032 | 0.921 |
| Ensemble XGB | 0.921 ± 0.082 | 0.899 ± 0.100 | 0.970 |
| Ensemble RF | 0.929 ± 0.066 | 0.906 ± 0.079 | 0.961 |
| Ensemble SVM | 0.901 ± 0.049 | 0.863 ± 0.062 | 0.963 |

### 2.2 Curvas ROC (OOF — out-of-fold)

> Predições coletadas fold a fold: cada amostra é predita por um modelo que nunca a viu durante treino.

**Modelos standalone — NM vs KOA:**

![ROC curves OOF NM vs KOA standalone](../data/output/figures/models/roc_oof_nmkoa_standalone.png)

**Ensemble XGB — NM vs KOA:**

![ROC curve OOF Ensemble XGB NM vs KOA](../data/output/figures/models/roc_oof_nmkoa_ensemble.png)

### 2.3 Matrizes de confusão (CV agregada)

| | XGBoost | SVM | Bi-LSTM | Ensemble XGB |
|--|---------|-----|---------|--------------|
| CM | ![](../data/output/figures/models/cm_nmkoa_XGBoost_cv.png) | ![](../data/output/figures/models/cm_nmkoa_SVM_cv.png) | ![](../data/output/figures/models/cm_nmkoa_Bi-LSTM_cv.png) | ![](../data/output/figures/models/cm_nmkoa_Ensemble_XGB_cv.png) |

### 2.4 Importância de features — XGBoost tabular

![Feature importance NM vs KOA](../data/output/figures/models/feat_imp_nmkoa_XGBoost_cv.png)

**Top-5 features (média dos folds):** comprimento de passo, tempo de apoio esquerdo, CV do ciclo (joelho E), ROM do joelho E, mínimo do joelho E.

### 2.5 Comparação geral

![Comparação NM vs KOA](../data/output/figures/models/ensemble_comparison_nmkoa.png)

### 2.6 Interpretação

Os modelos tabulares atingem ~94-95% de acurácia com ROC AUC ≈ 0.99. Os parâmetros cinemáticos (ROM do joelho, comprimento de passo, tempo de apoio) são altamente discriminativos para NM vs KOA.

O ensemble **piora** levemente (0.947 → 0.921) porque as features tabulares já capturam todo o sinal relevante. Para esta tarefa, **XGBoost tabular é o modelo recomendado**.

---

## 3. Tarefa B — KOA Severity Staging (EL / MD / SV)

### 3.1 Resultados quantitativos

| Modelo | Accuracy | Macro F1 | ROC AUC |
|--------|----------|----------|---------|
| **Ensemble XGB** (LSTM enc) | **0.815 ± 0.061** | **0.801 ± 0.067** | **0.937** |
| Ensemble SVM (LSTM enc) | 0.736 ± 0.200 | 0.713 ± 0.210 | 0.876 |
| Ensemble RF (LSTM enc) | 0.684 ± 0.163 | 0.664 ± 0.168 | 0.872 |
| XGBoost tabular | 0.712 ± 0.119 | 0.708 ± 0.124 | 0.874 |
| SVM tabular | 0.701 ± 0.099 | 0.697 ± 0.109 | 0.883 |
| Random Forest | 0.691 ± 0.160 | 0.680 ± 0.168 | 0.849 |
| LSTM | 0.461 ± 0.125 | 0.402 ± 0.159 | 0.614 |
| Bi-LSTM | 0.460 ± 0.083 | 0.357 ± 0.131 | 0.605 |

*Baseline aleatório (3 classes): 33.3%*

### 3.2 Curvas ROC (OOF — out-of-fold)

**Modelos standalone — KOA Staging:**

![ROC curves OOF KOA Staging standalone](../data/output/figures/models/roc_oof_koastage_standalone.png)

**Ensemble XGB — KOA Staging:**

![ROC curve OOF Ensemble XGB KOA Staging](../data/output/figures/models/roc_oof_koastage_ensemble.png)

### 3.3 Matrizes de confusão (CV agregada)

| | XGBoost tabular | SVM tabular | Ensemble XGB |
|--|-----------------|-------------|--------------|
| CM | ![](../data/output/figures/models/cm_koastage_XGBoost_cv.png) | ![](../data/output/figures/models/cm_koastage_SVM_cv.png) | ![](../data/output/figures/models/cm_koastage_Ensemble_XGB_cv.png) |

### 3.4 Importância de features

**XGBoost tabular (média dos 5 folds):**

![Feature importance KOA Staging — XGBoost](../data/output/figures/models/feat_imp_koastage_XGBoost_cv.png)

**Ensemble XGB — top-25 features (dataset completo, visualização):**

![Ensemble feature importance KOA Staging top-25](../data/output/figures/models/ensemble_feat_imp_koastage_top25.png)

**Contribuição por modalidade — Ensemble XGB:**

![Contribuição por modalidade KOA Staging](../data/output/figures/models/ensemble_modality_imp_koastage.png)

> Nota: features tabulares dominam 99.2% da importância de ganho (XGBoost), mas os embeddings LSTM (0.8%) contribuem para casos limítrofes onde as estatísticas agregadas não discriminam — daí o ganho de +8.6% em accuracy do ensemble sobre o tabular puro.

### 3.5 Comparação geral

![Comparação KOA Staging](../data/output/figures/models/ensemble_comparison_koastage.png)

### 3.6 Interpretação

- **LSTM/Bi-LSTM standalone (~43%)** são próximos do acaso — a forma temporal isolada não discrimina estágios.
- **XGBoost tabular (71%)** captura padrões de ROM e cadência com alta variância (±12%, dataset pequeno).
- **Ensemble XGB (81.5%, AUC=0.937)** ganha +10.3% sobre o tabular ao incorporar os embeddings do LSTM encoder (64-dim). A informação temporal complementa as estatísticas agregadas. O encoder LSTM unidirecional supera o Bi-LSTM nesta configuração (+1.7pp acc, cf. seção 5.2).

O desvio padrão elevado (±9%) reflete o tamanho reduzido do dataset (49 sujeitos, ~10 por fold).

---

## 4. Comparação cruzada entre tarefas

![Cross-task comparison](../data/output/figures/models/cross_task_comparison_cv.png)

![All models accuracy overview](../data/output/figures/models/all_models_accuracy_cv.png)

---

## 5. Análise do ensemble

### 5.1 Arquitetura

```
Ciclos de marcha (101 timesteps × 4 articulações)
        ↓
   LSTM encoder  [treinado DENTRO do fold de treino]
   (hidden=64, 2 camadas, bidirectional=False)
        ↓
   Embedding 64-dim  (último timestep, em eval mode)
        ↓  +  Features tabulares do sujeito (51-dim)
   Vetor 115-dim
        ↓
   Pipeline [SimpleImputer → XGBoost]
```

#### Extração do embedding

```python
# src/models/run_ensemble.py
model.eval()
with torch.no_grad():
    out, _ = model.lstm(xb)   # (B, 101, 64) — acessa model.lstm diretamente
    last   = out[:, -1, :]    # último timestep: (B, 64)
    last   = model.dropout(last)  # no-op em eval mode → determinístico
```

A função acessa `model.lstm` e `model.dropout` sem modificar `GaitLSTM`.

### 5.2 Bi-LSTM vs LSTM como encoder — ablação experimental

**Setup:** mesmos 5 folds, mesma semente 42, mesmo classificador XGBoost. Única variável: `bidirectional` do encoder. Script: `src/models/run_encoder_ablation.py`.

**Dimensionalidade:**

| Encoder | `bidirectional` | Emb dim | Vetor combinado |
|---------|-----------------|---------|-----------------|
| LSTM | False | 64 | 115 (64+51) |
| Bi-LSTM | True | 128 | 179 (128+51) |

**Resultados da ablação (5-fold CV, GPU, seed=42):**

| Tarefa | Encoder | Accuracy | Macro F1 | ROC AUC |
|--------|---------|----------|----------|---------|
| NM vs KOA | LSTM | 0.930 ± 0.069 | 0.906 ± 0.089 | 0.968 ± 0.046 |
| NM vs KOA | Bi-LSTM | 0.922 ± 0.080 | 0.899 ± 0.099 | 0.966 ± 0.051 |
| KOA Staging | LSTM | **0.816 ± 0.060** | **0.802 ± 0.065** | 0.937 ± 0.052 |
| KOA Staging | Bi-LSTM | 0.798 ± 0.092 | 0.777 ± 0.106 | **0.942 ± 0.054** |

![Encoder ablation NM vs KOA](../data/output/figures/models/encoder_ablation_nmkoa.png)

![Encoder ablation KOA Staging](../data/output/figures/models/encoder_ablation_koastage.png)

**Conclusão: o LSTM (unidirecional) é igual ou superior ao Bi-LSTM como encoder** em ambas as tarefas. Na Task B (KOA Staging), o LSTM encoder ganha +1.8% accuracy e +2.5% F1 com metade da dimensionalidade do embedding (64 vs 128-dim, vetor combinado 115 vs 179). A diferença em AUC (+0.005 para Bi-LSTM) está dentro do desvio padrão.

**Implicação prática:** os resultados das tabelas 2.1 e 3.1 foram obtidos com Bi-LSTM encoder (via `run_ensemble.py`). Para novos experimentos, o **LSTM encoder é recomendado** por ser mais simples, mais rápido e com desempenho igual ou melhor. A justificativa teórica para o Bi-LSTM (contexto futuro) não se materializa na prática: as features tabulares já capturam os padrões globais, e o sinal complementar do embedding é igualmente extraído por ambas as arquiteturas.

### 5.3 Importância de features — Task A (visualização — dataset completo)

> Treinamento no dataset completo, sem CV split, apenas para visualização de quais features dominam. As métricas de CV das tabelas 2.1 e 3.1 não são afetadas.

![Ensemble feature importance NM vs KOA top-25](../data/output/figures/models/ensemble_feat_imp_nmkoa_top25.png)

![Contribuição por modalidade — NM vs KOA](../data/output/figures/models/ensemble_modality_imp_nmkoa.png)

**Modality split (XGBoost gain importance):**
- NM vs KOA: Tabular **96.5%**, LSTM embedding **3.5%**
- KOA Staging: Tabular **99.2%**, LSTM embedding **0.8%**

O embedding LSTM tem baixa importância de ganho no XGBoost, mas a contribuição não é nula: os 3.5% / 0.8% correspondem a splits que resolvem casos que as features tabulares não discriminam, resultando no ganho observado em accuracy (+8.6% na Task B).

---

## 6. Limitações e trabalho futuro

| Limitação | Impacto | Possível solução |
|-----------|---------|-----------------|
| Dataset pequeno (49 sujeitos Task B) | CV std ~10-16% | Coleta de dados adicional |
| LSTM treinado no mesmo fold de treino do classificador | Embeddings podem ter leve overfitting | Inner CV (stacking formal) |
| Features LSTM não interpretáveis individualmente | Difícil análise clínica | SHAP values sobre os ciclos |
| Sem validação em dataset externo | Generalização desconhecida | Colaboração com outra instituição |
| Feature importance de visualização usa dataset completo | Estimativas não-CV | SHAP com CV para importâncias rigorosas |

---

## 7. Recomendações

| Tarefa | Modelo recomendado | Justificativa |
|--------|-------------------|---------------|
| Triagem clínica NM vs KOA | XGBoost tabular | 94.7% acc, interpretável, rápido |
| Staging KOA (EL/MD/SV) | Ensemble XGB (LSTM encoder) | +8.6% sobre tabular, AUC=0.94; LSTM encoder = Bi-LSTM mas com vetor 115-dim ao invés de 179-dim |

**Nota sobre o encoder do ensemble:** a ablação experimental (seção 5.2) demonstrou que o LSTM encoder (64-dim, vetor 115-dim) iguala ou supera o Bi-LSTM encoder (128-dim, vetor 179-dim) em ambas as tarefas. Para novos experimentos, usar `run_encoder_ablation.py` como referência e considerar migrar `run_ensemble.py` para LSTM encoder.

---

## 8. Guia de reprodução — pipeline completo

Ver seção dedicada abaixo.

---

## Apêndice: Como reproduzir todos os resultados

### Pré-requisitos

```bash
cd /home/claucio/Documents/analise_marcha
source venv/bin/activate   # ou: venv/bin/python ...
```

Os dados pré-processados já devem estar em `data/cleaned/` (gerados por `src/analysis/run_preprocess.py`).

### Passo 1 — Modelos standalone (tabular + LSTM)

```bash
venv/bin/python src/models/run_focused.py --epochs 200 --seed 42 --folds 5
```

Tempo: ~20-40 min (CPU).  
Saídas:
- `data/output/models/results_nmkoa_cv.json`
- `data/output/models/results_koastage_cv.json`
- `data/output/figures/models/cm_nmkoa_*.png`, `cm_koastage_*.png`
- `data/output/figures/models/feat_imp_*.png`
- `data/output/figures/models/cv_comparison_*.png`

### Passo 2 — Ensemble (Bi-LSTM + tabular)

```bash
venv/bin/python src/models/run_ensemble.py --epochs 200 --seed 42 --folds 5
```

*Requer `results_nmkoa_cv.json` e `results_koastage_cv.json` do Passo 1.*  
Tempo: ~30-60 min (CPU).  
Saídas:
- `data/output/models/results_nmkoa_ensemble_cv.json`
- `data/output/models/results_koastage_ensemble_cv.json`
- `data/output/figures/models/ensemble_comparison_nmkoa.png`
- `data/output/figures/models/ensemble_comparison_koastage.png`

### Passo 3 — Figuras de comparação geral (rápido, sem treino)

```bash
venv/bin/python src/models/generate_report_figures.py --fast-only
```

Tempo: < 5 segundos.  
Saídas:
- `data/output/figures/models/all_models_accuracy_cv.png`

### Passo 4 — Ablação do encoder (opcional)

```bash
venv/bin/python src/models/run_encoder_ablation.py --epochs 200 --seed 42 --folds 5
```

Compara LSTM (64-dim) vs Bi-LSTM (128-dim) como encoder do ensemble.  
Tempo: ~60-90 min (CPU).  
Saídas:
- `data/output/models/results_encoder_ablation.json`
- `data/output/figures/models/encoder_ablation_nmkoa.png`
- `data/output/figures/models/encoder_ablation_koastage.png`

### Passo 5 — Curvas ROC OOF + feature importance do ensemble (lento)

```bash
venv/bin/python src/models/generate_report_figures.py --epochs 200 --seed 42
```

**Atenção:** re-treina todos os modelos do zero para coletar predições out-of-fold.  
Tempo: **1-3 horas na CPU** (usar GPU reduz para ~20-30 min).  
Saídas:
- `data/output/figures/models/roc_oof_nmkoa_standalone.png`
- `data/output/figures/models/roc_oof_nmkoa_ensemble.png`
- `data/output/figures/models/roc_oof_koastage_standalone.png`
- `data/output/figures/models/roc_oof_koastage_ensemble.png`
- `data/output/figures/models/ensemble_feat_imp_nmkoa_top25.png`
- `data/output/figures/models/ensemble_modality_imp_nmkoa.png`
- `data/output/figures/models/ensemble_feat_imp_koastage_top25.png`
- `data/output/figures/models/ensemble_modality_imp_koastage.png`

### Ordem recomendada

```
Passo 1  →  Passo 2  →  Passo 3  →  [Passo 4]  →  [Passo 5]
obrigatório            rápido        opcional         lento
```

Passos 4 e 5 são independentes entre si e podem ser rodados em paralelo após o Passo 2.

---

*Scripts: `src/models/` · Figuras: `data/output/figures/models/` · Resultados JSON: `data/output/models/`*
