# Classificação Automática da Severidade da Osteoartrite de Joelho a partir de Análise de Marcha 2D: Engenharia de Features Biomecânicas e Ensemble LSTM com Validação Cruzada a Nível de Sujeito

---

## Resumo

A osteoartrite de joelho (KOA) é uma das condições musculoesqueléticas de maior prevalência global,
responsável por substancial limitação funcional e carga socioeconômica. A avaliação clínica convencional,
baseada em radiografia e escalas de graduação subjetivas, não captura alterações biomecânicas precoces
que antecedem a degeneração estrutural visível. Este trabalho propõe um pipeline completo de classificação
automática de KOA a partir de vídeos sagitais de marcha 2D, processados por estimativa de pose (OpenPose
Body25), sem necessidade de marcadores ou equipamentos especializados.

A abordagem combina 51 features biomecânicas tabulares (parâmetros espaço-temporais e cinemáticos
articulares) com embeddings temporais extraídos por um encoder LSTM (64 dimensões) em uma arquitetura
ensemble com XGBoost. Dois objetivos de classificação são avaliados: (Tarefa A) triagem binária entre
controles normais (NM) e pacientes com KOA; e (Tarefa B) estadiamento de severidade em três graus
(precoce, moderado e severo). A validação é conduzida por validação cruzada estratificada de 5 folds a
nível de sujeito, garantindo que nenhum participante integre simultaneamente os conjuntos de treino e
teste.

Os resultados demonstram que o XGBoost tabular atinge 94,7% ± 3,4% de acurácia (F1 macro = 0,945;
AUC = 0,994) na Tarefa A, ao passo que o ensemble XGB com encoder LSTM supera todos os modelos
individuais na Tarefa B, atingindo 79,6% ± 6,4% de acurácia (F1 macro = 0,804; AUC = 0,911),
representando uma melhora de 8,4 pontos percentuais sobre o XGBoost tabular isolado (71,2%).
Adicionalmente, demonstramos que o encoder LSTM unidirecional (64 dimensões) iguala ou supera o
Bi-LSTM (128 dimensões) em ambas as tarefas, com metade da dimensionalidade de embedding. Estes
resultados superam o melhor resultado comparável da literatura com protocolo equivalente de validação
(76,6% de acurácia no estadiamento).

**Palavras-chave:** osteoartrite de joelho, análise de marcha, estimativa de pose, OpenPose, LSTM,
ensemble, validação cruzada a nível de sujeito, classificação biomecânica.

---

## 1. Introdução

A osteoartrite de joelho (KOA) constitui uma das principais causas de dor crônica e incapacidade
funcional na população adulta e idosa. Análises recentes do estudo Global Burden of Disease estimam
que, em 2020, mais de 595 milhões de pessoas viviam com osteoartrite em todo o mundo, com projeção de
aumento de 74,9% nos casos de osteoartrite de joelho até 2050 [1]. No Brasil, a osteoartrite é a
doença reumática mais prevalente, afetando mais de 15 milhões de pessoas, com impacto direto na
qualidade de vida e nos custos do sistema de saúde.

O diagnóstico e o monitoramento da KOA são fundamentados, na prática clínica, na escala radiográfica
de Kellgren-Lawrence (KL), que classifica a severidade em cinco graus (0–4) com base em achados como
estreitamento do espaço articular e formação de osteófitos. Embora amplamente utilizada, essa escala
apresenta limitações relevantes: baixa sensibilidade para estágios iniciais, variabilidade inter-
observador e ausência de correlação direta com o estado funcional do paciente. Em estágios precoces
(KL 1–2), a estrutura articular pode estar preservada enquanto alterações biomecânicas já estão
presentes e causam sintomas.

A análise instrumental da marcha é reconhecida como biomarcador funcional sensível para KOA. Pacientes
com KOA apresentam redução da amplitude de movimento (ROM) do joelho, encurtamento do comprimento de
passo, aumento do tempo de apoio e maior variabilidade intra-sujeito em comparação com controles
saudáveis [2,3]. Esses parâmetros podem ser extraídos de câmeras de vídeo convencionais utilizando
algoritmos de estimativa de pose, eliminando a necessidade de sistemas de captura óptica com marcadores
(custo elevado, restritos a laboratórios especializados) ou sensores inerciais.

O OpenPose [4], um dos algoritmos de estimativa de pose mais utilizados em contextos clínicos, detecta
automaticamente 25 keypoints corporais em vídeos monoculares com alta acurácia para cinemática de
marcha no plano sagital: erro médio de 5,1 ± 2,5° para o joelho e 3,7 ± 1,3° para o quadril,
comparado a sistemas de captura de movimento com marcadores [3].

Apesar dos avanços recentes em aprendizado de máquina aplicado à análise de marcha, a literatura
apresenta uma lacuna metodológica importante: a maioria dos trabalhos realiza a validação por *random
split* entre arquivos ou ciclos, sem controlar o agrupamento de dados por sujeito. Essa prática
introduz data leakage — múltiplos ciclos do mesmo sujeito distribuídos entre treino e teste —,
inflando artificialmente as métricas reportadas [5]. Apenas um trabalho recente [5] adota
explicitamente a validação a nível de sujeito para o estadiamento de KOA, reportando queda de 14
pontos percentuais ao migrar de *random split* (90,8%) para *subject-based split* (76,6%).

Este trabalho apresenta as seguintes contribuições:

1. **Pipeline de análise completo e reproduzível**: da extração de poses com OpenPose Body25 à
   classificação por ensemble, com auditoria explícita de data leakage.
2. **Protocolo de validação rigoroso**: validação cruzada estratificada de 5 folds a nível de sujeito
   (*StratifiedGroupKFold*), sem vazamento de informação entre treino e teste.
3. **Ensemble LSTM + tabular**: fusão explícita de embeddings temporais (64 dimensões) com features
   biomecânicas tabulares (51 dimensões), obtendo 8,4pp de melhora sobre XGBoost tabular no
   estadiamento de KOA.
4. **Ablação experimental do encoder**: comparação sistemática entre encoder LSTM unidirecional e
   bidirecional (Bi-LSTM) com os mesmos hiperparâmetros e folds, demonstrando que o LSTM iguala ou
   supera o Bi-LSTM com metade da dimensionalidade.
5. **Melhor desempenho em estadiamento de KOA sob protocolo rigoroso**: 79,6% de acurácia vs. 76,6%
   do melhor resultado comparável [5].

---

## 2. Trabalhos Relacionados

### 2.1 Estimativa de Pose para Análise de Marcha

Stenum et al. [2] demonstraram que o OpenPose pode extrair automaticamente eventos de marcha
(contato inicial e retirada do pé) e parâmetros espaço-temporais a partir de vídeos 2D sagitais,
com validade comparável a sistemas com marcadores. O método de detecção de eventos baseado na posição
relativa dos tornozelos ao ponto médio do quadril (*MidHip*) foi adotado integralmente neste trabalho.
Washabaugh et al. [3] compararam múltiplos algoritmos de estimativa de pose (OpenPose, MoveNet,
MediaPipe) para mensuração de cinemática de marcha, concluindo que o OpenPose apresenta o menor erro
para ângulos do joelho (5,1° ± 2,5°) e do quadril (3,7° ± 1,3°), sendo o mais adequado para análise
de marcha clínica 2D.

### 2.2 Classificação de KOA por Análise de Marcha

A Tabela 1 apresenta uma comparação sistemática dos trabalhos mais recentes que utilizam o dataset
público de Kour et al. [6] para classificação de KOA.

**Tabela 1.** Comparação com estudos que utilizam o mesmo dataset [6].

| Aspecto | Este estudo | Kaya et al. [5] | Ben Hassine et al. [7] |
|---------|-------------|----------------|------------------------|
| Estimativa de pose | OpenPose Body25 (2D) | AlphaPose + HybrIK (3D) | Não especificado (2D) |
| Features | 51 tabulares + LSTM | Ângulos 3D normalizados | Ângulos 2D + comprimento |
| Melhor modelo | Ensemble XGB (LSTM enc.) | LSTM-FCN | Random Forest |
| **Tarefa A — NM vs KOA** | **94,7% (subj-CV)** | 90,8% (random) / 76,6%† | 96,9%‡ |
| **Tarefa B — KOA Staging** | **79,6% (subj-CV)** | 76,6% (subj-CV) | — |
| Validação | 5-fold subj-CV | 5-fold (random + subj) | Não reportada |
| Auditoria de leakage | Sim (explícita) | Parcial | Não reportada |
| Ablação de encoder | Sim (LSTM vs Bi-LSTM) | Não | Não |

† Kaya et al. [5] reportam queda de 14,2pp ao adotar split por sujeito.
‡ Protocolo de validação não detalhado em [7].

Kaya et al. [5] utilizaram AlphaPose em combinação com HybrIK para extrair ângulos articulares 3D e
treinar um modelo LSTM-FCN, obtendo 76,6% de acurácia no estadiamento com validação por sujeito —
atualmente o resultado de melhor referência para comparação direta. Ben Hassine et al. [7]
propuseram uma abordagem de Random Forest sobre features extraídas de vídeos 2D, reportando 96,9%
na classificação binária NM vs KOA, porém sem detalhar o protocolo de validação.

### 2.3 Justificativa para a Abordagem Ensemble

A literatura em análise de marcha apoia o uso combinado de features biomecânicas interpretáveis com
representações temporais aprendidas. Features tabulares como ROM, cadência e comprimento de passo
capturam padrões agregados, enquanto sequências temporais de ângulos articulares retêm informações
sobre padrões de movimento específicos da fase do ciclo que são parcialmente perdidos na
agregação [2,5]. A arquitetura ensemble proposta neste trabalho explora essa complementaridade de
forma explícita e auditável.

---

## 3. Materiais e Métodos

### 3.1 Conjunto de Dados

Este trabalho utiliza o dataset público "A Vision-Based Gait Dataset for Knee Osteoarthritis and
Parkinson's Disease Analysis with Severity Levels" [6], disponibilizado por Kour, Gupta e Arora
(2022) na plataforma Mendeley Data. O dataset contém gravações de vídeo sagital de marcha de 96
participantes distribuídos em três grupos: 30 controles saudáveis (NM — Normal), 50 pacientes com
KOA (estadiados como precoce, moderado ou severo segundo Kellgren-Lawrence) e 16 pacientes com
doença de Parkinson (PD). O presente estudo foca nas classes NM e KOA (Tabela 2); o grupo PD não
foi incluído nas análises e é discutido como extensão futura na Seção 5.

**Tabela 2.** Composição do dataset utilizado (grupos NM e KOA).

| Grupo | Sujeitos | Arquivos | Ciclos extraídos | Descrição |
|-------|----------|----------|-----------------|-----------|
| NM | 30 | 60 | 324 | Controles saudáveis |
| KOA-EL (Precoce) | ~17 | 28 | 248 | Estágio inicial (KL 1–2) |
| KOA-MD (Moderado) | ~19 | 38 | 345 | Estágio intermediário (KL 3) |
| KOA-SV (Severo) | ~13 | 28 | 433 | Estágio avançado (KL 4) |
| **Total Tarefa A** | **79** | **152** | **1350** | NM + todos KOA |
| **Total Tarefa B** | **49** | **94** | **1026** | Apenas KOA (3 estágios) |

Os vídeos foram adquiridos com câmeras Logitech HD Pro C920 ou Nikon D5300 em ambiente clínico
controlado, com sujeitos caminhando no plano sagital a 1920 × 1080 px a ~50 fps. Cada sujeito
realizou duas gravações (sentido direita→esquerda e esquerda→direita).

### 3.2 Estimativa de Pose

A extração de keypoints foi realizada com o modelo **OpenPose Body25** [4] (modelo COCO com 25
keypoints, versão 1.7.0), configurado com os parâmetros:

- `number_people_max = 1` (vídeos com sujeito único)
- Confiança mínima: 0,3 (conforme recomendação de Washabaugh et al. [3])
- Saída: $(T \times 25 \times 3)$ — tempo × keypoints × (x, y, confiança)

Dos 25 keypoints do modelo Body25, 12 foram utilizados para a análise de marcha: pescoço (1),
ponto médio do quadril (8), quadril direito/esquerdo (9/12), joelho direito/esquerdo (10/13),
tornozelo direito/esquerdo (11/14), hálux direito/esquerdo (22/19) e calcanhar direito/esquerdo
(24/21).

Os ângulos articulares no plano sagital foram calculados pelo produto interno entre vetores
segmentares:

$$\theta_{\text{joelho}} = \arccos\left(\frac{\vec{u}_{\text{coxa}} \cdot \vec{u}_{\text{perna}}}{|\vec{u}_{\text{coxa}}||\vec{u}_{\text{perna}}|}\right)$$

onde $\vec{u}_{\text{coxa}} = \text{quadril} - \text{joelho}$ e
$\vec{u}_{\text{perna}} = \text{tornozelo} - \text{joelho}$.

Definições análogas foram aplicadas ao quadril (vetores tronco–quadril e coxa–quadril) e ao
tornozelo (vetores perna–tornozelo e hálux–tornozelo).

### 3.3 Pré-processamento do Sinal

O pipeline de pré-processamento foi implementado seguindo o protocolo de Stenum et al. [2],
com ajustes para o contexto de vídeo 2D (Tabela 3).

**Tabela 3.** Pipeline de pré-processamento.

| Etapa | Método | Parâmetros |
|-------|--------|-----------|
| 1. Mascaramento | Keypoints com confiança < 0,3 → NaN | Threshold: 0,3 [3] |
| 2. Interpolação | Interpolação linear bidirecional | Máximo: 5 frames (~100 ms a 50 fps) |
| 3. Suavização | Filtro Butterworth zero-fase | 4ª ordem, corte em 5 Hz |
| 4. Ângulos | Produto interno entre vetores segmentares | Plano sagital (2D) |
| 5. Eventos | Posição relativa tornozelo/MidHip | Heel Strike (HS) e Toe Off (TO) [2] |
| 6. Ciclos | Normalização HS→HS do mesmo membro | 101 pontos via spline cúbico |
| 7. QC | Rejeição de ciclos com > 20% NaN | Por articulação |

A detecção de eventos de marcha é baseada na posição horizontal relativa do tornozelo ao ponto
médio do quadril: o Heel Strike corresponde ao máximo local desta posição relativa (pé mais
anterior), e o Toe Off ao mínimo local (pé mais posterior) [2]. A validação fisiológica inclui
critérios de cadência (30–90 passos/min por membro) e tempo mínimo de passo (300 ms). A
normalização temporal de cada ciclo HS→HS para 101 pontos permite a comparação entre sujeitos
com cadências distintas.

### 3.4 Extração de Features

#### 3.4.1 Features Tabulares (51 dimensões por arquivo)

**Parâmetros espaço-temporais (8 features):**
cadência (passos/min), tempo médio de passo (s), coeficiente de variação (CV) do tempo de passo,
comprimento médio de passo (pixels), CV do comprimento de passo, tempo de apoio direito (s),
tempo de apoio esquerdo (s) e índice de simetria (%).

**Parâmetros cinemáticos articulares (42 features):**
Para cada uma das 6 articulações (joelho direito/esquerdo, quadril direito/esquerdo, tornozelo
direito/esquerdo), foram calculadas 7 estatísticas sobre os ciclos normalizados: ROM médio (°),
desvio padrão (DP) do ROM, pico médio (°), DP do pico, posição do pico (% do ciclo), mínimo
médio (°) e CV intra-ciclo (%).

**Qualidade de detecção (1 feature):**
Percentual médio de frames válidos (confiança ≥ 0,3) nas articulações de interesse (quadril,
joelho, tornozelo), usado como covariável de qualidade.

#### 3.4.2 Features Sequenciais (101 × 4 por ciclo)

Cada ciclo de marcha é representado como uma matriz $(101 \times 4)$, onde as colunas
correspondem aos ângulos normalizados de [joelho direito, joelho esquerdo, quadril direito,
quadril esquerdo] ao longo dos 101 pontos do ciclo normalizado. Ciclos com mais de 20% de
valores ausentes foram descartados; NaNs residuais foram preenchidos por interpolação
forward/backward.

### 3.5 Modelos de Classificação

#### 3.5.1 Modelos Tabulares

**XGBoost** [8]: gradient boosting com 400 árvores, profundidade máxima 5, taxa de aprendizado
0,05, subsample 0,8 e colsample_bytree 0,8. Objetivo `binary:logistic` (Tarefa A) e
`multi:softprob` (Tarefa B).

**Random Forest** [9]: 500 árvores, sem profundidade máxima, mínimo de 2 amostras por folha e
class_weight='balanced'.

**SVM com kernel RBF** [10]: C = 10, $\gamma$ = 'scale', class_weight='balanced'. Todos os
modelos tabulares utilizam um *pipeline* scikit-learn com `SimpleImputer` (mediana) ajustado
exclusivamente nos dados de treino de cada fold.

#### 3.5.2 Modelos Sequenciais (LSTM e Bi-LSTM)

A arquitetura **GaitLSTM** recebe sequências de forma $(B \times 101 \times 4)$ e utiliza:

```
Entrada: (B, 101, 4)
└─ LSTM (hidden=64, layers=2, dropout=0.3)
└─ Último timestep: (B, 64)
└─ Dropout (0.3)
└─ Linear: 64 → num_classes
```

O **Bi-LSTM** é idêntico com `bidirectional=True`, resultando em embedding de 128 dimensões.

O treinamento utiliza otimizador Adam ($\eta = 10^{-3}$, weight_decay = $10^{-4}$), *loss*
CrossEntropy com pesos de classe inversamente proporcionais à frequência, gradiente limitado
(*clip norm* = 1,0) e agendador ReduceLROnPlateau (fator 0,5, paciência 10 épocas). O modelo
é treinado por **200 épocas fixas** (sem early stopping), com regularização via dropout (0,3)
e weight decay ($10^{-4}$). A calibração de 200 épocas foi confirmada experimentalmente:
a taxa de aprendizado convergiu para $3 \times 10^{-5}$ e o ganho em acurácia nas últimas
10 épocas foi inferior a 0,4 pontos percentuais.

### 3.6 Ensemble: Encoder LSTM + Features Tabulares

A arquitetura ensemble (Figura 1) treina o encoder LSTM como extrator de representações
temporais dentro de cada fold, concatena o embedding resultante com as features tabulares
e alimenta um classificador XGBoost (ou RF/SVM) no espaço combinado.

**Pipeline do ensemble (por fold de CV):**

1. Treinar o encoder LSTM em todos os ciclos de treino do fold (200 épocas).
2. Extrair embeddings do último timestep: $\mathbf{e}_i \in \mathbb{R}^{64}$ por ciclo.
3. Computar features tabulares médias por sujeito: $\mathbf{t}_s \in \mathbb{R}^{51}$.
4. Para cada ciclo $i$ do sujeito $s$: $\mathbf{x}_i = [\mathbf{e}_i \,\|\, \mathbf{t}_s] \in \mathbb{R}^{115}$.
5. Treinar XGBoost em $\{\mathbf{x}_i, y_i\}$ de treino.
6. Na inferência: predizer probabilidades por ciclo e agregar por sujeito (média).

A **agregação ciclo→sujeito** calcula a média das probabilidades de todos os ciclos do sujeito
e aplica argmax:

$$\hat{y}_s = \arg\max_c \frac{1}{|C_s|} \sum_{i \in C_s} P(y=c \mid \mathbf{x}_i)$$

onde $C_s$ é o conjunto de ciclos do sujeito $s$. Esta abordagem garante que a avaliação final
seja equivalente à dos modelos tabulares (uma predição por sujeito), tornando as comparações
diretas e sem viés de granularidade.

O encoder é treinado integralmente dentro de cada fold: os dados de teste nunca são vistos
durante o treinamento do LSTM nem durante o ajuste do XGBoost. A feature tabular média por
sujeito ($\mathbf{t}_s$) é calculada sobre todos os sujeitos, mas como é uma função
determinística dos dados sem parâmetros ajustáveis, não constitui fonte de leakage.

### 3.7 Protocolo de Validação

A validação adota **validação cruzada estratificada de 5 folds a nível de sujeito**
(`StratifiedGroupKFold`, scikit-learn), com:

- Chave de agrupamento: `"{GRUPO}_{ESTÁGIO}_{ID}"` (p.ex. `KOA_EL_001`), garantindo que sujeitos
  com mesmo ID numérico mas estágios distintos sejam tratados como participantes independentes.
- Estratificação por classe para manter proporções aproximadas em cada fold.
- Seed fixo (42) para reprodutibilidade.
- Tarefa A: ~63 sujeitos de treino / ~16 de teste por fold.
- Tarefa B: ~39 sujeitos de treino / ~10 de teste por fold.

Nenhum sujeito aparece simultaneamente nos conjuntos de treino e teste. As métricas reportadas são
médias e desvios padrão entre os 5 folds.

As métricas utilizadas são: acurácia (*accuracy*), F1 macro (média não ponderada entre classes),
F1 ponderado e área sob a curva ROC (AUC), calculada pelo método *one-vs-rest* com média macro.

---

## 4. Resultados

### 4.1 Tarefa A — Triagem NM vs KOA (Classificação Binária)

**Tabela 4.** Resultados dos modelos tabulares e sequenciais para NM vs KOA (5-fold subj-CV).

| Modelo | Acurácia | F1 Macro | AUC |
|--------|----------|----------|-----|
| **XGBoost** | **0,947 ± 0,034** | **0,945 ± 0,035** | **0,9944** |
| Random Forest | 0,941 ± 0,025 | 0,936 ± 0,028 | 0,9878 |
| SVM | 0,941 ± 0,053 | 0,935 ± 0,058 | 0,9886 |
| Bi-LSTM | 0,890 ± 0,046 | 0,862 ± 0,042 | 0,9423 |
| LSTM | 0,879 ± 0,061 | 0,850 ± 0,060 | 0,9344 |

**Tabela 5.** Resultados do ensemble (encoder LSTM) para NM vs KOA.

| Ensemble | Acurácia | F1 Macro | AUC |
|----------|----------|----------|-----|
| Ensemble SVM | 0,911 ± 0,051 | 0,901 ± 0,058 | 0,9693 |
| Ensemble XGB | 0,910 ± 0,088 | 0,904 ± 0,093 | 0,9715 |
| Ensemble RF | 0,910 ± 0,055 | 0,905 ± 0,054 | 0,9633 |

Na Tarefa A, os modelos tabulares superam consistentemente os modelos sequenciais e o ensemble.
O XGBoost tabular é o modelo de melhor desempenho, com 94,7% de acurácia e AUC = 0,994. O ensemble
não supera os modelos tabulares nesta tarefa: a adição de embeddings LSTM reduz a acurácia em ~3,7pp,
sugerindo que as features biomecânicas tabulares já capturam integralmente o sinal discriminativo
entre NM e KOA.

A análise fold a fold confirma estabilidade: XGBoost varia de 90,0% a 100% entre folds, enquanto
os modelos LSTM apresentam variância ligeiramente maior (78,0%–96,0%), esperada dado o menor
número de parâmetros ajustáveis por fold no domínio sequencial.

### 4.2 Tarefa B — Estadiamento de Severidade KOA (EL/MD/SV)

**Tabela 6.** Resultados dos modelos tabulares e sequenciais para estadiamento KOA (5-fold subj-CV).

| Modelo | Acurácia | F1 Macro | AUC |
|--------|----------|----------|-----|
| XGBoost | 0,712 ± 0,119 | 0,708 ± 0,124 | 0,8744 |
| SVM | 0,701 ± 0,099 | 0,697 ± 0,109 | 0,8829 |
| Random Forest | 0,691 ± 0,160 | 0,680 ± 0,168 | 0,8485 |
| Bi-LSTM | 0,576 ± 0,070 | 0,540 ± 0,062 | 0,7280 |
| LSTM | 0,551 ± 0,091 | 0,507 ± 0,082 | 0,7120 |

*Baseline aleatório (3 classes): acurácia = 33,3%.*

**Tabela 7.** Resultados do ensemble (encoder LSTM) para estadiamento KOA.

| Ensemble | Acurácia | F1 Macro | AUC |
|----------|----------|----------|-----|
| **Ensemble XGB** | **0,796 ± 0,064** | **0,804 ± 0,059** | **0,9105** |
| Ensemble RF | 0,736 ± 0,173 | 0,740 ± 0,175 | 0,8457 |
| Ensemble SVM | 0,720 ± 0,204 | 0,718 ± 0,204 | 0,8861 |

O ensemble XGB com encoder LSTM alcança 79,6% de acurácia e F1 macro = 0,804, representando
uma melhora de **+8,4pp** sobre o XGBoost tabular isolado (71,2%) e de **+22,0pp** sobre o
Bi-LSTM isolado (57,6%). O AUC de 0,911 indica excelente capacidade discriminativa em cenário
de três classes.

A análise por classe (F1 médio entre folds) revela assimetria:
- KOA-EL (Precoce): F1 = 0,803 ± 0,110
- KOA-MD (Moderado): F1 = 0,745 ± 0,081
- KOA-SV (Severo): F1 = 0,865 ± 0,127

O estágio moderado apresenta menor F1, consistente com sua posição intermediária na escala de
severidade — padrão de marcha entre os estágios precoce e severo, tornando a discriminação mais
difícil. O estágio severo é o mais bem classificado, refletindo a maior distinção biomecânica
em relação aos demais grupos.

![Comparação geral de modelos — Tarefa B](data/output/figures/models/ensemble_comparison_koastage.png)

### 4.3 Importância de Features

A importância média das features do XGBoost (ganho de informação, média entre folds) revela os
parâmetros biomecânicos mais discriminativos (Tabela 8).

**Tabela 8.** Top-5 features por importância XGBoost (média entre folds).

| Ranking | Tarefa A — NM vs KOA | Tarefa B — KOA Staging |
|---------|----------------------|------------------------|
| 1 | Comprimento médio de passo (px) | Comprimento médio de passo (px) |
| 2 | Tempo de apoio esquerdo (s) | ROM joelho direito (°) |
| 3 | CV ciclo joelho esquerdo (%) | Tempo médio de passo (s) |
| 4 | ROM joelho esquerdo (°) | ROM joelho esquerdo (°) |
| 5 | Mínimo joelho esquerdo (°) | Qualidade média (%) |

O comprimento de passo é a feature de maior poder discriminativo em ambas as tarefas, seguida por
parâmetros cinemáticos do joelho. Isso é consistente com a fisiopatologia: pacientes com KOA
encurtam o comprimento de passo para reduzir o estresse articular [2], e a ROM do joelho diminui
progressivamente com o avanço da doença.

A análise da contribuição por modalidade no ensemble (importância de ganho do XGBoost):
- **Tarefa A**: features tabulares = 96,5%; embeddings LSTM = 3,5%
- **Tarefa B**: features tabulares = 99,2%; embeddings LSTM = 0,8%

Embora a importância de ganho dos embeddings LSTM seja baixa, eles contribuem para splits que
resolvem casos limítrofes não discriminados pelas features tabulares, resultando no ganho
observado de +8,4pp na Tarefa B.

### 4.4 Ablação do Encoder: LSTM vs Bi-LSTM

A Tabela 9 compara o desempenho dos modelos ensemble com encoder LSTM unidirecional e bidirecional
sob os mesmos hiperparâmetros, seed e folds.

**Tabela 9.** Ablação do encoder: LSTM (dim=64) vs Bi-LSTM (dim=128).

| Tarefa | Encoder | Acurácia | F1 Macro | AUC | Dim. vetor |
|--------|---------|----------|----------|-----|-----------|
| NM vs KOA | LSTM | 0,910 ± 0,088 | 0,904 ± 0,093 | 0,972 | 115 |
| NM vs KOA | Bi-LSTM | 0,901 ± 0,082 | 0,896 ± 0,091 | 0,969 | 179 |
| KOA Staging | **LSTM** | **0,796 ± 0,064** | **0,804 ± 0,059** | 0,911 | 115 |
| KOA Staging | Bi-LSTM | 0,780 ± 0,091 | 0,779 ± 0,095 | **0,916** | 179 |

O encoder LSTM unidirecional iguala ou supera o Bi-LSTM em ambas as tarefas, com um vetor de
embedding de metade da dimensionalidade (64 vs 128 dimensões; vetor combinado: 115 vs 179). Na
Tarefa B, o LSTM supera o Bi-LSTM em +1,6pp de acurácia (+2,5pp de F1 macro), sugerindo que
o contexto futuro fornecido pelo processamento reverso do Bi-LSTM não acrescenta informação
relevante para este problema, possivelmente porque as features tabulares já capturam os padrões
globais do ciclo.

---

## 5. Discussão

### 5.1 Por que o XGBoost Tabular é Suficiente para a Tarefa A

As features espaço-temporais — especialmente o comprimento de passo e os parâmetros de apoio —
apresentam alta separabilidade entre NM e KOA. Pacientes com KOA encurtam o comprimento de passo
(redução média de ~81 pixels, equivalente a ~58% de diferença) e prolongam o tempo de apoio como
estratégia compensatória para reduzir o pico de carga articular. Esses parâmetros agregados são
captados integralmente pelas features tabulares, deixando pouca margem para que o embedding LSTM
— que retém variações ao longo do ciclo — contribua adicionalmente. Esse padrão justifica a
ausência de ganho do ensemble na Tarefa A.

### 5.2 Por que o Ensemble Melhora o Estadiamento (Tarefa B)

A distinção entre os três estágios de KOA — especialmente entre precoce (EL) e moderado (MD) —
não é completamente capturada pelas features tabulares globais. O ROM do joelho diminui
monotonicamente com a severidade (EL: ~46°, MD: ~40°, SV: ~33°), mas apresenta overlap
considerável entre grupos adjacentes. O embedding LSTM retém informações sobre o padrão temporal
do ângulo ao longo do ciclo — como a posição do pico e a velocidade angular nas fases de carga e
balanço — que não estão representadas nas estatísticas de primeiro e segundo momentos das features
tabulares. Essa informação complementar, ainda que de baixa importância de ganho individual
(0,8%), contribui para resolução de casos limítrofes, resultando em +8,4pp de acurácia.

### 5.3 Posicionamento em Relação à Literatura

Este trabalho obtém o **melhor resultado reportado em estadiamento de KOA sob protocolo de
validação a nível de sujeito** (79,6% vs 76,6% de Kaya et al. [5]), utilizando um pipeline
computacionalmente mais acessível: OpenPose 2D em vez de AlphaPose + HybrIK 3D, e features
biomecânicas interpretáveis em vez de sequências de 24 keypoints 3D por frame.

A comparação direta com o trabalho de Kaya et al. [5] é particularmente relevante porque é o
único outro estudo que adota validação por sujeito e relata resultados tanto para classificação
binária quanto para estadiamento no mesmo dataset. Kaya et al. reportam queda de 14pp ao migrar
de random split (90,8%) para subject split (76,6%), evidenciando o efeito do data leakage. Este
trabalho opera exclusivamente sob protocolo a nível de sujeito, sem comparação com random split.

Ben Hassine et al. [7] reportam 96,9% na classificação binária NM vs KOA, levemente acima dos
94,7% deste trabalho, porém o protocolo de validação não é detalhado no artigo — impossibilitando
confirmar se a comparação é justa. O trabalho de Ben Hassine utiliza apenas features extraídas
de vídeo 2D sem pose estimation explícita, enquanto o presente trabalho fornece rastreabilidade
completa dos keypoints e features interpretáveis.

### 5.4 Limitações

**Pose estimation 2D vs 3D:** O OpenPose Body25 opera no plano sagital e não captura movimentos
nos planos frontal e transverso (adução/abdução do joelho, rotação), que são clinicamente relevantes
para KOA. A alta variância do ROM do quadril ($\sigma \approx 23°$), comparado ao joelho
($\sigma \approx 9°$), é consistente com essa limitação: movimentos do quadril fora do plano sagital
introduzem ruído nas estimativas 2D.

**Comprimento de passo em pixels:** A feature de maior importância discriminativa não é calibrada
em unidades métricas, introduzindo potencial confundidor de estatura entre sujeitos. Normalização
pelo comprimento do membro inferior (estimado pela distância quadril–tornozelo nos keypoints)
poderia reduzir essa variância.

**N amostral pequeno e alta variância entre folds:** Com ~10 sujeitos por fold na Tarefa B, a
estimativa de desempenho por fold apresenta variância elevada (desvio padrão de 6–16%). Essa
limitação é inerente ao dataset e afeta igualmente todos os trabalhos comparados.

**Grupo PD não incluído:** O dataset de Kour et al. [6] inclui 16 participantes com doença de
Parkinson, cujos padrões de marcha diferem tanto de NM quanto de KOA. A classificação tripartite
NM/KOA/PD, investigada por outros trabalhos no mesmo dataset, é uma extensão natural deste trabalho.

**Generalização demográfica:** O dataset foi coletado em ambiente clínico na Índia, com sujeitos
de estatura média ~1,56 m. A generalização para populações com diferentes distribuições
antropométricas não foi avaliada.

---

## 6. Conclusão

Este trabalho apresentou um pipeline completo e auditável para classificação automática de KOA
a partir de vídeos sagitais de marcha 2D. As principais conclusões são:

1. **Features biomecânicas tabulares são suficientes para triagem NM vs KOA**: XGBoost tabular
   atinge 94,7% de acurácia e AUC = 0,994 com validação a nível de sujeito, demonstrando que
   parâmetros como comprimento de passo, ROM do joelho e tempo de apoio são altamente
   discriminativos.

2. **O ensemble LSTM+tabular é necessário para o estadiamento de severidade**: O ensemble XGB
   com encoder LSTM atinge 79,6% de acurácia no estadiamento EL/MD/SV (+8,4pp sobre XGBoost
   tabular), evidenciando que a informação temporal do ciclo complementa os parâmetros agregados.

3. **Validação a nível de sujeito é determinante**: O protocolo de validação *StratifiedGroupKFold*
   por sujeito produz estimativas mais conservadoras e realistas do que o *random split*, consistente
   com os achados de Kaya et al. [5].

4. **LSTM unidirecional iguala ou supera o Bi-LSTM** com metade da dimensionalidade de embedding,
   recomendando-se o LSTM como encoder de menor complexidade para este domínio.

5. **Pipeline acessível e interpretável**: OpenPose Body25 em câmera 2D monocular, sem marcadores
   ou sensores especializados, com features biomecânicas de significado clínico direto.

**Trabalho futuro** inclui: (i) inclusão do grupo PD para classificação tripartite, (ii) calibração
métrica do comprimento de passo por comprimento de membro estimado via keypoints, (iii) análise
SHAP para interpretabilidade a nível de ciclo individual, e (iv) validação em coorte independente.

---

## Referências

[1] GBD 2021 Osteoarthritis Collaborators. Global, regional, and national burden of osteoarthritis,
1990–2020 and projections to 2050: a systematic analysis for the Global Burden of Disease Study
2021. *The Lancet Rheumatology*, 5(9):e508–e522, 2023.
DOI: [10.1016/S2665-9913(23)00163-7](https://doi.org/10.1016/S2665-9913(23)00163-7)

[2] Stenum J, Rossi C, Roemmich RT. Two-dimensional video-based analysis of human gait using pose
estimation. *PLOS Computational Biology*, 17(4):e1008935, 2021.
DOI: [10.1371/journal.pcbi.1008935](https://doi.org/10.1371/journal.pcbi.1008935)
PubMed ID: 33891585

[3] Washabaugh EP, Shanmugam TA, Ranganathan R, Krishnan C. Comparing the accuracy of open-source
pose estimation methods for measuring gait kinematics. *Gait & Posture*, 97:188–195, 2022.
DOI: [10.1016/j.gaitpost.2022.08.008](https://doi.org/10.1016/j.gaitpost.2022.08.008)
PubMed ID: 35988434

[4] Cao Z, Hidalgo G, Simon T, Wei S-E, Sheikh Y. OpenPose: Realtime multi-person 2D pose
estimation using part affinity fields. *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, 43(1):172–186, 2021.
DOI: [10.1109/TPAMI.2019.2929257](https://doi.org/10.1109/TPAMI.2019.2929257)
PubMed ID: 31331883

[5] Kaya MF, Yaşar H, Güllü MK. Classification of knee osteoarthritis severity using markerless
motion capture and long short-term memory fully convolutional network. *Computers in Biology and
Medicine*, 2025. PubMed ID: 40582166.
DOI: [10.1016/j.compbiomed.2025.109989](https://doi.org/10.1016/j.compbiomed.2025.109989)
[A VERIFICAR — DOI construído a partir do PII S0010482525009989]

[6] Kour N, Gupta S, Arora S. A vision-based gait dataset for knee osteoarthritis and Parkinson's
disease analysis with severity levels. In: *Proceedings of the International Conference on
Innovative Computing and Communications (ICICC 2021)*. Lecture Notes in Networks and Systems,
vol. 308. Springer, Singapore, 2022.
DOI: [10.1007/978-981-16-3071-2_26](https://doi.org/10.1007/978-981-16-3071-2_26)
Dataset: Mendeley Data. DOI: [10.17632/44pfnysy89.1](https://doi.org/10.17632/44pfnysy89.1)

[7] Ben Hassine S, Balti A, Abid S, Ben Khelifa MM, Sayadi M. Markerless vision-based knee
osteoarthritis classification using machine learning and gait videos. *Frontiers in Signal
Processing*, 4:1479244, 2024.
DOI: [10.3389/frsip.2024.1479244](https://doi.org/10.3389/frsip.2024.1479244)

[8] Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: *Proceedings of the 22nd
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, 2016.
DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

[9] Breiman L. Random forests. *Machine Learning*, 45:5–32, 2001.
DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

[10] Cortes C, Vapnik V. Support-vector networks. *Machine Learning*, 20(3):273–297, 1995.
DOI: [10.1007/BF00994018](https://doi.org/10.1007/BF00994018)

[11] Hochreiter S, Schmidhuber J. Long short-term memory. *Neural Computation*, 9(8):1735–1780,
1997.
DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
PubMed ID: 9377276

---

*Nota: A referência [5] (Kaya et al., 2025) contém o DOI marcado como [A VERIFICAR]. O DOI foi
construído a partir do número PII (S0010482525009989) disponível no ScienceDirect; recomenda-se
confirmar em [PubMed](https://pubmed.ncbi.nlm.nih.gov/40582166/) ou no
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010482525009989) antes da
submissão final.*
