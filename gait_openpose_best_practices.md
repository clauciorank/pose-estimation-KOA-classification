# Melhores Práticas para Processamento de Sinal em Análise de Marcha com OpenPose
> **Documento de referência para implementação em Python**  
> Baseado em literatura científica revisada por pares (2021–2025)

---

## Sumário

1. [Contexto e Fundamentação](#1-contexto-e-fundamentação)
2. [Configuração de Captura de Vídeo](#2-configuração-de-captura-de-vídeo)
3. [Extração de Keypoints com OpenPose](#3-extração-de-keypoints-com-openpose)
4. [Limpeza e Qualidade do Sinal](#4-limpeza-e-qualidade-do-sinal)
5. [Filtragem e Suavização](#5-filtragem-e-suavização)
6. [Detecção de Eventos de Marcha](#6-detecção-de-eventos-de-marcha)
7. [Calibração Pixel → Metro](#7-calibração-pixel--metro)
8. [Cálculo de Ângulos Articulares](#8-cálculo-de-ângulos-articulares)
9. [Normalização do Ciclo de Marcha](#9-normalização-do-ciclo-de-marcha)
10. [Extração de Features](#10-extração-de-features)
11. [Validação e Métricas de Qualidade](#11-validação-e-métricas-de-qualidade)
12. [Pipeline Completo em Python](#12-pipeline-completo-em-python)
13. [Limitações Conhecidas e Como Mitigá-las](#13-limitações-conhecidas-e-como-mitigá-las)
14. [Referências Científicas](#14-referências-científicas)

---

## 1. Contexto e Fundamentação

### O que é e o que não é o OpenPose

O OpenPose (Cao et al., 2021) é um estimador de pose 2D em tempo real baseado em *Part Affinity Fields* (PAF). Ele detecta automaticamente até **25 keypoints** corporais (modelo BODY_25) em frames de vídeo RGB, gerando coordenadas em pixels e um score de confiança (0–1) por keypoint por frame.

**O que o OpenPose oferece:**
- Coordenadas 2D (x, y) em pixels de cada keypoint
- Score de confiança por keypoint
- Exportação em JSON por frame

**O que o OpenPose NÃO oferece:**
- Profundidade (sem câmeras calibradas estéreo ou RGB-D)
- Dados de força ou torque
- Precisão absoluta equivalente ao VICON (gold standard)

### Evidências de acurácia (literatura)

Washabaugh et al. (2022) compararam OpenPose, MoveNet e DeepLabCut contra captura de movimento 3D em 32 participantes saudáveis:

| Método         | Erro médio - Quadril | Erro médio - Joelho |
|----------------|----------------------|----------------------|
| **OpenPose**   | **3.7 ± 1.3°**       | **5.1 ± 2.5°**       |
| MoveNet Thunder| 4.6 ± 1.8°           | 6.3 ± 2.1°           |
| DeepLabCut     | 6.8 ± 1.6°           | 7.4 ± 2.0°           |

> **Conclusão:** OpenPose é o mais preciso entre os métodos open-source para cinemática de marcha sagital, e pode ser alternativa válida ao motion capture convencional quando este não está disponível.

Henry et al. (2024) compararam OpenPose com QGA (Quantitative Gait Analysis) em 20 crianças:
- Joelhos: diferença média de **0.54°** (p=0.361) — clinicamente irrelevante
- Quadris: diferença de **7–9°** — atenção necessária
- Tornozelos: diferença de **~7°** — limitação importante

**Implicação prática:** foque análises em joelhos e quadris; trate tornozelos com cautela extra.

---

## 2. Configuração de Captura de Vídeo

A qualidade do sinal final depende criticamente das condições de captura. Pequenas melhorias no setup reduzem drasticamente o erro posterior.

### Requisitos mínimos e recomendados

| Parâmetro         | Mínimo        | Recomendado       | Justificativa                              |
|-------------------|---------------|-------------------|--------------------------------------------|
| Frame rate        | 30 fps        | **60–120 fps**    | Nyquist: marcha tem componentes até ~10 Hz |
| Resolução         | 800×600       | **1080p ou mais** | Keypoints mais estáveis                    |
| Plano de filmagem | Sagital       | Sagital + Frontal | Parâmetros espaciais mais completos        |
| Distância         | 3–5 m         | **4–6 m**         | Equilíbrio entre resolução e FOV           |
| Altura da câmera  | 0.8–1.0 m     | **~1.3 m**        | Nível do quadril, minimiza distorção       |

### Configuração do ambiente

```
✓ Iluminação uniforme — evitar sombras fortes e contraluz
✓ Fundo limpo e contrastante com a roupa do sujeito
✓ Câmera perpendicular à direção de caminhada (≤ 5° de desvio)
✓ Marcar no chão o corredor de análise (referência para calibração)
✓ Sujeito caminhando com roupas ajustadas (não largas)
✓ Incluir objeto de comprimento conhecido no campo de visão para calibração
```

> **Referência:** Viswakumar et al. (2022) validaram o sistema OMGait com smartphone a 30 fps, obtendo erro médio < 9° em diferentes condições de iluminação e vestuário.

---

## 3. Extração de Keypoints com OpenPose

### Modelo BODY_25 — Índices dos Keypoints

```python
BODY_25_KEYPOINTS = {
    0:  "Nose",
    1:  "Neck",
    2:  "RShoulder",  3: "RElbow",   4: "RWrist",
    5:  "LShoulder",  6: "LElbow",   7: "LWrist",
    8:  "MidHip",
    9:  "RHip",       10: "RKnee",   11: "RAnkle",
    12: "LHip",       13: "LKnee",   14: "LAnkle",
    15: "REye",       16: "LEye",    17: "REar",    18: "LEar",
    19: "LBigToe",    20: "LSmallToe", 21: "LHeel",
    22: "RBigToe",    23: "RSmallToe", 24: "RHeel",
    # 25 keypoints no total
}

# Índices mais relevantes para análise de marcha
GAIT_KEYPOINTS = {
    "MidHip": 8,
    "RHip": 9,    "RKnee": 10,  "RAnkle": 11,
    "LHip": 12,   "LKnee": 13,  "LAnkle": 14,
    "LBigToe": 19, "LHeel": 21,
    "RBigToe": 22, "RHeel": 24,
    "Neck": 1,
}
```

### Parsing dos JSONs gerados pelo OpenPose

```python
import json
import numpy as np
from pathlib import Path

def load_openpose_json(json_dir: str) -> np.ndarray:
    """
    Carrega todos os frames JSON gerados pelo OpenPose.
    
    Retorna:
        keypoints: array shape (n_frames, 25, 3)
                   eixo 2 = [x_pixel, y_pixel, confidence]
    """
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*_keypoints.json"))
    
    all_keypoints = []
    
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        
        people = data.get("people", [])
        
        if len(people) == 0:
            # Frame sem detecção — array de zeros
            kp = np.zeros((25, 3))
        else:
            # Selecionar a pessoa com maior número de keypoints com confiança > 0
            best_person = max(
                people,
                key=lambda p: np.sum(np.array(p["pose_keypoints_2d"]).reshape(-1, 3)[:, 2] > 0.1)
            )
            kp = np.array(best_person["pose_keypoints_2d"]).reshape(25, 3)
        
        all_keypoints.append(kp)
    
    return np.array(all_keypoints)  # (n_frames, 25, 3)
```

---

## 4. Limpeza e Qualidade do Sinal

### Filtragem por confiança

O score de confiança do OpenPose é a primeira linha de defesa contra dados espúrios.

```python
def apply_confidence_mask(keypoints: np.ndarray, 
                           threshold: float = 0.3) -> np.ndarray:
    """
    Zera keypoints abaixo do limiar de confiança.
    
    Args:
        keypoints: (n_frames, 25, 3) — [x, y, conf]
        threshold: confiança mínima (0.3 é o padrão da literatura)
    
    Returns:
        keypoints com posições NaN onde confiança é insuficiente
    """
    kp = keypoints.copy().astype(float)
    low_conf_mask = kp[:, :, 2] < threshold
    kp[low_conf_mask, 0] = np.nan  # x → NaN
    kp[low_conf_mask, 1] = np.nan  # y → NaN
    return kp
```

### Interpolação de gaps

Stenum et al. (2021) estabeleceram que interpolação linear para gaps de até 2 frames (≈ 80 ms a 25 fps) é aceitável sem introdução de artefatos relevantes.

```python
import pandas as pd

def interpolate_gaps(keypoints: np.ndarray, 
                     max_gap_frames: int = 5) -> np.ndarray:
    """
    Preenche gaps (NaN) via interpolação linear.
    
    Args:
        max_gap_frames: gaps maiores que este valor NÃO são interpolados.
                        Regra prática: ~80–150 ms
                        30fps → 3–5 frames
                        60fps → 5–9 frames
    """
    kp = keypoints.copy()
    n_frames, n_kp, _ = kp.shape
    
    for kp_idx in range(n_kp):
        for coord in range(2):  # x, y
            series = pd.Series(kp[:, kp_idx, coord])
            series_interp = series.interpolate(
                method="linear",
                limit=max_gap_frames,
                limit_direction="both"
            )
            kp[:, kp_idx, coord] = series_interp.values
    
    return kp

def get_data_quality_report(keypoints: np.ndarray, 
                              threshold: float = 0.3) -> dict:
    """
    Gera relatório de qualidade dos dados por keypoint.
    """
    n_frames = keypoints.shape[0]
    report = {}
    
    for kp_idx in range(keypoints.shape[1]):
        conf = keypoints[:, kp_idx, 2]
        n_valid = np.sum(conf >= threshold)
        report[kp_idx] = {
            "valid_frames": int(n_valid),
            "total_frames": n_frames,
            "pct_valid": float(n_valid / n_frames * 100),
            "mean_confidence": float(np.mean(conf[conf > 0]))
        }
    
    return report
```

---

## 5. Filtragem e Suavização

### Escolha do filtro — evidências da literatura

A literatura convergiu para dois filtros principais:

| Filtro                  | Ordem | Cutoff    | Uso recomendado                      |
|-------------------------|-------|-----------|--------------------------------------|
| Butterworth zero-phase  | 4     | 5–7 Hz    | Padrão — Stenum (2021), Washabaugh (2022), Stenum (2024) |
| Savitzky-Golay          | 3     | window 11 | Quando picos precisam ser preservados|

> **Atenção crítica com 30 fps:** o Nyquist é 15 Hz. Um cutoff de 5 Hz é conservador e seguro. Em 60 fps (Nyquist = 30 Hz), pode-se usar até 10 Hz sem aliasing.

```python
from scipy.signal import butter, filtfilt, savgol_filter

def butterworth_lowpass(signal: np.ndarray, 
                         cutoff_hz: float = 5.0,
                         fps: float = 30.0,
                         order: int = 4) -> np.ndarray:
    """
    Filtro Butterworth passa-baixa zero-phase (filtfilt).
    
    Parâmetros validados na literatura:
    - Stenum et al. (2021): cutoff = 5 Hz, ordem 4
    - Stenum et al. (2024): cutoff = 7 Hz (populações clínicas)
    - Washabaugh et al. (2022): cutoff = 5 Hz
    
    IMPORTANTE: Usar filtfilt (zero-phase) para não introduzir
    atraso temporal que desalinharia os eventos de marcha.
    """
    nyquist = fps / 2.0
    
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"Cutoff ({cutoff_hz} Hz) >= Nyquist ({nyquist} Hz). "
            f"Reduza o cutoff ou aumente o fps."
        )
    
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    
    # Verificar se há dados suficientes para filtragem
    if len(signal) < 3 * (order + 1):
        return signal  # Sinal muito curto — não filtrar
    
    # Ignorar NaNs: filtrar apenas segmentos contínuos
    filtered = np.full_like(signal, np.nan, dtype=float)
    valid = ~np.isnan(signal)
    
    if np.sum(valid) > 3 * (order + 1):
        filtered[valid] = filtfilt(b, a, signal[valid])
    
    return filtered


def smooth_keypoints(keypoints: np.ndarray,
                      fps: float = 30.0,
                      cutoff_hz: float = 5.0) -> np.ndarray:
    """
    Aplica filtro Butterworth a todas as trajetórias de keypoints.
    
    Args:
        keypoints: (n_frames, 25, 3)
        fps: frames por segundo do vídeo
        cutoff_hz: frequência de corte
    
    Returns:
        keypoints suavizados, mesmo shape
    """
    smoothed = keypoints.copy()
    n_kp = keypoints.shape[1]
    
    for kp_idx in range(n_kp):
        for coord in range(2):  # x, y — não filtrar o canal de confiança
            smoothed[:, kp_idx, coord] = butterworth_lowpass(
                keypoints[:, kp_idx, coord], 
                cutoff_hz=cutoff_hz, 
                fps=fps
            )
    
    return smoothed


def savgol_smooth_keypoints(keypoints: np.ndarray,
                              window_length: int = 11,
                              polyorder: int = 3) -> np.ndarray:
    """
    Alternativa com Savitzky-Golay — melhor preservação de picos.
    Útil quando se quer analisar picos de velocidade angular.
    """
    smoothed = keypoints.copy()
    
    for kp_idx in range(keypoints.shape[1]):
        for coord in range(2):
            sig = keypoints[:, kp_idx, coord]
            if np.sum(~np.isnan(sig)) > window_length:
                # Preencher NaNs temporariamente com interpolação para o filtro
                series = pd.Series(sig).interpolate("linear")
                filtered = savgol_filter(series.values, window_length, polyorder)
                # Restaurar NaNs originais
                filtered[np.isnan(sig)] = np.nan
                smoothed[:, kp_idx, coord] = filtered
    
    return smoothed
```

---

## 6. Detecção de Eventos de Marcha

### Método validado — Trajetória relativa ao MidHip

Este é o método estabelecido por Stenum et al. (2021, 2024), validado contra VICON em populações saudáveis e clínicas (AVC, Parkinson):

- **Heel Strike (HS):** pico positivo da trajetória horizontal do tornozelo **relativo ao MidHip**
- **Toe Off (TO):** pico negativo da mesma trajetória

A relativização ao MidHip remove o componente de translação global do corpo, deixando apenas o movimento dos membros.

```python
from scipy.signal import find_peaks

# Índices BODY_25
MID_HIP = 8
R_ANKLE = 11
L_ANKLE = 14
R_TOE   = 22
L_TOE   = 19

def detect_gait_events(keypoints: np.ndarray,
                        fps: float = 30.0,
                        min_step_duration_s: float = 0.3) -> dict:
    """
    Detecta Heel Strike e Toe Off via trajetória relativa ao MidHip.
    
    Método: Stenum et al. (2021) — PLOS Comp. Biol.
    Validação: erro médio < 1 frame (40 ms a 25 fps) vs. VICON.
    
    Args:
        keypoints: (n_frames, 25, 3) — suavizados, coordenadas em pixels
        fps: frames por segundo
        min_step_duration_s: duração mínima de um passo (evitar falsos picos)
    
    Returns:
        dict com arrays de frame indices para HS e TO de cada perna
    """
    min_dist_frames = int(min_step_duration_s * fps)
    
    mid_hip_x = keypoints[:, MID_HIP, 0]
    
    events = {}
    
    for side, ankle_idx, toe_idx in [("R", R_ANKLE, R_TOE), 
                                      ("L", L_ANKLE, L_TOE)]:
        
        ankle_x = keypoints[:, ankle_idx, 0]
        
        # Trajetória relativa: posição do tornozelo em relação ao centro do quadril
        relative_x = ankle_x - mid_hip_x
        
        # Mascarar frames com NaN
        valid_mask = ~(np.isnan(relative_x) | np.isnan(mid_hip_x))
        
        if np.sum(valid_mask) < min_dist_frames * 2:
            events[f"{side}_HS"] = np.array([], dtype=int)
            events[f"{side}_TO"] = np.array([], dtype=int)
            continue
        
        # Heel Strike = máximo local (pé mais à frente)
        hs_frames, hs_props = find_peaks(
            relative_x,
            distance=min_dist_frames,
            prominence=np.nanstd(relative_x) * 0.5  # Evitar micro-picos
        )
        
        # Toe Off = mínimo local (pé mais atrás)
        to_frames, to_props = find_peaks(
            -relative_x,
            distance=min_dist_frames,
            prominence=np.nanstd(relative_x) * 0.5
        )
        
        events[f"{side}_HS"] = hs_frames
        events[f"{side}_TO"] = to_frames
    
    return events


def validate_gait_events(events: dict, fps: float = 30.0) -> dict:
    """
    Verifica consistência dos eventos detectados.
    Após cada HS deve haver um TO, e vice-versa.
    """
    validation = {}
    
    for side in ["R", "L"]:
        hs = events.get(f"{side}_HS", np.array([]))
        to = events.get(f"{side}_TO", np.array([]))
        
        if len(hs) == 0 or len(to) == 0:
            validation[side] = {"valid": False, "reason": "Sem eventos detectados"}
            continue
        
        # Calcular cadência aproximada
        if len(hs) >= 2:
            step_times_s = np.diff(hs) / fps
            cadence = 60 / np.mean(step_times_s)  # passos/min
        else:
            cadence = None
        
        # Verificar se cadência está na faixa fisiológica (60–180 passos/min)
        if cadence is not None and not (60 <= cadence <= 180):
            validation[side] = {
                "valid": False,
                "reason": f"Cadência fora da faixa fisiológica: {cadence:.1f} passos/min",
                "n_HS": len(hs),
                "n_TO": len(to),
                "cadence": cadence
            }
        else:
            validation[side] = {
                "valid": True,
                "n_HS": len(hs),
                "n_TO": len(to),
                "cadence": cadence
            }
    
    return validation
```

---

## 7. Calibração Pixel → Metro

Sem calibração, todos os dados espaciais são em pixels — sem significado clínico. Esta etapa é **obrigatória** para comprimento do passo, velocidade e outras medidas métricas.

```python
def compute_scale_factor(keypoints: np.ndarray,
                           known_length_m: float,
                           method: str = "height") -> float:
    """
    Calcula fator de escala pixels → metros.
    
    Métodos disponíveis:
    - "height": usa a distância pescoço–tornozelo como proxy da altura
    - "floor_marks": usa marcas no chão de comprimento conhecido
    
    Args:
        keypoints: (n_frames, 25, 3)
        known_length_m: comprimento real em metros
            - método "height": altura do sujeito em m
            - método "floor_marks": distância entre marcas no chão em m
        method: "height" ou "floor_marks"
    
    Returns:
        scale: metros por pixel (m/px)
    
    Nota: Stenum et al. (2021) utilizaram marcas no chão para calibração.
    Para uso clínico sem marcas, a altura do sujeito é alternativa razoável.
    """
    NECK    = 1
    R_ANKLE = 11
    L_ANKLE = 14
    
    if method == "height":
        neck_y    = keypoints[:, NECK, 1]
        r_ankle_y = keypoints[:, R_ANKLE, 1]
        l_ankle_y = keypoints[:, L_ANKLE, 1]
        
        # Usar frames com alta confiança
        neck_conf    = keypoints[:, NECK, 2]
        r_ankle_conf = keypoints[:, R_ANKLE, 2]
        
        valid_frames = (neck_conf > 0.5) & (r_ankle_conf > 0.5)
        
        if np.sum(valid_frames) < 10:
            raise ValueError("Confiança insuficiente para calibração por altura.")
        
        # Distância vertical pescoço–tornozelo (proxy da altura, ~92% da estatura)
        height_px = np.median(
            np.abs(neck_y[valid_frames] - r_ankle_y[valid_frames])
        )
        
        # Ajuste: pescoço-tornozelo ≈ 92% da estatura total
        full_height_px = height_px / 0.92
        
        scale = known_length_m / full_height_px  # m/px
        
    else:
        raise NotImplementedError(f"Método '{method}' não implementado.")
    
    return scale


def pixels_to_meters(keypoints: np.ndarray, scale: float) -> np.ndarray:
    """
    Converte coordenadas de pixels para metros.
    Aplica apenas aos canais x e y (não ao canal de confiança).
    """
    kp_meters = keypoints.copy()
    kp_meters[:, :, 0] *= scale  # x
    kp_meters[:, :, 1] *= scale  # y
    return kp_meters
```

---

## 8. Cálculo de Ângulos Articulares

### Ângulos no plano sagital

```python
def joint_angle_2d(p1: np.ndarray, 
                    vertex: np.ndarray, 
                    p2: np.ndarray) -> np.ndarray:
    """
    Calcula ângulo (em graus) no vértice formado por três keypoints.
    
    Args:
        p1, vertex, p2: arrays (n_frames, 2) com coordenadas [x, y]
    
    Returns:
        angles: array (n_frames,) em graus [0°, 180°]
    """
    v1 = p1 - vertex  # vetor vertex → p1
    v2 = p2 - vertex  # vetor vertex → p2
    
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    
    # Evitar divisão por zero
    eps = 1e-8
    v1_norm = v1 / (norm1 + eps)
    v2_norm = v2 / (norm2 + eps)
    
    dot = np.sum(v1_norm * v2_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)  # Estabilidade numérica
    
    angles = np.degrees(np.arccos(dot))
    
    # Frames com keypoints NaN → NaN
    nan_mask = (np.isnan(p1).any(axis=1) | 
                np.isnan(vertex).any(axis=1) | 
                np.isnan(p2).any(axis=1))
    angles[nan_mask] = np.nan
    
    return angles


def compute_all_joint_angles(keypoints: np.ndarray) -> dict:
    """
    Calcula todos os ângulos articulares relevantes para marcha.
    
    Retorna dicionário com arrays de ângulos por articulação.
    
    Nota de Henry et al. (2024): joelhos apresentam excelente concordância
    com QGA (diferença < 1°). Quadris e tornozelos requerem interpretação 
    mais cuidadosa.
    """
    kp = keypoints[:, :, :2]  # apenas x, y
    
    # Índices BODY_25
    R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
    L_HIP, L_KNEE, L_ANKLE = 12, 13, 14
    MID_HIP = 8
    NECK    = 1
    R_TOE, L_TOE = 22, 19
    
    angles = {}
    
    # --- Joelhos ---
    angles["R_knee"] = joint_angle_2d(kp[:, R_HIP], kp[:, R_KNEE], kp[:, R_ANKLE])
    angles["L_knee"] = joint_angle_2d(kp[:, L_HIP], kp[:, L_KNEE], kp[:, L_ANKLE])
    
    # --- Quadris (ângulo da coxa em relação ao tronco vertical) ---
    # Vetor tronco: MidHip → Neck (aproximação)
    trunk_vec = kp[:, NECK] - kp[:, MID_HIP]
    
    # Vetor coxa direita: RHip → RKnee
    r_thigh_vec = kp[:, R_KNEE] - kp[:, R_HIP]
    l_thigh_vec = kp[:, L_KNEE] - kp[:, L_HIP]
    
    # Ângulo entre vetor tronco e vetor coxa (ângulo do quadril)
    angles["R_hip"] = joint_angle_2d(kp[:, NECK], kp[:, R_HIP], kp[:, R_KNEE])
    angles["L_hip"] = joint_angle_2d(kp[:, NECK], kp[:, L_HIP], kp[:, L_KNEE])
    
    # --- Tornozelos (com cautela — maior erro no OpenPose) ---
    angles["R_ankle"] = joint_angle_2d(kp[:, R_KNEE], kp[:, R_ANKLE], kp[:, R_TOE])
    angles["L_ankle"] = joint_angle_2d(kp[:, L_KNEE], kp[:, L_ANKLE], kp[:, L_TOE])
    
    return angles
```

---

## 9. Normalização do Ciclo de Marcha

### Por que normalizar

Ciclos de marcha têm durações variáveis entre sujeitos e entre si. A normalização para 0–100% (101 pontos) permite:
- Comparação entre sujeitos de diferentes estaturas/velocidades
- Cálculo de médias e desvios-padrão entre ciclos
- Uso de referências normativas da literatura

```python
from scipy.interpolate import interp1d

def normalize_gait_cycle(signal: np.ndarray, 
                          n_points: int = 101) -> np.ndarray:
    """
    Normaliza um ciclo de marcha para 101 pontos (0–100%).
    
    Args:
        signal: array 1D com os dados do ciclo
        n_points: número de pontos normalizados (padrão: 101)
    
    Returns:
        normalized: array (101,)
    """
    if len(signal) < 3:
        return np.full(n_points, np.nan)
    
    x_orig = np.linspace(0, 100, len(signal))
    x_new  = np.linspace(0, 100, n_points)
    
    # Interpolação cúbica para suavidade; linear para dados esparsos
    valid = ~np.isnan(signal)
    
    if np.sum(valid) < 4:
        return np.full(n_points, np.nan)
    
    if np.sum(valid) == len(signal):
        f = interp1d(x_orig, signal, kind="cubic", fill_value="extrapolate")
    else:
        f = interp1d(x_orig[valid], signal[valid], kind="linear",
                     bounds_error=False, fill_value=np.nan)
    
    return f(x_new)


def extract_gait_cycles(signal: np.ndarray, 
                         heel_strikes: np.ndarray,
                         n_points: int = 101) -> np.ndarray:
    """
    Extrai e normaliza múltiplos ciclos de marcha.
    
    Args:
        signal: sinal contínuo (ângulo ou posição)
        heel_strikes: frames dos heel strikes consecutivos
        n_points: pontos por ciclo normalizado
    
    Returns:
        cycles: array (n_cycles, n_points)
    """
    cycles = []
    
    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end   = heel_strikes[i + 1]
        
        if end <= start:
            continue
        
        cycle = signal[start:end]
        normalized = normalize_gait_cycle(cycle, n_points)
        cycles.append(normalized)
    
    return np.array(cycles)
```

---

## 10. Extração de Features

### Features espaciotemporais

```python
def compute_spatiotemporal_params(events: dict,
                                   keypoints_m: np.ndarray,
                                   fps: float = 30.0) -> dict:
    """
    Calcula parâmetros espaciotemporais de marcha.
    
    Parâmetros baseados em Stenum et al. (2021, 2024).
    Coordenadas devem estar em metros (após calibração).
    """
    R_HS = events["R_HS"]
    L_HS = events["L_HS"]
    R_TO = events["R_TO"]
    L_TO = events["L_TO"]
    
    R_ANKLE = 11
    L_ANKLE = 14
    MID_HIP  = 8
    
    results = {}
    
    # --- Cadência ---
    all_hs = np.sort(np.concatenate([R_HS, L_HS]))
    if len(all_hs) >= 2:
        step_times = np.diff(all_hs) / fps
        results["cadence_steps_per_min"] = float(60 / np.mean(step_times))
        results["mean_step_time_s"]      = float(np.mean(step_times))
    
    # --- Comprimento do passo (Método 1: distância entre tornozelos no HS) ---
    step_lengths = []
    for hs_frame in R_HS:
        r_ankle_x = keypoints_m[hs_frame, R_ANKLE, 0]
        l_ankle_x = keypoints_m[hs_frame, L_ANKLE, 0]
        if not (np.isnan(r_ankle_x) or np.isnan(l_ankle_x)):
            step_lengths.append(abs(r_ankle_x - l_ankle_x))
    
    if step_lengths:
        results["mean_step_length_m"] = float(np.mean(step_lengths))
        results["step_length_cv"]     = float(np.std(step_lengths) / 
                                               np.mean(step_lengths) * 100)
    
    # --- Velocidade de marcha ---
    if "mean_step_length_m" in results and "mean_step_time_s" in results:
        results["gait_speed_m_s"] = float(
            results["mean_step_length_m"] / results["mean_step_time_s"]
        )
    
    # --- Tempo de apoio (Stance time) ---
    stance_times = []
    for hs_frame in R_HS:
        # Encontrar o TO direito mais próximo após este HS
        future_to = R_TO[R_TO > hs_frame]
        if len(future_to) > 0:
            stance_frames = future_to[0] - hs_frame
            stance_times.append(stance_frames / fps)
    
    if stance_times:
        results["mean_stance_time_s"] = float(np.mean(stance_times))
    
    # --- Índice de simetria ---
    if len(R_HS) >= 2 and len(L_HS) >= 2:
        r_cadence = float(60 / np.mean(np.diff(R_HS) / fps))
        l_cadence = float(60 / np.mean(np.diff(L_HS) / fps))
        # Symmetry Index: 0 = simetria perfeita
        results["symmetry_index"] = float(
            abs(r_cadence - l_cadence) / ((r_cadence + l_cadence) / 2) * 100
        )
    
    return results


def compute_kinematic_features(cycles: np.ndarray) -> dict:
    """
    Extrai features cinemáticas de um conjunto de ciclos normalizados.
    
    Args:
        cycles: (n_cycles, 101) — ângulos normalizados por ciclo
    
    Returns:
        dict com features estatísticas
    """
    if cycles.shape[0] == 0:
        return {}
    
    mean_curve   = np.nanmean(cycles, axis=0)
    std_curve    = np.nanstd(cycles, axis=0)
    
    features = {
        "mean_curve":     mean_curve,
        "std_curve":      std_curve,
        "ROM":            float(np.nanmax(mean_curve) - np.nanmin(mean_curve)),
        "peak_value":     float(np.nanmax(mean_curve)),
        "peak_timing_pct":float(np.nanargmax(mean_curve)),
        "min_value":      float(np.nanmin(mean_curve)),
        "min_timing_pct": float(np.nanargmin(mean_curve)),
        "mean_value":     float(np.nanmean(mean_curve)),
        "variability_cv": float(np.nanmean(std_curve) / abs(np.nanmean(mean_curve)) * 100),
        "n_cycles":       int(cycles.shape[0]),
    }
    
    return features
```

---

## 11. Validação e Métricas de Qualidade

```python
def compute_icc(measurements_A: np.ndarray, 
                measurements_B: np.ndarray,
                icc_type: str = "ICC2,1") -> dict:
    """
    Calcula Intraclass Correlation Coefficient (ICC).
    
    ICC > 0.90: excelente concordância (alvo para análise de marcha)
    ICC 0.75–0.90: boa
    ICC 0.50–0.75: moderada
    ICC < 0.50: fraca
    
    Implementação simplificada — para uso clínico, usar pingouin.intraclass_corr()
    """
    try:
        import pingouin as pg
        import pandas as pd
        
        n = len(measurements_A)
        df = pd.DataFrame({
            "judges": ["A"] * n + ["B"] * n,
            "targets": list(range(n)) * 2,
            "ratings": list(measurements_A) + list(measurements_B)
        })
        icc_result = pg.intraclass_corr(data=df, targets="targets", 
                                         raters="judges", ratings="ratings")
        icc_row = icc_result[icc_result["Type"] == icc_type].iloc[0]
        
        return {
            "ICC": float(icc_row["ICC"]),
            "CI_lower": float(icc_row["CI95%"][0]),
            "CI_upper": float(icc_row["CI95%"][1]),
            "p_value": float(icc_row["pval"])
        }
    except ImportError:
        # Fallback manual (ICC tipo 2,1 simplificado)
        n = len(measurements_A)
        grand_mean = np.mean([measurements_A, measurements_B])
        ss_between = n * np.var([np.mean(measurements_A), np.mean(measurements_B)])
        ss_within  = np.sum((measurements_A - measurements_B)**2) / (2 * n)
        icc = (ss_between - ss_within) / (ss_between + ss_within)
        return {"ICC": float(icc), "note": "Cálculo simplificado — instalar pingouin para resultado completo"}


def bland_altman_stats(method_A: np.ndarray, 
                        method_B: np.ndarray,
                        label: str = "") -> dict:
    """
    Estatísticas de Bland-Altman para comparação de métodos.
    Usado para validar OpenPose contra gold standard (quando disponível).
    """
    diff   = method_A - method_B
    mean   = (method_A + method_B) / 2
    
    bias = np.mean(diff)
    sd   = np.std(diff)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    
    return {
        "label": label,
        "bias_mean_diff": float(bias),
        "std_diff":       float(sd),
        "LoA_upper":      float(loa_upper),
        "LoA_lower":      float(loa_lower),
        "MAE":            float(np.mean(np.abs(diff)))
    }
```

---

## 12. Pipeline Completo em Python

```python
"""
Pipeline completo de análise de marcha com OpenPose.
Baseado em Stenum et al. (2021, 2024) e Washabaugh et al. (2022).
"""

import numpy as np
import json
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES — ajustar conforme setup experimental
# ============================================================
CONFIG = {
    "fps":                  30.0,    # fps do vídeo
    "confidence_threshold": 0.3,     # limiar de confiança dos keypoints
    "max_gap_frames":       5,       # máximo de frames interpolados
    "filter_cutoff_hz":     5.0,     # frequência de corte do filtro
    "filter_order":         4,       # ordem do Butterworth
    "min_step_duration_s":  0.3,     # duração mínima de um passo
    "subject_height_m":     1.70,    # altura do sujeito (para calibração)
    "n_cycle_points":       101,     # pontos por ciclo normalizado
}

def run_gait_analysis_pipeline(json_dir: str, config: dict = CONFIG) -> dict:
    """
    Pipeline completo: JSON → Features de marcha.
    
    Args:
        json_dir: diretório com os JSONs do OpenPose
        config: dicionário de configurações
    
    Returns:
        results: dicionário com todos os resultados
    """
    
    print("=" * 60)
    print("  PIPELINE DE ANÁLISE DE MARCHA — OpenPose")
    print("=" * 60)
    
    # PASSO 1: Carregar keypoints
    print("\n[1/8] Carregando keypoints...")
    kp = load_openpose_json(json_dir)
    print(f"      {kp.shape[0]} frames carregados.")
    
    # PASSO 2: Aplicar máscara de confiança
    print("[2/8] Filtrando por confiança...")
    kp = apply_confidence_mask(kp, threshold=config["confidence_threshold"])
    quality = get_data_quality_report(kp, threshold=config["confidence_threshold"])
    
    # Alertar sobre keypoints com baixa qualidade
    for kp_idx, q in quality.items():
        if q["pct_valid"] < 80:
            print(f"      ⚠️  Keypoint {kp_idx}: apenas {q['pct_valid']:.1f}% de frames válidos")
    
    # PASSO 3: Interpolar gaps
    print("[3/8] Interpolando gaps...")
    kp = interpolate_gaps(kp, max_gap_frames=config["max_gap_frames"])
    
    # PASSO 4: Suavizar com Butterworth
    print("[4/8] Suavizando (Butterworth zero-phase)...")
    kp_smooth = smooth_keypoints(
        kp,
        fps=config["fps"],
        cutoff_hz=config["filter_cutoff_hz"]
    )
    
    # PASSO 5: Detectar eventos de marcha
    print("[5/8] Detectando eventos de marcha...")
    events = detect_gait_events(kp_smooth, fps=config["fps"],
                                  min_step_duration_s=config["min_step_duration_s"])
    validation = validate_gait_events(events, fps=config["fps"])
    
    for side, v in validation.items():
        status = "✓" if v["valid"] else "✗"
        print(f"      {status} Lado {side}: {v.get('n_HS', 0)} HS, "
              f"{v.get('n_TO', 0)} TO, cadência ≈ {v.get('cadence', 0):.1f} p/min")
    
    # PASSO 6: Calibração pixel → metro
    print("[6/8] Calibrando escala...")
    try:
        scale = compute_scale_factor(
            kp_smooth,
            known_length_m=config["subject_height_m"],
            method="height"
        )
        kp_meters = pixels_to_meters(kp_smooth, scale)
        print(f"      Escala: {scale*1000:.2f} mm/pixel")
    except Exception as e:
        print(f"      ⚠️ Calibração falhou: {e}. Usando pixels.")
        kp_meters = kp_smooth
        scale = None
    
    # PASSO 7: Calcular ângulos articulares
    print("[7/8] Calculando ângulos articulares...")
    angles = compute_all_joint_angles(kp_smooth)  # ângulos não dependem de escala
    
    # Extrair ciclos normalizados para cada articulação
    angle_cycles = {}
    for joint_name, angle_signal in angles.items():
        side_prefix = joint_name[0]  # "R" ou "L"
        hs_key = f"{side_prefix}_HS"
        
        if hs_key in events and len(events[hs_key]) >= 2:
            cycles = extract_gait_cycles(
                angle_signal,
                events[hs_key],
                n_points=config["n_cycle_points"]
            )
            angle_cycles[joint_name] = cycles
    
    # PASSO 8: Extrair features
    print("[8/8] Extraindo features...")
    spatiotemporal = compute_spatiotemporal_params(events, kp_meters, fps=config["fps"])
    kinematic_features = {
        joint: compute_kinematic_features(cycles)
        for joint, cycles in angle_cycles.items()
    }
    
    results = {
        "config":           config,
        "n_frames":         kp.shape[0],
        "scale_m_per_px":   scale,
        "quality_report":   quality,
        "events":           {k: v.tolist() for k, v in events.items()},
        "event_validation": validation,
        "spatiotemporal":   spatiotemporal,
        "angle_cycles":     {k: v.tolist() for k, v in angle_cycles.items()},
        "kinematic_features": kinematic_features,
    }
    
    print("\n✓ Pipeline concluído com sucesso.")
    print(f"  Velocidade de marcha: {spatiotemporal.get('gait_speed_m_s', 'N/A'):.2f} m/s")
    print(f"  Cadência:             {spatiotemporal.get('cadence_steps_per_min', 'N/A'):.1f} passos/min")
    
    return results
```

---

## 13. Limitações Conhecidas e Como Mitigá-las

| Limitação | Impacto Clínico | Estratégia de Mitigação |
|-----------|----------------|-------------------------|
| **2D apenas** (câmera única) | Sem rotações fora do plano sagital | Câmera sagital estrita; aceitar como limitação no relatório |
| **Erro no quadril ~7–9°** (Henry 2024) | Subestimação de extensão no pré-balanço | Interpretar tendências, não valores absolutos; validar com referência local |
| **Erro no tornozelo ~7°** (Henry 2024) | Dorsiflexão/flexão plantar menos confiável | Evitar conclusões clínicas baseadas apenas no tornozelo |
| **Jitter de keypoints** | Artefatos em velocidade e aceleração | Butterworth 5 Hz + interpolação antes de diferenciar |
| **30 fps** | Imprecisão temporal em eventos (~33 ms/frame) | Gravar em 60+ fps; reportar resolução temporal |
| **Oclusão de membros** | Gaps no sinal | Interpolação + flag de qualidade; excluir ciclos com >20% de gaps |
| **Troca de lados** (L↔R) | Inversão de parâmetros de simetria | Verificar visualmente e corrigir com `keypoints[:, R, :] = keypoints[:, L, :]` |
| **Pessoa incorreta detectada** | Dados de outro indivíduo | Selecionar pessoa com maior número de keypoints válidos (implementado no parser) |

---

## 14. Referências Científicas

1. **Cao Z, Hidalgo G, Simon T, Wei SE, Sheikh Y.** OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. *IEEE Trans Pattern Anal Mach Intell.* 2021;43:172–186. doi:10.1109/TPAMI.2019.2929257

2. **Stenum J, Rossi C, Roemmich RT.** Two-dimensional video-based analysis of human gait using pose estimation. *PLOS Comput Biol.* 2021;17(4):e1008935. doi:10.1371/journal.pcbi.1008935
   > Referência metodológica principal: validação vs. VICON, método de detecção de eventos por trajetória relativa ao MidHip, filtro Butterworth 5 Hz.

3. **Stenum J, Cherry-Allen KM, Crenshaw JR, Roemmich RT.** Clinical gait analysis using video-based pose estimation: multiple perspectives, clinical populations, and measuring change. *PLOS Digital Health.* 2024;3(3):e0000467. doi:10.1371/journal.pdig.0000467
   > Extensão para populações clínicas (AVC, Parkinson); perspectivas frontal e sagital.

4. **Washabaugh EP, Shanmugam TA, Ranganathan R, Krishnan C.** Comparing the accuracy of open-source pose estimation methods for measuring gait kinematics. *Gait Posture.* 2022;97:188–195. doi:10.1016/j.gaitpost.2022.08.008
   > Comparação OpenPose vs. MoveNet vs. DeepLabCut; OpenPose superior para joelhos (5.1±2.5°) e quadris (3.7±1.3°).

5. **Henry R, Cordillet S, et al.** Comparison of the OpenPose system and the reference optoelectronic system for gait analysis of lower-limb angular parameters in children. *Orthop Traumatol Surg Res.* 2024. doi:10.1016/j.otsr.2024.104044
   > Validação clínica em crianças; joelhos: diferença <1°; quadris e tornozelos: limitações importantes.

6. **Viswakumar A, Rajagopalan V, Ray T, Gottipati P, Parimi C.** Development of a Robust, Simple, and Affordable Human Gait Analysis System Using Bottom-Up Pose Estimation With a Smartphone Camera. *Front Physiol.* 2022;12:784865. doi:10.3389/fphys.2021.784865
   > Validação em diferentes condições de iluminação e vestuário; erro <9° com smartphone 30fps.

7. **Molteni LE, Andreoni G.** Comparing the Accuracy of Markerless Motion Analysis and Optoelectronic System for Measuring Gait Kinematics of Lower Limb. *Bioengineering.* 2025;12:424. doi:10.3390/bioengineering12040424
   > ICC bom a excelente para parâmetros espaciotemporais; incluindo populações com hemiplegia e paraparesia.

8. **Ali MM, et al.** Human Pose Estimation for Clinical Analysis of Gait Pathologies. *Bioinform Biol Insights.* 2024;18. doi:10.1177/11779322241231108
   > Pipeline OpenPose + MeTRAbs para estimativa 3D; extração de CGFs (Cyclic Gait Features).

9. **Zhu et al.** Markerless gait analysis through a single camera and computer vision. *J Biomech.* 2024. doi:10.1016/j.jbiomech.2024.112200

10. **Madgwick SOH, Harrison AJL, Vaidyanathan R.** Estimation of IMU and MARG orientation using a gradient descent algorithm. *IEEE Int Symp Robot Sens Environ.* 2011.
    > Referência para fusão de sensores (se combinado com IMU).

---

*Documento gerado em abril de 2026. Verificar literatura mais recente para atualizações metodológicas.*
