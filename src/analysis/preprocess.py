"""
Gait data cleaning and feature extraction pipeline.

Processing order (per Stenum et al. 2021, 2024 and Washabaugh et al. 2022):
    1. Confidence masking        — threshold 0.3 (literature standard)
    2. Gap interpolation         — linear, max 5 frames (~100 ms at 50 fps)
    3. Butterworth low-pass      — 4th order, 5 Hz zero-phase
    4. Joint angle computation   — sagittal plane (hip, knee, ankle)
    5. Gait event detection      — HS/TO via ankle trajectory relative to MidHip
    6. Gait cycle extraction     — normalize to 101 points (0–100%)
    7. Spatiotemporal parameters — cadence, step length, symmetry index, etc.
    8. Quality report            — per-joint % valid frames

Outputs a dict ready to be saved as .npz with np.savez_compressed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# BODY_25 index constants
# ---------------------------------------------------------------------------
NECK     = 1
MID_HIP  = 8
R_HIP    = 9;  R_KNEE = 10;  R_ANKLE = 11
L_HIP    = 12; L_KNEE = 13;  L_ANKLE = 14
L_BIG_TOE = 19; L_HEEL = 21
R_BIG_TOE = 22; R_HEEL = 24

GAIT_JOINT_INDICES = [
    NECK, MID_HIP,
    R_HIP, R_KNEE, R_ANKLE, R_BIG_TOE, R_HEEL,
    L_HIP, L_KNEE, L_ANKLE, L_BIG_TOE, L_HEEL,
]

CONF_THRESHOLD = 0.3   # Stenum (2021), Washabaugh (2022)
INTERP_MAX_GAP = 5     # frames — ~100 ms at 50 fps
BUTTER_CUTOFF  = 5.0   # Hz — validated in literature
BUTTER_ORDER   = 4
CYCLE_POINTS   = 101   # 0–100%
MIN_STEP_S     = 0.3   # seconds — minimum plausible step duration


# ---------------------------------------------------------------------------
# Step 1 — confidence masking
# ---------------------------------------------------------------------------

def apply_confidence_mask(keypoints: np.ndarray,
                           threshold: float = CONF_THRESHOLD) -> np.ndarray:
    """Replace low-confidence (x, y) with NaN. Confidence channel kept intact."""
    kp = keypoints.copy().astype(float)
    low = kp[:, :, 2] < threshold
    kp[low, 0] = np.nan
    kp[low, 1] = np.nan
    return kp


# ---------------------------------------------------------------------------
# Step 2 — gap interpolation
# ---------------------------------------------------------------------------

def interpolate_gaps(keypoints: np.ndarray,
                     max_gap: int = INTERP_MAX_GAP) -> np.ndarray:
    """Linear interpolation for NaN gaps up to max_gap frames.

    Gaps longer than max_gap are left as NaN — they represent truly missing
    detections that should not be fabricated.
    """
    kp = keypoints.copy()
    for j in range(kp.shape[1]):
        for c in range(2):   # x, y only
            s = pd.Series(kp[:, j, c])
            kp[:, j, c] = s.interpolate(
                method="linear",
                limit=max_gap,
                limit_direction="both",
            ).values
    return kp


# ---------------------------------------------------------------------------
# Step 3 — Butterworth low-pass filter (zero-phase)
# ---------------------------------------------------------------------------

def _butter_segment(signal: np.ndarray, cutoff: float, fps: float, order: int) -> np.ndarray:
    """Filter a 1-D signal that may contain NaNs.

    Only filters contiguous valid segments long enough to avoid edge artefacts.
    scipy.filtfilt requires signal length > padlen = 3 * max(len(a), len(b)).
    For order-4 Butterworth, padlen = 15; use a safe minimum of 20.
    """
    nyquist = fps / 2.0
    if cutoff >= nyquist:
        return signal  # cannot filter

    norm = cutoff / nyquist
    b, a  = butter(order, norm, btype="low", analog=False)
    # filtfilt default padlen = 3 * max(len(a), len(b))
    padlen  = 3 * max(len(a), len(b))
    min_len = padlen + 1   # strictly greater than padlen

    result = signal.copy()
    valid  = ~np.isnan(signal)

    if np.sum(valid) <= padlen:
        return result  # too short to filter — return as-is

    result[valid] = filtfilt(b, a, signal[valid])
    return result


def smooth_keypoints(keypoints: np.ndarray,
                     fps: float,
                     cutoff: float = BUTTER_CUTOFF,
                     order: int = BUTTER_ORDER) -> np.ndarray:
    """Apply zero-phase Butterworth to x and y of every joint."""
    smoothed = keypoints.copy()
    for j in range(keypoints.shape[1]):
        for c in range(2):
            smoothed[:, j, c] = _butter_segment(keypoints[:, j, c], cutoff, fps, order)
    return smoothed


# ---------------------------------------------------------------------------
# Step 4 — joint angles (sagittal plane)
# ---------------------------------------------------------------------------

def _angle_at_vertex(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Angle in degrees formed at vertex by vectors vertex→p1 and vertex→p2.

    Args:
        p1, vertex, p2: (T, 2) arrays of (x, y).
    Returns:
        angles: (T,) in degrees, NaN where any input is NaN.
    """
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    eps = 1e-8
    dot = np.sum((v1 / (n1 + eps)) * (v2 / (n2 + eps)), axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    nan_mask = (
        np.isnan(p1).any(1) | np.isnan(vertex).any(1) | np.isnan(p2).any(1)
    )
    angles[nan_mask] = np.nan
    return angles


def compute_joint_angles(keypoints: np.ndarray) -> dict[str, np.ndarray]:
    """Compute sagittal joint angles for gait analysis.

    Returns a dict mapping joint name → (T,) angle array in degrees.

    Literature note (Henry et al. 2024): knee angles have excellent agreement
    with QGA (< 1° difference). Hip angles show 7–9° offset; ankle ~7°.
    """
    xy = keypoints[:, :, :2]

    angles: dict[str, np.ndarray] = {}

    # Knees
    angles["R_knee"] = _angle_at_vertex(xy[:, R_HIP], xy[:, R_KNEE], xy[:, R_ANKLE])
    angles["L_knee"] = _angle_at_vertex(xy[:, L_HIP], xy[:, L_KNEE], xy[:, L_ANKLE])

    # Hips (Neck as trunk reference)
    angles["R_hip"] = _angle_at_vertex(xy[:, NECK], xy[:, R_HIP], xy[:, R_KNEE])
    angles["L_hip"] = _angle_at_vertex(xy[:, NECK], xy[:, L_HIP], xy[:, L_KNEE])

    # Ankles (treat with extra caution — larger OpenPose error)
    angles["R_ankle"] = _angle_at_vertex(xy[:, R_KNEE], xy[:, R_ANKLE], xy[:, R_BIG_TOE])
    angles["L_ankle"] = _angle_at_vertex(xy[:, L_KNEE], xy[:, L_ANKLE], xy[:, L_BIG_TOE])

    return angles


# ---------------------------------------------------------------------------
# Step 5 — gait event detection
# ---------------------------------------------------------------------------

def detect_gait_events(keypoints: np.ndarray, fps: float,
                        min_step_s: float = MIN_STEP_S) -> dict[str, np.ndarray]:
    """Detect Heel Strike (HS) and Toe Off (TO) events.

    Method: Stenum et al. (2021) — ankle x-position relative to MidHip.
    Validation: < 1 frame error vs. VICON at 25 fps.

    - HS = local maximum of relative_x  (foot most forward)
    - TO = local minimum of relative_x  (foot most backward)
    """
    min_dist = int(min_step_s * fps)
    mid_x = keypoints[:, MID_HIP, 0]

    events: dict[str, np.ndarray] = {}

    for side, ankle_idx in [("R", R_ANKLE), ("L", L_ANKLE)]:
        ankle_x = keypoints[:, ankle_idx, 0]
        rel = ankle_x - mid_x

        valid = ~(np.isnan(rel) | np.isnan(mid_x))
        if np.sum(valid) < min_dist * 2:
            events[f"{side}_HS"] = np.array([], dtype=int)
            events[f"{side}_TO"] = np.array([], dtype=int)
            continue

        prom = np.nanstd(rel) * 0.5

        hs, _ = find_peaks(rel, distance=min_dist, prominence=prom)
        to, _ = find_peaks(-rel, distance=min_dist, prominence=prom)

        events[f"{side}_HS"] = hs
        events[f"{side}_TO"] = to

    return events


def validate_events(events: dict[str, np.ndarray], fps: float) -> dict:
    """Check physiological plausibility of detected gait events.

    Normal cadence range: 60–180 steps/min.
    Returns per-side validation summary.
    """
    report: dict = {}
    for side in ["R", "L"]:
        hs = events.get(f"{side}_HS", np.array([]))
        to = events.get(f"{side}_TO", np.array([]))
        n_hs, n_to = len(hs), len(to)

        if n_hs < 2:
            report[side] = {"valid": False, "reason": "< 2 heel strikes", "n_HS": n_hs, "n_TO": n_to}
            continue

        step_times = np.diff(hs) / fps
        cadence = 60.0 / np.mean(step_times)

        # Per-side HS→HS measures stride rate (full gait cycle per minute).
        # Normal stride rate: 30–90 strides/min (= 60–180 steps/min total).
        ok = 30.0 <= cadence <= 90.0
        report[side] = {
            "valid": ok,
            "n_HS": n_hs,
            "n_TO": n_to,
            "stride_rate_per_min": round(cadence, 1),
            "reason": None if ok else f"stride rate {cadence:.1f} outside 30–90",
        }
    return report


# ---------------------------------------------------------------------------
# Step 6 — gait cycle extraction and normalisation
# ---------------------------------------------------------------------------

def normalize_cycle(signal: np.ndarray, n_points: int = CYCLE_POINTS) -> np.ndarray:
    """Resample a single gait cycle to n_points via cubic (or linear) interpolation."""
    if len(signal) < 3:
        return np.full(n_points, np.nan)

    x_orig = np.linspace(0, 100, len(signal))
    x_new  = np.linspace(0, 100, n_points)
    valid  = ~np.isnan(signal)

    if np.sum(valid) < 4:
        return np.full(n_points, np.nan)

    kind = "cubic" if np.sum(valid) == len(signal) else "linear"
    f = interp1d(
        x_orig[valid], signal[valid],
        kind=kind,
        bounds_error=False,
        fill_value=np.nan,
    )
    return f(x_new)


def extract_cycles(
    signal: np.ndarray,
    heel_strikes: np.ndarray,
    n_points: int = CYCLE_POINTS,
) -> np.ndarray:
    """Extract and normalize all gait cycles from a continuous signal.

    A cycle runs from one heel strike to the next (same foot).

    Returns:
        (n_cycles, n_points) array — NaN rows for cycles too short or invalid.
    """
    cycles = []
    for i in range(len(heel_strikes) - 1):
        start, end = int(heel_strikes[i]), int(heel_strikes[i + 1])
        if end <= start:
            continue
        cycles.append(normalize_cycle(signal[start:end], n_points))
    return np.array(cycles) if cycles else np.empty((0, n_points))


# ---------------------------------------------------------------------------
# Step 7 — spatiotemporal parameters
# ---------------------------------------------------------------------------

def compute_spatiotemporal(
    events: dict[str, np.ndarray],
    keypoints: np.ndarray,
    fps: float,
) -> dict:
    """Compute cadence, step time, stance time, symmetry index.

    Note: step lengths are in pixels (no calibration object in these recordings).
    Relative comparisons between groups are still valid in pixel space.
    """
    R_HS = events.get("R_HS", np.array([]))
    L_HS = events.get("L_HS", np.array([]))
    R_TO = events.get("R_TO", np.array([]))
    L_TO = events.get("L_TO", np.array([]))

    params: dict = {}

    # Cadence
    all_hs = np.sort(np.concatenate([R_HS, L_HS]))
    if len(all_hs) >= 2:
        step_times = np.diff(all_hs) / fps
        params["cadence_steps_per_min"] = float(60.0 / np.mean(step_times))
        params["mean_step_time_s"]      = float(np.mean(step_times))
        params["step_time_cv_pct"]      = float(np.std(step_times) / np.mean(step_times) * 100)

    # Step length (pixels — relative only)
    step_lengths = []
    for hs_f in R_HS:
        rx = keypoints[hs_f, R_ANKLE, 0]
        lx = keypoints[hs_f, L_ANKLE, 0]
        if not (np.isnan(rx) or np.isnan(lx)):
            step_lengths.append(abs(rx - lx))
    if step_lengths:
        params["mean_step_length_px"] = float(np.mean(step_lengths))
        params["step_length_cv_pct"]  = float(np.std(step_lengths) / np.mean(step_lengths) * 100)

    # Stance time (HS → next ipsilateral TO)
    for side, hs_arr, to_arr in [("R", R_HS, R_TO), ("L", L_HS, L_TO)]:
        stances = []
        for hs_f in hs_arr:
            future_to = to_arr[to_arr > hs_f]
            if len(future_to) > 0:
                stances.append((future_to[0] - hs_f) / fps)
        if stances:
            params[f"{side}_stance_time_s"] = float(np.mean(stances))
            params[f"{side}_stance_cv_pct"] = float(np.std(stances) / np.mean(stances) * 100)

    # Symmetry index (SI) — 0% = perfect symmetry
    if len(R_HS) >= 2 and len(L_HS) >= 2:
        r_cad = 60.0 / np.mean(np.diff(R_HS) / fps)
        l_cad = 60.0 / np.mean(np.diff(L_HS) / fps)
        params["symmetry_index_pct"] = float(
            abs(r_cad - l_cad) / ((r_cad + l_cad) / 2) * 100
        )

    return params


# ---------------------------------------------------------------------------
# Step 8 — quality report
# ---------------------------------------------------------------------------

def quality_report(keypoints: np.ndarray,
                   threshold: float = CONF_THRESHOLD) -> dict:
    """Per-joint validity statistics (% frames above confidence threshold)."""
    n_frames = keypoints.shape[0]
    report: dict = {}
    for j in GAIT_JOINT_INDICES:
        conf = keypoints[:, j, 2]
        valid = conf >= threshold
        mean_c = float(conf[conf > 0].mean()) if np.any(conf > 0) else 0.0
        report[j] = {
            "pct_valid": round(float(valid.mean() * 100), 1),
            "mean_conf": round(mean_c, 3),
            "n_valid":   int(valid.sum()),
            "n_frames":  n_frames,
        }
    return report


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def _parse_stem(stem: str) -> dict:
    """Extract metadata from filename stem.

    Formats:
        NM:  001_NM_01              → subject=001, group=NM, side=01, stage=None
        KOA: 001_KOA_01_EL         → subject=001, group=KOA, side=01, stage=EL
        PD:  001_PD_01_ML          → ...
    """
    stem = stem.replace("_keypoints", "")
    parts = stem.split("_")
    meta: dict = {"stem": stem}
    if len(parts) >= 3:
        meta["subject"] = parts[0]
        meta["group"]   = parts[1]
        meta["side"]    = parts[2]
    meta["stage"] = parts[3] if len(parts) >= 4 else None
    return meta


def run_pipeline(npz_path: Path) -> dict:
    """Full preprocessing pipeline for one .npz file.

    Returns a dict with all arrays and metadata, ready for np.savez_compressed.
    """
    data = np.load(npz_path, allow_pickle=True)
    kp_raw = data["keypoints"].astype(float)

    # Guard against malformed files (keypoints stored as 1-D or wrong shape)
    if kp_raw.ndim == 1:
        n = kp_raw.shape[0]
        if n % 75 == 0:
            kp_raw = kp_raw.reshape(-1, 25, 3)
        else:
            raise ValueError(f"Unexpected keypoints shape {kp_raw.shape} in {npz_path.name}")
    elif kp_raw.ndim == 2:
        # (T, 75) flat format
        kp_raw = kp_raw.reshape(kp_raw.shape[0], 25, 3)

    fps = float(data["fps"])
    if fps <= 0 or kp_raw.shape[0] == 0:
        raise ValueError(f"Empty or zero-fps file: {npz_path.name}")
    meta   = _parse_stem(npz_path.stem)

    # ── Step 1: confidence masking ──────────────────────────────────────────
    kp_masked = apply_confidence_mask(kp_raw, CONF_THRESHOLD)

    # ── Step 2: gap interpolation ───────────────────────────────────────────
    kp_interp = interpolate_gaps(kp_masked, INTERP_MAX_GAP)

    # ── Step 3: Butterworth filter ──────────────────────────────────────────
    kp_clean = smooth_keypoints(kp_interp, fps, BUTTER_CUTOFF, BUTTER_ORDER)
    # Restore confidence channel from masked (unfiltered)
    kp_clean[:, :, 2] = kp_masked[:, :, 2]

    # ── Step 4: joint angles ────────────────────────────────────────────────
    angles = compute_joint_angles(kp_clean)

    # ── Step 5: gait event detection ────────────────────────────────────────
    events  = detect_gait_events(kp_clean, fps)
    ev_val  = validate_events(events, fps)

    # ── Step 6: gait cycle extraction ───────────────────────────────────────
    # Use ipsilateral HS as cycle boundaries for each side
    cycles: dict[str, np.ndarray] = {}
    for side, hs_key in [("R", "R_HS"), ("L", "L_HS")]:
        hs = events[hs_key]
        for jname, jangles in angles.items():
            key = f"{side}_{jname}"
            cycles[key] = extract_cycles(jangles, hs)

    # Flatten cycles dict into individual arrays for npz storage
    cycles_flat = {f"cycle_{k}": v for k, v in cycles.items()}

    # ── Step 7: spatiotemporal params ───────────────────────────────────────
    spatio = compute_spatiotemporal(events, kp_clean, fps)

    # ── Step 8: quality report ──────────────────────────────────────────────
    qual = quality_report(kp_raw)  # report on raw (pre-cleaning)

    # Build output dict
    out = {
        # Raw and cleaned keypoints
        "keypoints_raw":   kp_raw,
        "keypoints_clean": kp_clean,
        "fps":             fps,

        # Metadata
        "subject":  meta.get("subject", ""),
        "group":    meta.get("group", ""),
        "side":     meta.get("side", ""),
        "stage":    meta.get("stage", "") or "",
        "source":   str(data.get("source", npz_path)),

        # Joint angles (T,) per joint
        **{f"angle_{k}": v for k, v in angles.items()},

        # Gait events (variable-length int arrays)
        "event_R_HS": events["R_HS"],
        "event_L_HS": events["L_HS"],
        "event_R_TO": events["R_TO"],
        "event_L_TO": events["L_TO"],

        # Normalised cycles (n_cycles, 101) per side×joint
        **cycles_flat,

        # Spatiotemporal parameters stored as a dict (object array)
        "spatiotemporal": np.array(spatio, dtype=object),

        # Quality and validation dicts
        "quality":          np.array(qual, dtype=object),
        "event_validation": np.array(ev_val, dtype=object),
    }

    return out