"""
Dataset construction for gait classification (NM / KOA / PD).

Two representations are built from the same cleaned .npz files:

  Tabular  — one row per FILE; features = spatiotemporal params +
             per-joint cycle statistics (ROM, peak, timing, variability).
             Unit of analysis: recording.

  Sequence — one row per CYCLE; features = (101, 4) normalised angle
             time-series [R_knee, L_knee, R_hip, L_hip].
             Unit of analysis: gait cycle.

Both share the same subject-level group key so the train/test split is
identical: no subject's cycles ever appear in both train and test.

Subject key format: "{GROUP}_{SUBJECT}"  e.g. "KOA_001", "NM_015"
This ensures that a KOA subject appearing in EL + MD + SV stages is
always kept together.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

CLEANED_DIR = Path("data/cleaned")

GROUP_LABEL = {"NM": 0, "KOA": 1, "PD": 2}
LABEL_GROUP = {v: k for k, v in GROUP_LABEL.items()}

# For binary NM vs KOA
NM_KOA_LABEL = {"NM": 0, "KOA": 1}

# For KOA severity staging
STAGE_LABEL = {"EL": 0, "MD": 1, "SV": 2}
LABEL_STAGE = {v: k for k, v in STAGE_LABEL.items()}

# Joints used as sequence features (ankles excluded — higher OpenPose error)
SEQ_JOINTS = ["R_knee", "L_knee", "R_hip", "L_hip"]
N_TIMESTEPS = 101
N_SEQ_FEATURES = len(SEQ_JOINTS)   # 4


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _subject_key(d) -> str:
    grp   = str(d["group"])
    subj  = str(d["subject"])
    stage = str(d["stage"])
    # Include stage so 001_KOA_EL and 001_KOA_MD are treated as different subjects
    return f"{grp}_{stage}_{subj}" if stage else f"{grp}_{subj}"


def _cycle_stats(cycles: np.ndarray) -> dict:
    """Aggregate statistics across normalised cycles (n_cycles, 101)."""
    if cycles.ndim != 2 or cycles.shape[0] == 0:
        return {k: np.nan for k in [
            "n_cycles", "mean_ROM", "std_ROM",
            "mean_peak", "std_peak", "mean_peak_pct",
            "mean_min", "mean_cv",
        ]}
    roms   = np.nanmax(cycles, axis=1) - np.nanmin(cycles, axis=1)
    peaks  = np.nanmax(cycles, axis=1)
    mins   = np.nanmin(cycles, axis=1)
    mean_c = np.nanmean(cycles, axis=0)
    std_c  = np.nanstd(cycles, axis=0)
    cv     = np.nanmean(std_c) / (abs(np.nanmean(mean_c)) + 1e-8) * 100

    return {
        "n_cycles":      float(cycles.shape[0]),
        "mean_ROM":      float(np.nanmean(roms)),
        "std_ROM":       float(np.nanstd(roms)),
        "mean_peak":     float(np.nanmean(peaks)),
        "std_peak":      float(np.nanstd(peaks)),
        "mean_peak_pct": float(np.nanargmax(mean_c)),
        "mean_min":      float(np.nanmean(mins)),
        "mean_cv":       float(cv),
    }


def _quality_mean(quality: dict, indices=(9, 10, 11, 12, 13, 14)) -> float:
    vals = [quality[i]["pct_valid"] for i in indices if i in quality]
    return float(np.mean(vals)) if vals else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class TabularDataset(NamedTuple):
    X:        pd.DataFrame        # (n_files, n_features)
    y:        np.ndarray          # (n_files,) int labels
    groups:   np.ndarray          # (n_files,) subject key strings
    labels:   list[str]           # ["NM", "KOA", "PD"]
    feature_names: list[str]


class SequenceDataset(NamedTuple):
    X:        np.ndarray          # (n_cycles, 101, 4)
    y:        np.ndarray          # (n_cycles,) int labels
    groups:   np.ndarray          # (n_cycles,) subject key strings
    labels:   list[str]


def load_tabular_dataset(
    cleaned_dir: Path = CLEANED_DIR,
    filter_groups: list[str] | None = None,
    label_mode: str = "group",          # "group" | "nm_koa" | "stage"
) -> TabularDataset:
    """Build a tabular feature matrix — one row per cleaned .npz file."""
    rows = []

    for p in sorted(cleaned_dir.rglob("*_cleaned.npz")):
        d   = np.load(p, allow_pickle=True)
        grp = str(d["group"])
        stage = str(d["stage"])

        if grp not in GROUP_LABEL:
            continue
        if filter_groups is not None and grp not in filter_groups:
            continue

        # Resolve label based on mode
        if label_mode == "nm_koa":
            if grp not in NM_KOA_LABEL:
                continue
            label = NM_KOA_LABEL[grp]
        elif label_mode == "stage":
            if stage not in STAGE_LABEL:
                continue
            label = STAGE_LABEL[stage]
        else:
            label = GROUP_LABEL[grp]

        row: dict = {
            "subject_key": _subject_key(d),
            "label":       label,
            "group":       grp,
            "stage":       stage,
        }

        # ── Spatiotemporal features ─────────────────────────────────────────
        sp = d["spatiotemporal"].item() if d["spatiotemporal"].ndim == 0 else {}
        for key in [
            "cadence_steps_per_min", "mean_step_time_s", "step_time_cv_pct",
            "mean_step_length_px", "step_length_cv_pct",
            "R_stance_time_s", "L_stance_time_s",
            "symmetry_index_pct",
        ]:
            row[f"st_{key}"] = sp.get(key, np.nan)

        # ── Cycle statistics per joint ───────────────────────────────────────
        for side in ["R", "L"]:
            for joint in ["knee", "hip", "ankle"]:
                key = f"cycle_{side}_{side}_{joint}"
                cycles = d.get(key, np.empty((0, 101)))
                if isinstance(cycles, np.ndarray) and cycles.ndim == 2:
                    stats = _cycle_stats(cycles)
                else:
                    stats = _cycle_stats(np.empty((0, 101)))
                for stat_name, val in stats.items():
                    row[f"{side}_{joint}_{stat_name}"] = val

        # ── Data quality ─────────────────────────────────────────────────────
        qual = d["quality"].item() if d["quality"].ndim == 0 else {}
        row["quality_mean_pct"] = _quality_mean(qual)

        rows.append(row)

    df = pd.DataFrame(rows)
    # Drop n_cycles: reflects recording duration, not gait quality
    feature_cols = [c for c in df.columns
                    if c not in ("subject_key", "label", "group", "stage")
                    and not c.endswith("_n_cycles")]

    # NaNs are left as-is here; imputation must happen inside each CV fold
    # (fitted on train split only) to avoid leakage.
    # XGBoost handles NaN natively; RF and SVM use per-fold SimpleImputer
    # pipelines defined in tabular.py.

    X = df[feature_cols]
    y = df["label"].values
    groups = df["subject_key"].values

    if label_mode == "nm_koa":
        label_names = ["NM", "KOA"]
    elif label_mode == "stage":
        label_names = ["EL", "MD", "SV"]
    else:
        label_names = ["NM", "KOA", "PD"]

    return TabularDataset(
        X=X, y=y, groups=groups,
        labels=label_names,
        feature_names=list(X.columns),
    )


def load_sequence_dataset(
    cleaned_dir: Path = CLEANED_DIR,
    filter_groups: list[str] | None = None,
    label_mode: str = "group",
) -> SequenceDataset:
    """Build a per-cycle sequence dataset — one row per normalised gait cycle.

    Each sample: (101, 4) array with [R_knee, L_knee, R_hip, L_hip] angles.
    Cycles from both R and L heel-strike boundaries are included.
    Cycles with >20% NaN timesteps are dropped.
    """
    seqs, labels, groups = [], [], []

    for p in sorted(cleaned_dir.rglob("*_cleaned.npz")):
        d     = np.load(p, allow_pickle=True)
        grp   = str(d["group"])
        stage = str(d["stage"])

        if grp not in GROUP_LABEL:
            continue
        if filter_groups is not None and grp not in filter_groups:
            continue

        if label_mode == "nm_koa":
            if grp not in NM_KOA_LABEL:
                continue
            label = NM_KOA_LABEL[grp]
        elif label_mode == "stage":
            if stage not in STAGE_LABEL:
                continue
            label = STAGE_LABEL[stage]
        else:
            label = GROUP_LABEL[grp]

        subj = _subject_key(d)

        for hs_side in ["R", "L"]:
            # Collect one matrix per cycle side
            joint_cycles: list[np.ndarray] = []
            all_ok = True
            for joint in SEQ_JOINTS:
                key = f"cycle_{hs_side}_{joint}"
                mat = d.get(key, np.empty((0, 101)))
                if not isinstance(mat, np.ndarray) or mat.ndim != 2 or mat.shape[1] != N_TIMESTEPS:
                    all_ok = False
                    break
                joint_cycles.append(mat)

            if not all_ok or not joint_cycles:
                continue

            n_cycles = joint_cycles[0].shape[0]
            if not all(m.shape[0] == n_cycles for m in joint_cycles):
                continue

            for i in range(n_cycles):
                # Stack joints → (101, 4)
                cycle = np.stack([m[i] for m in joint_cycles], axis=1)

                # Drop cycles with > 20% NaN
                nan_frac = np.isnan(cycle).mean()
                if nan_frac > 0.20:
                    continue

                # Forward-fill remaining NaNs within the cycle
                cycle_df = pd.DataFrame(cycle).ffill().bfill()
                if cycle_df.isnull().any().any():
                    continue  # still NaN (entire column) — skip

                seqs.append(cycle_df.values.astype(np.float32))
                labels.append(label)
                groups.append(subj)

    X      = np.stack(seqs, axis=0)      # (N, 101, 4)
    y      = np.array(labels, dtype=int)
    groups = np.array(groups)

    if label_mode == "nm_koa":
        label_names = ["NM", "KOA"]
    elif label_mode == "stage":
        label_names = ["EL", "MD", "SV"]
    else:
        label_names = ["NM", "KOA", "PD"]

    return SequenceDataset(X=X, y=y, groups=groups, labels=label_names)


def subject_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subject-level train/test split.

    Subjects are stratified by their majority class so that each group
    (NM, KOA, PD) is represented in both splits.

    Returns:
        train_idx, test_idx — indices into the original arrays.
    """
    from sklearn.model_selection import GroupShuffleSplit

    # Derive majority label per subject
    unique_subjects = np.unique(groups)
    subj_labels = {}
    for s in unique_subjects:
        mask = groups == s
        vals, counts = np.unique(y[mask], return_counts=True)
        subj_labels[s] = vals[np.argmax(counts)]

    subj_arr  = np.array(unique_subjects)
    label_arr = np.array([subj_labels[s] for s in unique_subjects])

    # Stratified split at subject level
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_subj_idx, test_subj_idx = next(sss.split(subj_arr, label_arr))

    train_subjects = set(subj_arr[train_subj_idx])
    test_subjects  = set(subj_arr[test_subj_idx])

    train_idx = np.where(np.isin(groups, list(train_subjects)))[0]
    test_idx  = np.where(np.isin(groups, list(test_subjects)))[0]

    return train_idx, test_idx


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights for imbalanced data.

    Returns an array sized to the number of unique classes in y,
    ordered by class index (0, 1, ..., n_classes-1).
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    # Build an array indexed 0..max(classes)
    n = int(classes.max()) + 1
    weight_arr = np.ones(n)
    for c, w in zip(classes, weights):
        weight_arr[c] = w
    return weight_arr


if __name__ == "__main__":
    print("Loading tabular dataset...")
    tab = load_tabular_dataset()
    print(f"  X shape : {tab.X.shape}")
    print(f"  Classes : {dict(zip(*np.unique(tab.y, return_counts=True)))}")
    print(f"  Subjects: {len(np.unique(tab.groups))}")

    print("\nLoading sequence dataset...")
    seq = load_sequence_dataset()
    print(f"  X shape : {seq.X.shape}  (cycles × timesteps × features)")
    print(f"  Classes : {dict(zip(*np.unique(seq.y, return_counts=True)))}")
    print(f"  Subjects: {len(np.unique(seq.groups))}")

    print("\nSubject-level split (75/25):")
    tr, te = subject_split(tab.y, tab.groups)
    from collections import Counter
    print(f"  Train: {len(tr)} files  {dict(Counter(tab.y[tr]))}")
    print(f"  Test : {len(te)} files  {dict(Counter(tab.y[te]))}")