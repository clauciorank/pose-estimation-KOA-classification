"""
Generate all visualizations from extracted keypoints.

Run after extraction is complete:
    python3 src/visualization/generate_all.py

Outputs to data/output/plots/:
  - trajectory_[subject]_[group]_[stage].png  — per-subject joint trajectories
  - group_comparison_[joint].png               — boxplot across KOA/PD/NM
  - detection_quality.png                      — missed-frame rates per subject
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "output" / "plots"
MANIFEST = PROCESSED / "manifest.json"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Gait joints only (no head)
GAIT_JOINTS = {
    "Neck": 1, "MidHip": 8,
    "RHip": 9, "RKnee": 10, "RAnkle": 11,
    "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "LBigToe": 19, "LHeel": 21,
    "RBigToe": 22, "RHeel": 24,
}
CONF_THRESH = 0.1


# ── helpers ──────────────────────────────────────────────────────────────────

def load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def masked(kp, joint_idx, coord):
    """Return coordinate array with low-confidence frames as NaN."""
    idx = 0 if coord == "x" else 1
    conf = kp[:, joint_idx, 2]
    vals = kp[:, joint_idx, idx].astype(float)
    vals[conf < CONF_THRESH] = np.nan
    return vals


def time_axis(kp, fps):
    return np.arange(len(kp)) / float(fps)


# ── 1. per-subject trajectory plots ──────────────────────────────────────────

def plot_subject_trajectories(record: dict):
    path = Path(record["keypoints"])
    if not path.exists():
        return

    d = load_npz(path)
    kp = d["keypoints"]
    if kp.ndim != 3 or kp.shape[1:] != (25, 3):
        return  # malformed file, skip
    fps = float(d["fps"])
    t = time_axis(kp, fps)

    label = f"{record['subject']} | {record['group']} | {record.get('stage') or 'NM'} | side {record['side']}"

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(label, fontsize=11)

    # Row 0: Hip Y (vertical position)
    for name, idx in [("RHip", 9), ("LHip", 12), ("MidHip", 8)]:
        axes[0].plot(t, masked(kp, idx, "y"), label=name, lw=0.8)
    axes[0].set_ylabel("Y (px)")
    axes[0].set_title("Hip vertical position")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].invert_yaxis()

    # Row 1: Knee Y
    for name, idx in [("RKnee", 10), ("LKnee", 13)]:
        axes[1].plot(t, masked(kp, idx, "y"), label=name, lw=0.8)
    axes[1].set_ylabel("Y (px)")
    axes[1].set_title("Knee vertical position")
    axes[1].legend(fontsize=7)
    axes[1].invert_yaxis()

    # Row 2: Ankle Y
    for name, idx in [("RAnkle", 11), ("LAnkle", 14)]:
        axes[2].plot(t, masked(kp, idx, "y"), label=name, lw=0.8)
    axes[2].set_ylabel("Y (px)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Ankle vertical position")
    axes[2].legend(fontsize=7)
    axes[2].invert_yaxis()

    plt.tight_layout()
    stem = path.stem.replace("_keypoints", "")
    fig.savefig(OUTPUT / f"trajectory_{stem}.png", dpi=100)
    plt.close(fig)


# ── 2. group comparison boxplots ─────────────────────────────────────────────

def build_metrics_df(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        path = Path(rec["keypoints"])
        if not path.exists():
            continue
        d = load_npz(path)
        kp = d["keypoints"]
        if kp.ndim != 3 or kp.shape[1:] != (25, 3):
            continue
        group = rec["group"]
        stage = rec.get("stage") or "NM"

        for name, idx in GAIT_JOINTS.items():
            conf = kp[:, idx, 2]
            valid = kp[conf > CONF_THRESH, idx, :]
            if len(valid) < 10:
                continue
            y_vals = valid[:, 1].astype(float)
            rows.append({
                "subject": rec["subject"],
                "group": group,
                "stage": stage,
                "side": rec["side"],
                "joint": name,
                "range_px": float(y_vals.max() - y_vals.min()),
                "std_px": float(y_vals.std()),
                "detection_rate": float((conf > CONF_THRESH).mean()),
            })
    return pd.DataFrame(rows)


def plot_group_comparisons(df: pd.DataFrame):
    group_order = ["NM", "KOA", "PD"]
    palette = {"NM": "#4C9BE8", "KOA": "#E8834C", "PD": "#8B4CE8"}

    key_joints = ["RKnee", "LKnee", "RAnkle", "LAnkle", "RHip", "LHip"]
    sub = df[df["joint"].isin(key_joints)]

    # Range comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Joint vertical range by group (all subjects)", fontsize=13)
    for ax, joint in zip(axes.flat, key_joints):
        data = sub[sub["joint"] == joint]
        sns.boxplot(data=data, x="group", y="range_px", order=group_order,
                    palette=palette, ax=ax, width=0.5)
        sns.stripplot(data=data, x="group", y="range_px", order=group_order,
                      color="black", size=3, alpha=0.4, ax=ax)
        ax.set_title(joint)
        ax.set_xlabel("")
        ax.set_ylabel("Range (px)")
    plt.tight_layout()
    fig.savefig(OUTPUT / "group_comparison_range.png", dpi=120)
    plt.close(fig)

    # Detection quality
    det = df.groupby(["subject", "group", "joint"])["detection_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=det, x="joint", y="detection_rate", hue="group",
                hue_order=group_order, palette=palette, ax=ax)
    ax.set_title("Keypoint detection rate by joint and group")
    ax.set_ylabel("Detection rate (confidence > 0.1)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(0.7, color="red", lw=1, ls="--", label="70% threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT / "detection_quality.png", dpi=120)
    plt.close(fig)


# ── 3. missed-frame summary ───────────────────────────────────────────────────

def plot_missed_frames(records: list[dict]):
    rows = []
    for rec in records:
        path = Path(rec["keypoints"])
        if not path.exists():
            continue
        missed = rec.get("missed_frames", 0)
        total = rec.get("total_frames", 1)
        rows.append({
            "label": f"{rec['subject']}_{rec.get('stage') or 'NM'}_{rec['side']}",
            "group": rec["group"],
            "miss_pct": 100 * missed / max(total, 1),
        })
    df = pd.DataFrame(rows).sort_values("miss_pct", ascending=False)

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.2), 5))
    colors = {"KOA": "#E8834C", "PD": "#8B4CE8", "NM": "#4C9BE8"}
    bar_colors = [colors.get(g, "gray") for g in df["group"]]
    ax.bar(range(len(df)), df["miss_pct"], color=bar_colors, alpha=0.8)
    ax.axhline(30, color="red", lw=1, ls="--", label="30% warning threshold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=90, fontsize=6)
    ax.set_ylabel("Missed frames (%)")
    ax.set_title("Detection miss rate per subject (head blur impact)")
    ax.legend()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in colors.items()] +
              [plt.Line2D([0], [0], color="red", ls="--", label="30% threshold")],
              fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT / "missed_frames.png", dpi=120)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def records_from_filesystem() -> list[dict]:
    """Build records by scanning processed/ when manifest.json is absent."""
    import re
    records = []
    for npz in sorted(PROCESSED.rglob("*.npz")):
        stem = npz.stem.replace("_keypoints", "")
        # KOA/PD: SUBJECT_GROUP_SIDE_STAGE  e.g. 001_KOA_01_SV
        # NM:     SUBJECT_GROUP_SIDE         e.g. 001_NM_01
        m = re.match(r"^(\d+)_(KOA|PD|NM)_(\d+)(?:_([A-Z]+))?$", stem)
        if not m:
            continue
        subject, group, side, stage = m.groups()
        d = np.load(npz, allow_pickle=True)
        records.append({
            "subject": subject,
            "group": group,
            "side": side,
            "stage": stage,
            "keypoints": str(npz),
            "missed_frames": int(d.get("missed_frames", 0)),
            "total_frames": int(d.get("total_frames", len(d["keypoints"]))),
        })
    return records


def main():
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            records = json.load(f)
        # Remap Docker container paths (/data/processed/...) to host paths
        for r in records:
            r["keypoints"] = str(PROCESSED / Path(r["keypoints"]).relative_to("/data/processed"))
    else:
        print("manifest.json not found — scanning filesystem.")
        records = records_from_filesystem()

    processed = [r for r in records if Path(r["keypoints"]).exists()]
    print(f"Found {len(processed)} processed files.")

    print("Generating per-subject trajectory plots...")
    for i, rec in enumerate(processed, 1):
        plot_subject_trajectories(rec)
        if i % 10 == 0:
            print(f"  {i}/{len(processed)}")

    print("Building metrics dataframe...")
    df = build_metrics_df(processed)
    df.to_csv(OUTPUT / "gait_metrics.csv", index=False)
    print(f"  Saved gait_metrics.csv ({len(df)} rows)")

    if len(df) > 0:
        print("Generating group comparison plots...")
        plot_group_comparisons(df)

        print("Generating missed-frame summary...")
        plot_missed_frames(processed)

    print(f"\nAll visualizations saved to {OUTPUT}")


if __name__ == "__main__":
    main()