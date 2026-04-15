"""
Gait cycle validation plot for 3 individuals.

For each subject generates a 3-row figure:
  Row 1 — Filtered ankle trajectory (relative to MidHip) with HS/TO events
  Row 2 — Knee angle time-series with HS events and extracted cycle windows
  Row 3 — All individual normalised knee cycles overlaid (R side)

Run:
    python3 src/analysis/validate_cycles.py
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── samples to validate ────────────────────────────────────────────────────────
SAMPLES = [
    ("data/cleaned/NM/NM/005_NM_01_cleaned.npz",   "NM — 005_NM_01"),
    ("data/cleaned/KOA/EL/003_KOA_01_EL_cleaned.npz", "KOA EL — 003_KOA_01"),
    ("data/cleaned/PD/ML/004_PD_01_ML_cleaned.npz",   "PD ML — 004_PD_01"),
]

OUT_DIR = Path("data/output/figures")
GROUP_COLORS = {"NM": "#2ecc71", "KOA": "#e74c3c", "PD": "#3498db"}
CYCLE_COLORS = plt.cm.tab20.colors   # up to 20 distinct colors for individual cycles

MID_HIP, R_ANKLE, L_ANKLE = 8, 11, 14
CYCLE_PCT = np.linspace(0, 100, 101)


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def relative_ankle(kp_clean: np.ndarray, ankle_idx: int) -> np.ndarray:
    return kp_clean[:, ankle_idx, 0] - kp_clean[:, MID_HIP, 0]


def make_validation_figure(npz_path: str, label: str) -> plt.Figure:
    d = load(npz_path)

    kp     = d["keypoints_clean"]          # (T, 25, 3)
    fps    = float(d["fps"])
    time   = np.arange(kp.shape[0]) / fps

    r_hs   = d["event_R_HS"].astype(int)
    l_hs   = d["event_L_HS"].astype(int)
    r_to   = d["event_R_TO"].astype(int)
    l_to   = d["event_L_TO"].astype(int)

    r_knee  = d["angle_R_knee"]
    l_knee  = d["angle_L_knee"]
    r_rel   = relative_ankle(kp, R_ANKLE)
    l_rel   = relative_ankle(kp, L_ANKLE)

    # Normalised cycles already stored: (n_cycles, 101)
    cycles_r = d.get("cycle_R_R_knee", np.empty((0, 101)))
    cycles_l = d.get("cycle_L_L_knee", np.empty((0, 101)))

    # Group colour
    grp = str(d.get("group", b"")).replace("b'", "").replace("'", "")
    color = GROUP_COLORS.get(grp, "#888888")

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(f"Cycle validation — {label}", fontsize=13, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(
        3, 2, hspace=0.45, wspace=0.3,
        height_ratios=[1, 1, 1.1],
    )

    # ── Row 0: relative ankle trajectories ──────────────────────────────────
    for col, (side, rel, hs, to, ankle_col) in enumerate([
        ("Right", r_rel, r_hs, r_to, "#2980b9"),
        ("Left",  l_rel, l_hs, l_to, "#8e44ad"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(time, rel, color=ankle_col, lw=1.0, alpha=0.85, label="ankle – MidHip")

        # Mark HS and TO
        for fr in hs:
            ax.axvline(fr / fps, color="green", lw=1.0, alpha=0.8, zorder=3)
        for fr in to:
            ax.axvline(fr / fps, color="red", lw=1.0, alpha=0.6, ls="--", zorder=3)

        # Shade cycle windows between consecutive HS
        for i in range(len(hs) - 1):
            t0, t1 = hs[i] / fps, hs[i + 1] / fps
            ax.axvspan(t0, t1, alpha=0.07 if i % 2 == 0 else 0.0,
                       color=color, zorder=0)

        ax.set_title(f"{side} ankle relative to MidHip", fontsize=9)
        ax.set_ylabel("Relative X (px)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

        hs_patch = mpatches.Patch(color="green", label=f"Heel Strike (n={len(hs)})")
        to_patch = mpatches.Patch(color="red",   label=f"Toe Off (n={len(to)})")
        ax.legend(handles=[hs_patch, to_patch], fontsize=7, loc="upper left")

    # ── Row 1: knee angle time-series + cycle windows ───────────────────────
    for col, (side, angle, hs, knee_col) in enumerate([
        ("Right", r_knee, r_hs, "#e74c3c"),
        ("Left",  l_knee, l_hs, "#e67e22"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(time, angle, color=knee_col, lw=0.9, alpha=0.85, label=f"{side} knee angle")

        for i, fr in enumerate(hs):
            ax.axvline(fr / fps, color="green", lw=1.0, alpha=0.7, zorder=3)

        # Shade each cycle window with cycle number label
        for i in range(len(hs) - 1):
            t0, t1 = hs[i] / fps, hs[i + 1] / fps
            ax.axvspan(t0, t1, alpha=0.10, color=CYCLE_COLORS[i % len(CYCLE_COLORS)], zorder=0)
            ax.text(
                (t0 + t1) / 2,
                ax.get_ylim()[0] if ax.get_ylim()[0] != 0.0 else 125,
                str(i + 1),
                ha="center", va="bottom", fontsize=7, color="gray",
            )

        ax.set_title(f"{side} knee angle (°) — cycle windows", fontsize=9)
        ax.set_ylabel("Angle (°)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # ── Row 2: individual normalised cycles overlaid ─────────────────────────
    for col, (side, cycles, hs_key) in enumerate([
        ("Right (R_HS boundaries)", cycles_r, r_hs),
        ("Left  (L_HS boundaries)", cycles_l, l_hs),
    ]):
        ax = fig.add_subplot(gs[2, col])

        valid_cycles = [c for c in cycles if not np.all(np.isnan(c))]
        n = len(valid_cycles)

        for i, cyc in enumerate(valid_cycles):
            cyc_color = CYCLE_COLORS[i % len(CYCLE_COLORS)]
            ax.plot(CYCLE_PCT, cyc, color=cyc_color, lw=1.5, alpha=0.75,
                    label=f"Cycle {i + 1}")

        if n > 1:
            mat  = np.array(valid_cycles)
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0)
            ax.plot(CYCLE_PCT, mean, color="black", lw=2.2, zorder=5, label="Mean")
            ax.fill_between(CYCLE_PCT, mean - std, mean + std,
                            color="black", alpha=0.12, zorder=4, label="±1 SD")

        ax.set_xlabel("Gait cycle (%)", fontsize=8)
        ax.set_ylabel("Knee angle (°)", fontsize=8)
        ax.set_title(f"{side} knee — {n} normalised cycles", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if n <= 10:
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.legend(fontsize=6, loc="upper right", ncol=2)

        # Annotate ROM
        if n > 0:
            mat_all = np.array(valid_cycles)
            mean_c  = np.nanmean(mat_all, axis=0)
            rom     = np.nanmax(mean_c) - np.nanmin(mean_c)
            ax.text(0.02, 0.97, f"ROM = {rom:.1f}°", transform=ax.transAxes,
                    fontsize=8, va="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for npz_path, label in SAMPLES:
        p = Path(npz_path)
        if not p.exists():
            print(f"  [MISSING] {npz_path}")
            continue

        print(f"  Generating: {label}")
        fig = make_validation_figure(npz_path, label)

        slug = label.replace(" ", "_").replace("—", "").replace("/", "-")
        out  = OUT_DIR / f"validate_cycles_{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()