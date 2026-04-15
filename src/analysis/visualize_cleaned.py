"""
Diagnostic and comparative visualizations for cleaned gait data.

Generates plots to:
  1. Inspect signal quality (confidence heat-map)
  2. Compare raw vs. filtered trajectories + detected events
  3. Overlay normalised gait cycles (mean ± SD) per group
  4. Compare kinematic ROM and spatiotemporal parameters across groups

Usage:
    python3 src/analysis/visualize_cleaned.py [--sample KOA/EL/001_KOA_01_EL]

All figures are saved to data/output/figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.preprocess import (
    CONF_THRESHOLD, GAIT_JOINT_INDICES,
    run_pipeline,
    apply_confidence_mask, interpolate_gaps, smooth_keypoints,
    detect_gait_events, compute_joint_angles, quality_report,
    NECK, MID_HIP, R_ANKLE, L_ANKLE, R_KNEE, R_HIP, L_KNEE, L_HIP,
)

PROCESSED_DIR = Path("data/processed")
CLEANED_DIR   = Path("data/cleaned")
OUTPUT_DIR    = Path("data/output/figures")

JOINT_NAMES = {
    1: "Neck", 8: "MidHip",
    9: "RHip", 10: "RKnee", 11: "RAnkle",
    12: "LHip", 13: "LKnee", 14: "LAnkle",
    19: "LBigToe", 21: "LHeel", 22: "RBigToe", 24: "RHeel",
}

GROUP_COLORS = {"NM": "#2ecc71", "KOA": "#e74c3c", "PD": "#3498db"}
GROUP_ORDER  = ["NM", "KOA", "PD"]

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Signal quality heat-map
# ──────────────────────────────────────────────────────────────────────────────

def fig_quality_heatmap(npz_path: Path, out_dir: Path) -> None:
    """Confidence heat-map: frames × gait joints for one recording."""
    data = np.load(npz_path, allow_pickle=True)
    kp   = data["keypoints"].astype(float)
    fps  = float(data["fps"])
    time = np.arange(kp.shape[0]) / fps

    indices = GAIT_JOINT_INDICES
    conf_mat = kp[:, indices, 2].T   # (n_joints, T)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(
        conf_mat, aspect="auto", origin="lower",
        extent=[time[0], time[-1], -0.5, len(indices) - 0.5],
        vmin=0, vmax=1, cmap="RdYlGn",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([JOINT_NAMES.get(i, str(i)) for i in indices], fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Keypoint confidence — {npz_path.stem}")
    plt.colorbar(im, ax=ax, label="Confidence")
    _save(fig, out_dir, f"quality_{npz_path.stem}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Raw vs filtered + gait events
# ──────────────────────────────────────────────────────────────────────────────

def fig_raw_vs_filtered(npz_path: Path, out_dir: Path) -> None:
    """Four-panel plot: raw trajectory, filtered, relative (R and L ankles)."""
    data  = np.load(npz_path, allow_pickle=True)
    kp_r  = data["keypoints"].astype(float)
    fps   = float(data["fps"])
    time  = np.arange(kp_r.shape[0]) / fps

    kp_m  = apply_confidence_mask(kp_r, CONF_THRESHOLD)
    kp_i  = interpolate_gaps(kp_m)
    kp_f  = smooth_keypoints(kp_i, fps)
    kp_f[:, :, 2] = kp_m[:, :, 2]

    events = detect_gait_events(kp_f, fps)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Raw vs Filtered + Events — {npz_path.stem}", fontsize=11)

    for row, (side, ankle_idx) in enumerate([("Right", R_ANKLE), ("Left", L_ANKLE)]):
        # Raw
        ax = axes[row, 0]
        ax.plot(time, kp_r[:, ankle_idx, 0], alpha=0.4, lw=0.8, color="gray", label="raw x")
        ax.plot(time, kp_f[:, ankle_idx, 0], lw=1.2, color="steelblue", label="filtered x")
        ax.set_ylabel("X (px)")
        ax.set_title(f"{side} ankle — absolute x position")
        ax.legend(fontsize=7)

        # Relative + events
        ax = axes[row, 1]
        rel = kp_f[:, ankle_idx, 0] - kp_f[:, MID_HIP, 0]
        rel_r = kp_r[:, ankle_idx, 0] - kp_r[:, MID_HIP, 0]
        ax.plot(time, rel_r, alpha=0.3, lw=0.8, color="gray", label="raw rel")
        ax.plot(time, rel,   lw=1.2, color="steelblue", label="filtered rel")
        hs_key, to_key = f"{side[0]}_HS", f"{side[0]}_TO"
        for fr in events.get(hs_key, []):
            ax.axvline(fr / fps, color="green", lw=0.8, alpha=0.7)
        for fr in events.get(to_key, []):
            ax.axvline(fr / fps, color="red", lw=0.8, alpha=0.7)
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], color="green", label="Heel Strike"),
                Line2D([0], [0], color="red",   label="Toe Off"),
            ],
            fontsize=7,
        )
        ax.set_ylabel("Relative X (px)")
        ax.set_title(f"{side} ankle relative to MidHip")

    for ax in axes[1]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    _save(fig, out_dir, f"raw_vs_filtered_{npz_path.stem}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Normalised gait cycles mean ± SD per group
# ──────────────────────────────────────────────────────────────────────────────

def _load_all_cleaned(groups=("KOA", "PD", "NM")) -> dict[str, list[Path]]:
    """Return dict group → list of cleaned .npz paths."""
    paths: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(CLEANED_DIR.rglob("*_cleaned.npz")):
        for grp in groups:
            if f"/{grp}/" in str(p) or f"\\{grp}\\" in str(p):
                paths[grp].append(p)
                break
    return paths


def fig_gait_cycles_by_group(out_dir: Path, joint: str = "R_knee") -> None:
    """Mean ± SD normalised gait cycle for each group."""
    group_paths = _load_all_cleaned()

    if not any(group_paths.values()):
        print("  No cleaned files found — run run_preprocess.py first.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pct = np.linspace(0, 100, 101)
    cycle_key = f"cycle_R_{joint}"

    for grp in GROUP_ORDER:
        all_cycles = []
        for p in group_paths.get(grp, []):
            d = np.load(p, allow_pickle=True)
            if cycle_key in d:
                mat = d[cycle_key]   # (n_cycles, 101)
                if mat.ndim == 2 and mat.shape[1] == 101:
                    all_cycles.append(mat)
        if not all_cycles:
            continue
        mat = np.vstack(all_cycles)
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        n    = mat.shape[0]
        color = GROUP_COLORS[grp]
        ax.plot(pct, mean, color=color, lw=2, label=f"{grp} (n={n} cycles)")
        ax.fill_between(pct, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Angle (°)")
    ax.set_title(f"Normalised {joint.replace('_', ' ')} angle — mean ± SD by group")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, f"cycles_{joint}_by_group.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — ROM and cadence box-plots across groups
# ──────────────────────────────────────────────────────────────────────────────

def fig_spatiotemporal_boxplots(out_dir: Path) -> None:
    """Box-plots of cadence, symmetry index, and stance time by group."""
    group_paths = _load_all_cleaned()

    if not any(group_paths.values()):
        print("  No cleaned files found — run run_preprocess.py first.")
        return

    params_of_interest = [
        "cadence_steps_per_min",
        "mean_step_time_s",
        "symmetry_index_pct",
        "R_stance_time_s",
        "L_stance_time_s",
        "step_time_cv_pct",
    ]
    param_labels = [
        "Cadence (steps/min)",
        "Mean step time (s)",
        "Symmetry index (%)",
        "R stance time (s)",
        "L stance time (s)",
        "Step time CV (%)",
    ]

    data_by_param: dict[str, dict[str, list]] = {
        p: {g: [] for g in GROUP_ORDER} for p in params_of_interest
    }

    for grp in GROUP_ORDER:
        for fp in group_paths.get(grp, []):
            d = np.load(fp, allow_pickle=True)
            spatio = d.get("spatiotemporal", None)
            if spatio is None:
                continue
            if isinstance(spatio, np.ndarray):
                spatio = spatio.item() if spatio.ndim == 0 else {}
            for p in params_of_interest:
                if p in spatio:
                    data_by_param[p][grp].append(spatio[p])

    n_params = len(params_of_interest)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (param, label) in enumerate(zip(params_of_interest, param_labels)):
        ax = axes[i]
        plot_data  = [data_by_param[param][g] for g in GROUP_ORDER]
        colors     = [GROUP_COLORS[g] for g in GROUP_ORDER]
        bp = ax.boxplot(
            [d for d in plot_data],
            labels=GROUP_ORDER,
            patch_artist=True,
            medianprops={"color": "black", "lw": 2},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # Overlay individual points
        for j, (grp_data, grp) in enumerate(zip(plot_data, GROUP_ORDER)):
            x = np.random.normal(j + 1, 0.06, size=len(grp_data))
            ax.scatter(x, grp_data, alpha=0.5, s=18, color=GROUP_COLORS[grp], zorder=3)
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    plt.suptitle("Spatiotemporal Parameters by Group", fontsize=12)
    plt.tight_layout()
    _save(fig, out_dir, "spatiotemporal_boxplots.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — ROM box-plots for each joint angle by group
# ──────────────────────────────────────────────────────────────────────────────

def fig_rom_by_group(out_dir: Path) -> None:
    """Range of motion (ROM) extracted from normalised cycles, by group."""
    group_paths = _load_all_cleaned()

    if not any(group_paths.values()):
        print("  No cleaned files found — run run_preprocess.py first.")
        return

    joints = ["R_knee", "L_knee", "R_hip", "L_hip"]
    cycle_keys = {j: f"cycle_R_{j}" if j.startswith("R") else f"cycle_L_{j}" for j in joints}

    rom: dict[str, dict[str, list]] = {j: {g: [] for g in GROUP_ORDER} for j in joints}

    for grp in GROUP_ORDER:
        for fp in group_paths.get(grp, []):
            d = np.load(fp, allow_pickle=True)
            for j, ck in cycle_keys.items():
                if ck in d:
                    mat = d[ck]
                    if mat.ndim == 2 and mat.shape[0] > 0:
                        mean_cycle = np.nanmean(mat, axis=0)
                        r = float(np.nanmax(mean_cycle) - np.nanmin(mean_cycle))
                        if not np.isnan(r):
                            rom[j][grp].append(r)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax, j in zip(axes, joints):
        plot_data = [rom[j][g] for g in GROUP_ORDER]
        colors    = [GROUP_COLORS[g] for g in GROUP_ORDER]
        bp = ax.boxplot(
            plot_data, labels=GROUP_ORDER,
            patch_artist=True,
            medianprops={"color": "black", "lw": 2},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for k, (gd, grp) in enumerate(zip(plot_data, GROUP_ORDER)):
            x = np.random.normal(k + 1, 0.06, size=len(gd))
            ax.scatter(x, gd, alpha=0.5, s=18, color=GROUP_COLORS[grp], zorder=3)
        ax.set_title(j.replace("_", " "), fontsize=10)
        ax.set_ylabel("ROM (°)")
        ax.grid(True, alpha=0.25, axis="y")

    plt.suptitle("Joint ROM by Group (mean cycle per file)", fontsize=12)
    plt.tight_layout()
    _save(fig, out_dir, "rom_by_group.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — Angle trajectory for one sample file (all 6 joints overlaid)
# ──────────────────────────────────────────────────────────────────────────────

def fig_angle_trajectories(npz_path: Path, out_dir: Path) -> None:
    """Time-series of all joint angles for one recording after cleaning."""
    result = run_pipeline(npz_path)
    fps  = float(result["fps"])
    T    = result["angle_R_knee"].shape[0]
    time = np.arange(T) / fps

    joints = ["R_knee", "L_knee", "R_hip", "L_hip", "R_ankle", "L_ankle"]
    colors = ["#e74c3c", "#c0392b", "#3498db", "#2980b9", "#e67e22", "#d35400"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, j, col in zip(axes, joints, colors):
        angle = result[f"angle_{j}"]
        ax.plot(time, angle, color=col, lw=0.9, alpha=0.8)
        ax.set_ylabel("Angle (°)")
        ax.set_title(j.replace("_", " "))
        ax.grid(True, alpha=0.25)

        # Mark HS events
        side = j[0]
        for fr in result.get(f"event_{side}_HS", []):
            ax.axvline(fr / fps, color="green", lw=0.7, alpha=0.5)

    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(f"Joint angles after preprocessing — {npz_path.stem}", fontsize=11)
    plt.tight_layout()
    _save(fig, out_dir, f"angles_{npz_path.stem}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        default="KOA/EL/001_KOA_01_EL",
        help="Stem relative to data/processed/ for per-file plots",
    )
    args = parser.parse_args()

    out_dir = OUTPUT_DIR

    # Per-file diagnostics
    sample_path = PROCESSED_DIR / (args.sample + "_keypoints.npz")
    if sample_path.exists():
        print(f"\n[Per-file plots] {sample_path.name}")
        fig_quality_heatmap(sample_path, out_dir)
        fig_raw_vs_filtered(sample_path, out_dir)
        fig_angle_trajectories(sample_path, out_dir)
    else:
        print(f"  Sample not found: {sample_path}")

    # Group-level comparisons (require cleaned files)
    print("\n[Group comparison plots]")
    for joint in ["R_knee", "L_knee", "R_hip", "L_hip"]:
        fig_gait_cycles_by_group(out_dir, joint)
    fig_spatiotemporal_boxplots(out_dir)
    fig_rom_by_group(out_dir)

    print("\nAll figures saved to", out_dir)


if __name__ == "__main__":
    main()