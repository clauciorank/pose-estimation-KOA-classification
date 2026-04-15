"""
Visualization utilities for extracted gait keypoints.

Supports:
  - Skeleton overlay on video frames
  - Time-series plots of joint angles / trajectories
  - Side-by-side group comparison plots
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Body25 skeleton connectivity (OpenPose)
SKELETON_PAIRS = [
    (1, 0), (1, 2), (2, 3), (3, 4),    # neck -> nose, right arm
    (1, 5), (5, 6), (6, 7),             # left arm
    (1, 8),                             # neck -> mid-hip
    (8, 9), (9, 10), (10, 11),          # right leg
    (11, 22), (11, 24),                 # right foot
    (8, 12), (12, 13), (13, 14),        # left leg
    (14, 19), (14, 21),                 # left foot
    (0, 15), (15, 17),                  # right eye/ear
    (0, 16), (16, 18),                  # left eye/ear
]

JOINT_COLOR = (0, 255, 0)
BONE_COLOR = (255, 128, 0)
CONFIDENCE_THRESHOLD = 0.1


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Overlay Body25 skeleton on a BGR frame.

    Args:
        frame: BGR image (H, W, 3).
        keypoints: (25, 3) array of (x, y, confidence).
        scale: resize output by this factor.

    Returns:
        Annotated BGR frame.
    """
    img = frame.copy()
    h, w = img.shape[:2]

    for a, b in SKELETON_PAIRS:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca < CONFIDENCE_THRESHOLD or cb < CONFIDENCE_THRESHOLD:
            continue
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), BONE_COLOR, 2)

    for x, y, conf in keypoints:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        cv2.circle(img, (int(x), int(y)), 4, JOINT_COLOR, -1)

    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def render_video_with_skeleton(
    video_path: Path,
    keypoints_path: Path,
    output_path: Path,
    scale: float = 1.0,
) -> None:
    """Write a new video with skeleton overlay."""
    data = np.load(keypoints_path, allow_pickle=True)
    keypoints = data["keypoints"]  # (T, 25, 3)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i, kp in enumerate(keypoints):
        ret, frame = cap.read()
        if not ret:
            break
        annotated = draw_skeleton(frame, kp, scale=scale)
        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"Saved: {output_path}")


def plot_joint_trajectory(
    keypoints_path: Path,
    joint_indices: list[int],
    joint_names: list[str] | None = None,
    axis: str = "y",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot joint coordinate over time for one or more joints.

    Args:
        keypoints_path: path to .npz produced by extract_keypoints.
        joint_indices: list of Body25 joint indices to plot.
        axis: 'x' or 'y' coordinate.
        ax: existing Axes to draw on; created if None.
    """
    data = np.load(keypoints_path, allow_pickle=True)
    kp = data["keypoints"]  # (T, 25, 3)
    fps = float(data["fps"])
    stored_names = list(data["joint_names"])

    ax = ax or plt.subplots(1, 1, figsize=(12, 4))[1]
    ax_idx = 0 if axis == "x" else 1
    time = np.arange(len(kp)) / fps

    for ji in joint_indices:
        label = (joint_names or stored_names)[ji] if joint_names else stored_names[ji]
        conf = kp[:, ji, 2]
        values = np.where(conf > CONFIDENCE_THRESHOLD, kp[:, ji, ax_idx], np.nan)
        ax.plot(time, values, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{axis.upper()} coordinate (px)")
    ax.set_title(title or keypoints_path.stem)
    ax.legend(fontsize=8)
    return ax


def compare_groups(
    npz_files: dict[str, list[Path]],
    joint_index: int,
    axis: str = "y",
    title: str | None = None,
) -> plt.Figure:
    """Box-plot comparison of a joint trajectory metric across groups.

    Args:
        npz_files: {'KOA': [path, ...], 'PD': [...], 'NM': [...]}
        joint_index: Body25 joint to compare.
        axis: 'x' or 'y'.
    """
    import pandas as pd
    import seaborn as sns

    records = []
    ax_idx = 0 if axis == "x" else 1

    for group, paths in npz_files.items():
        for p in paths:
            data = np.load(p, allow_pickle=True)
            kp = data["keypoints"]
            conf = kp[:, joint_index, 2]
            values = kp[conf > CONFIDENCE_THRESHOLD, joint_index, ax_idx]
            if len(values) == 0:
                continue
            records.append({"group": group, "range": float(values.max() - values.min())})

    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="group", y="range", ax=ax, order=["NM", "KOA", "PD"])
    ax.set_title(title or f"Joint {joint_index} {axis.upper()} range by group")
    ax.set_ylabel("Range (px)")
    return fig
