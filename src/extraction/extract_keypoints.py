"""
Batch keypoint extraction from gait analysis videos using OpenPose.

Processes all videos under /data/videos and saves per-frame keypoints
as JSON and a consolidated NumPy array per video to /data/processed.

All recordings are sagittal view. File naming:
  KOA/PD: [SUBJECT]_[GROUP]_[SIDE]_[STAGE].MOV  e.g. 001_KOA_01_SV.MOV
  NM:     [SUBJECT]_[GROUP]_[SIDE].MOV            e.g. 001_NM_01.MOV  (no stage — healthy)

  SIDE:  01 = right side, 02 = left side
  STAGE: KOA: EL=Early, MD=Moderate, SV=Severe
         PD:  ML=Mild,  MD=Moderate, SV=Severe
         NM:  (none)
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# OpenPose Python bindings (available inside the Docker image)
try:
    sys.path.insert(0, "/openpose/build/python")
    from openpose import pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    print("WARNING: OpenPose not found — running in dry-run mode.")

VIDEO_DIR = Path("/data/videos")
OUTPUT_DIR = Path("/data/processed")
OPENPOSE_MODELS = "/openpose/models"

# Body25 joint names (OpenPose default model)
BODY25_JOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip",
    "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel",
]


def init_openpose(net_resolution: str = "368x-1") -> op.WrapperPython:
    params = {
        "model_folder": OPENPOSE_MODELS,
        "net_resolution": net_resolution,
        "number_people_max": 1,  # single-subject gait recordings
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper


# Lower-body joints used for gait analysis — the only ones that matter for this project.
# Head joints (0, 15-18) are blurred in all videos; Neck (1) may be partially affected.
GAIT_JOINTS = [1, 8, 9, 10, 11, 12, 13, 14, 19, 21, 22, 24]  # Neck, hips, knees, ankles, toes, heels
DETECTION_WARN_THRESHOLD = 0.30  # warn if >30% of frames have no lower-body detection


def remove_anonymization_oval(frame: np.ndarray) -> np.ndarray:
    """Replace the grey anonymization oval with dark green background.

    NM videos have a large grey filled ellipse over the head/neck that prevents
    OpenPose from anchoring the skeleton.  Detect achromatic mid-brightness
    pixels in the top half and fill with a dark green matching the backdrop.
    """
    out = np.ascontiguousarray(frame, dtype=np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([0,   0,  60], dtype=np.uint8),
                       np.array([180, 40, 220], dtype=np.uint8))
    half = out.shape[0] // 2
    mask[half:] = 0
    if int(mask.sum()) < 500:
        return out
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)
    out[mask > 0] = [40, 80, 40]  # dark green matching backdrop
    return np.ascontiguousarray(out, dtype=np.uint8)


def extract_video(video_path: Path, output_dir: Path, wrapper,
                  inpaint: bool = False) -> Path:
    """Extract per-frame keypoints from a single video.

    Returns the path to the saved .npz file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / (video_path.stem + "_keypoints.npz")

    if npz_path.exists():
        return npz_path  # already processed

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    all_keypoints = []  # shape: (T, 25, 3)  — x, y, confidence
    missed_frames = 0

    with tqdm(total=total_frames, desc=video_path.name, unit="frame", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            datum = op.Datum()  # fresh datum per frame avoids state corruption
            datum.cvInputData = remove_anonymization_oval(frame) if inpaint else np.ascontiguousarray(frame, dtype=np.uint8)
            wrapper.emplaceAndPop(op.VectorDatum([datum]))
            kp = datum.poseKeypoints  # ndarray (1, 25, 3) or None
            if kp is not None and len(kp) > 0:
                all_keypoints.append(kp[0])  # take first person
            else:
                all_keypoints.append(np.zeros((25, 3), dtype=np.float32))
                missed_frames += 1
            pbar.update(1)

    cap.release()

    keypoints_array = np.array(all_keypoints, dtype=np.float32)  # (T, 25, 3)

    # Warn if too many frames lost — blurred heads can cause full-skeleton detection failures
    if total_frames > 0:
        miss_rate = missed_frames / total_frames
        if miss_rate > DETECTION_WARN_THRESHOLD:
            print(f"  WARNING {video_path.name}: {miss_rate:.0%} frames with no detection "
                  f"({missed_frames}/{total_frames}) — check video quality")

    np.savez_compressed(
        npz_path,
        keypoints=keypoints_array,
        joint_names=BODY25_JOINTS,
        fps=fps,
        source=str(video_path),
        missed_frames=missed_frames,
        total_frames=total_frames,
    )
    return npz_path


def parse_filename(stem: str) -> dict:
    """Parse subject/group/side/stage from filename stem.

    KOA/PD: '001_KOA_01_SV' -> {'subject': '001', 'group': 'KOA', 'side': '01', 'stage': 'SV'}
    NM:     '001_NM_01'      -> {'subject': '001', 'group': 'NM',  'side': '01', 'stage': None}
    """
    parts = stem.split("_")
    if len(parts) == 3:
        # NM files — healthy controls have no disease stage
        return {"subject": parts[0], "group": parts[1], "side": parts[2], "stage": None}
    if len(parts) >= 4:
        return {"subject": parts[0], "group": parts[1], "side": parts[2], "stage": parts[3]}
    return {"subject": "UNK", "group": "UNK", "side": "UNK", "stage": None}


def main():
    parser = argparse.ArgumentParser(description="Batch OpenPose keypoint extraction")
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--group", choices=["KOA", "PD", "NM", "all"], default="all")
    parser.add_argument("--stage", default=None,
                        help="Disease stage suffix to filter (KOA: EL/MD/SV, PD: ML/MD/SV). "
                             "Omit to process all stages.")
    parser.add_argument("--net-resolution", default="368x-1")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N videos (useful for testing)")
    parser.add_argument("--inpaint", action="store_true",
                        help="Remove grey anonymization oval before detection (use for NM videos)")
    args = parser.parse_args()

    videos = sorted(args.video_dir.rglob("*.MOV")) + sorted(args.video_dir.rglob("*.mp4"))

    if args.group != "all":
        videos = [v for v in videos if f"_{args.group}_" in v.stem or f"/{args.group}/" in str(v)]
    if args.stage:
        videos = [v for v in videos if v.stem.endswith(f"_{args.stage}")]

    if args.sample:
        import random
        random.seed(42)
        videos = random.sample(videos, min(args.sample, len(videos)))
        videos = sorted(videos)

    print(f"Found {len(videos)} video(s) to process.")
    if not videos:
        return

    wrapper = init_openpose(args.net_resolution) if OPENPOSE_AVAILABLE else None

    manifest = []
    for video in tqdm(videos, desc="Videos", unit="video"):
        meta = parse_filename(video.stem)
        stage_dir = meta["stage"] if meta["stage"] else "NM"
        out_subdir = args.output_dir / meta["group"] / stage_dir
        if OPENPOSE_AVAILABLE:
            npz_path = extract_video(video, out_subdir, wrapper, inpaint=args.inpaint)
        else:
            npz_path = out_subdir / (video.stem + "_keypoints.npz")
        entry = {**meta, "source": str(video), "keypoints": str(npz_path)}
        if OPENPOSE_AVAILABLE:
            d = np.load(npz_path)
            entry["missed_frames"] = int(d["missed_frames"])
            entry["total_frames"] = int(d["total_frames"])
        manifest.append(entry)

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
