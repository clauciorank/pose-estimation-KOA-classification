"""
Batch preprocessing runner.

Processes all .npz keypoint files under data/processed/ and saves cleaned
outputs to data/cleaned/, mirroring the same directory structure.

Usage:
    python3 src/analysis/run_preprocess.py [--group KOA] [--dry-run]

Output per file: data/cleaned/[GROUP]/[STAGE]/[stem]_cleaned.npz
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

# Ensure src is importable when run from project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.preprocess import run_pipeline

PROCESSED_DIR = Path("data/processed")
CLEANED_DIR   = Path("data/cleaned")


def iter_npz(group_filter: str | None = None):
    """Yield all keypoint .npz paths under data/processed/."""
    for p in sorted(PROCESSED_DIR.rglob("*_keypoints.npz")):
        if group_filter and p.parts[2] != group_filter:
            continue
        yield p


def main():
    parser = argparse.ArgumentParser(description="Batch gait preprocessing")
    parser.add_argument("--group", help="Process only this group (KOA, PD, NM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without processing")
    args = parser.parse_args()

    paths = list(iter_npz(args.group))
    print(f"Found {len(paths)} files to process.")

    if args.dry_run:
        for p in paths:
            print(" ", p)
        return

    ok = err = skipped = 0
    manifest = []

    for npz_path in paths:
        # Mirror the relative path under data/cleaned/
        rel = npz_path.relative_to(PROCESSED_DIR)
        out_path = CLEANED_DIR / rel.parent / (npz_path.stem.replace("_keypoints", "_cleaned") + ".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = run_pipeline(npz_path)

            # Validate events before saving
            ev_val = result.get("event_validation", {})
            if isinstance(ev_val, np.ndarray):
                ev_val = ev_val.item() if ev_val.ndim == 0 else {}

            r_valid = ev_val.get("R", {}).get("valid", False) if ev_val else False
            l_valid = ev_val.get("L", {}).get("valid", False) if ev_val else False

            if not r_valid and not l_valid:
                skipped += 1
                print(f"  [SKIP] {npz_path.name} — no valid gait events detected")
                status = "skipped_no_events"
            else:
                np.savez_compressed(out_path, **result)
                ok += 1
                status = "ok"
                n_r = len(result.get("event_R_HS", []))
                n_l = len(result.get("event_L_HS", []))
                spatio = result.get("spatiotemporal", {})
                if isinstance(spatio, np.ndarray):
                    spatio = spatio.item() if spatio.ndim == 0 else {}
                cad = spatio.get("cadence_steps_per_min", None)
                cad_str = f"{cad:.1f} spm" if cad else "—"
                print(f"  [OK]   {npz_path.name}  R_HS={n_r}  L_HS={n_l}  cadence={cad_str}")

            manifest.append({
                "source": str(npz_path),
                "output": str(out_path),
                "status": status,
            })

        except Exception as exc:
            err += 1
            print(f"  [ERR]  {npz_path.name}: {exc}")
            traceback.print_exc()
            manifest.append({
                "source": str(npz_path),
                "output": str(out_path),
                "status": f"error: {exc}",
            })

    # Save processing manifest
    manifest_path = CLEANED_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone — OK: {ok}  Skipped: {skipped}  Errors: {err}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()