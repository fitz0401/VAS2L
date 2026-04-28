from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def first_frame_hash(video_path: Path) -> str | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return hashlib.sha256(frame.tobytes()).hexdigest()


def scan_duplicates(video_dir: Path, pattern: str = "*.mp4") -> Tuple[List[Path], List[Path]]:
    files = sorted(video_dir.glob(pattern))
    seen: Dict[str, Path] = {}
    keep: List[Path] = []
    duplicates: List[Path] = []

    for f in files:
        h = first_frame_hash(f)
        if h is None:
            continue
        if h not in seen:
            seen[h] = f
            keep.append(f)
        else:
            duplicates.append(f)

    return keep, duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter duplicate videos by identical first frame")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="dataset/custom_droid_dataset/videos/chunk-000/wrist_image_left",
        help="Directory that contains wrist videos",
    )
    parser.add_argument("--pattern", type=str, default="*.mp4", help="Video filename pattern")
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Actually delete duplicates. If omitted, only print duplicates.",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir).resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"Video dir not found: {video_dir}")

    keep, duplicates = scan_duplicates(video_dir, args.pattern)

    print(f"[INFO] Total videos: {len(keep) + len(duplicates)}")
    print(f"[INFO] Keep: {len(keep)}")
    print(f"[INFO] Duplicates: {len(duplicates)}")

    if duplicates:
        print("[INFO] Duplicate files:")
        for p in duplicates:
            print(f"  - {p}")

    if args.remove and duplicates:
        for p in duplicates:
            p.unlink(missing_ok=True)
        print(f"[INFO] Removed {len(duplicates)} duplicate files")


if __name__ == "__main__":
    main()
