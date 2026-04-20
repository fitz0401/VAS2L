from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Deque, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from VA2L.utils.gripper_tracker import GripperBBoxTracker, GripperDetection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO gripper tracker demo with overlay saving.")
    parser.add_argument("--source", type=str, choices=["folder", "realsense"], default="folder")
    parser.add_argument("--video-dir", type=Path, default=Path("dataset/robotiq_insert_tube/color"))
    parser.add_argument("--model-path", type=Path, default=Path("ckpts/yolo_best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--show", action="store_true", help="Show live overlay window")
    parser.add_argument("--save-overlay", action="store_true", default=True, help="Save overlay images (default: on)")
    parser.add_argument("--no-save-overlay", action="store_false", dest="save_overlay", help="Disable overlay saving")
    parser.add_argument("--overlay-save-dir", type=Path, default=Path("dataset/robotiq_gripper_tracking"))
    parser.add_argument("--device-id", type=str, default="")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser


def _frame_paths(video_dir: Path) -> List[Path]:
    frames = sorted(list(video_dir.glob("*.png")) + list(video_dir.glob("*.jpg")))
    if not frames:
        raise FileNotFoundError(f"No frames found in: {video_dir}")
    return frames


def _iter_folder_frames(video_dir: Path) -> Iterator[Tuple[Path, np.ndarray]]:
    for frame_path in _frame_paths(video_dir):
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        yield frame_path, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _draw_and_save(
    tracker: GripperBBoxTracker,
    frame_rgb: np.ndarray,
    detection: Optional[GripperDetection],
    history: Deque[Tuple[float, Tuple[int, int]]],
    save_path: Optional[Path],
) -> np.ndarray:
    centers = [center for _, center in history]
    overlay = tracker.draw_overlay(frame_rgb, detection, centers)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay


def _export_video_from_images(image_paths: List[Path], output_path: Path, fps: float) -> bool:
    valid_paths = [path for path in image_paths if path.exists()]
    if not valid_paths:
        return False

    first_frame = cv2.imread(str(valid_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        return False

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (width, height),
    )
    if not writer.isOpened():
        return False

    try:
        for image_path in valid_paths:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    return output_path.exists() and output_path.stat().st_size > 0


def _cleanup_images(image_paths: List[Path]) -> int:
    deleted = 0
    for image_path in image_paths:
        try:
            image_path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def _finalize_saved_overlays(image_paths: List[Path], overlay_save_dir: Path, fps: float) -> None:
    if not image_paths:
        print("No overlays were saved in this run.")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = overlay_save_dir / f"overlay_{timestamp}.mp4"
    if _export_video_from_images(image_paths, video_path, fps):
        deleted_count = _cleanup_images(image_paths)
        print(f"Saved overlay video: {video_path}")
        print(f"Deleted {deleted_count} overlay images from this run.")
        return

    print("Failed to export overlay video; keeping images for debugging.")


def _run_folder(args: argparse.Namespace, tracker: GripperBBoxTracker) -> None:
    history: Deque[Tuple[float, Tuple[int, int]]] = deque()
    saved_image_paths: List[Path] = []
    dt = 1.0 / max(1e-6, args.fps)
    print("Folder tracking started. Press q to stop if --show is enabled.")

    for frame_path, frame_rgb in _iter_folder_frames(args.video_dir):
        t0 = time.time()
        detection = tracker.detect(frame_rgb)
        if detection is not None:
            history.append((t0, detection.center))
        cutoff = t0 - 2.0
        while history and history[0][0] < cutoff:
            history.popleft()

        center_text = "none" if detection is None else f"({detection.center[0]}, {detection.center[1]})"
        bbox_text = "none" if detection is None else f"{detection.bbox}"
        print(f"{frame_path.name}: bbox={bbox_text} center={center_text}")

        save_path = None
        if args.save_overlay:
            save_path = args.overlay_save_dir / f"{frame_path.stem}.png"
        overlay = _draw_and_save(tracker, frame_rgb, detection, history, save_path)
        if save_path is not None and save_path.exists():
            saved_image_paths.append(save_path)

        if args.show:
            cv2.imshow("gripper-tracker", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        elapsed = time.time() - t0
        sleep_sec = dt - elapsed
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if args.show:
        cv2.destroyAllWindows()
    if args.save_overlay:
        _finalize_saved_overlays(saved_image_paths, args.overlay_save_dir, args.fps)


def _run_realsense(args: argparse.Namespace, tracker: GripperBBoxTracker) -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError("pyrealsense2 is required for realsense mode") from exc

    history: Deque[Tuple[float, Tuple[int, int]]] = deque()
    saved_image_paths: List[Path] = []
    pipeline = rs.pipeline()
    config = rs.config()
    if args.device_id:
        config.enable_device(args.device_id)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, int(round(args.fps)))
    pipeline.start(config)

    print("Realtime tracking started. Press Ctrl+C to stop.")
    try:
        while True:
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue

            t0 = time.time()
            bgr = np.asanyarray(color_frame.get_data())
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detection = tracker.detect(frame_rgb)
            if detection is not None:
                history.append((t0, detection.center))
            cutoff = t0 - 2.0
            while history and history[0][0] < cutoff:
                history.popleft()

            center_text = "none" if detection is None else f"({detection.center[0]}, {detection.center[1]})"
            bbox_text = "none" if detection is None else f"{detection.bbox}"
            print(f"bbox={bbox_text} center={center_text}")

            save_path = None
            if args.save_overlay:
                frame_idx = int(round(t0 * 1000.0))
                save_path = args.overlay_save_dir / f"{frame_idx:012d}.png"
            overlay = _draw_and_save(tracker, frame_rgb, detection, history, save_path)
            if save_path is not None and save_path.exists():
                saved_image_paths.append(save_path)

            if args.show:
                cv2.imshow("gripper-tracker", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()
        if args.save_overlay:
            _finalize_saved_overlays(saved_image_paths, args.overlay_save_dir, args.fps)


def main() -> None:
    args = _build_parser().parse_args()
    tracker = GripperBBoxTracker(
        model_path=args.model_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    print(f"Loaded gripper tracker model: {tracker.model_path}")

    if args.source == "folder":
        _run_folder(args, tracker)
    else:
        _run_realsense(args, tracker)


if __name__ == "__main__":
    main()
