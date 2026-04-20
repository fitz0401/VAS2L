from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Collect RGB frames from RealSense for gripper training.")
	parser.add_argument("--out-root", type=Path, default=Path("dataset"), help="Root directory to save demo folders.")
	parser.add_argument(
		"--demo-name",
		type=str,
		default=None,
		help="Demo folder name. If omitted, use demo_YYYYmmdd_HHMMSS.",
	)
	parser.add_argument("--device-id", type=str, default="", help="Optional RealSense serial number.")
	parser.add_argument("--width", type=int, default=1280)
	parser.add_argument("--height", type=int, default=720)
	parser.add_argument("--camera-fps", type=int, default=30, help="RealSense stream FPS.")
	parser.add_argument("--save-hz", type=float, default=2.0, help="Frame saving frequency.")
	parser.add_argument("--show", action="store_true", help="Show preview window and press q/ESC to stop.")
	return parser


def _prepare_dirs(out_root: Path, demo_name: str | None) -> tuple[Path, Path]:
	if demo_name is None:
		demo_name = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	demo_dir = out_root / demo_name
	color_dir = demo_dir / "color"
	color_dir.mkdir(parents=True, exist_ok=True)
	return demo_dir, color_dir


def main() -> None:
	args = _build_parser().parse_args()

	if args.save_hz <= 0:
		raise ValueError("--save-hz must be > 0")

	demo_dir, color_dir = _prepare_dirs(args.out_root, args.demo_name)
	save_interval = 1.0 / float(args.save_hz)

	pipeline = rs.pipeline()
	config = rs.config()
	if args.device_id:
		config.enable_device(args.device_id)
	config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.camera_fps)

	print(f"Saving RGB sequence to: {color_dir}")
	print(f"Save frequency: {args.save_hz:.3f} Hz (interval: {save_interval:.3f} s)")
	if args.show:
		print("Press 'q' or ESC in the preview window to stop.")
	else:
		print("Press Ctrl+C to stop.")

	frame_idx = 0
	last_save_t = -1e9
	start_t = time.time()

	pipeline.start(config)
	try:
		while True:
			frameset = pipeline.wait_for_frames()
			color_frame = frameset.get_color_frame()
			if not color_frame:
				continue

			now_t = time.time()
			bgr = np.asanyarray(color_frame.get_data())

			if (now_t - last_save_t) >= save_interval:
				out_path = color_dir / f"{frame_idx:04d}.png"
				cv2.imwrite(str(out_path), bgr)
				print(f"[{frame_idx:04d}] saved {out_path.name}")
				frame_idx += 1
				last_save_t = now_t

			if args.show:
				vis = bgr.copy()
				cv2.putText(
					vis,
					f"saved: {frame_idx}  hz: {args.save_hz:.2f}",
					(10, 28),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.8,
					(0, 255, 0),
					2,
					cv2.LINE_AA,
				)
				cv2.imshow("collect_gripper_video", vis)
				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord("q")):
					break

	except KeyboardInterrupt:
		print("Interrupted by user.")
	finally:
		pipeline.stop()
		if args.show:
			cv2.destroyAllWindows()

		end_t = time.time()
		meta = {
			"demo_dir": str(demo_dir),
			"color_dir": str(color_dir),
			"frame_count": frame_idx,
			"elapsed_sec": round(end_t - start_t, 3),
			"save_hz": float(args.save_hz),
			"camera": {
				"device_id": args.device_id,
				"width": int(args.width),
				"height": int(args.height),
				"camera_fps": int(args.camera_fps),
			},
		}
		(demo_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
		print(f"Done. Saved {frame_idx} frames to {color_dir}")


if __name__ == "__main__":
	main()
