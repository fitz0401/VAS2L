import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from groundingdino.util.inference import annotate

from mask_generator import MaskGenerator
from som_generator import SoMGenerator


def _resolve_frame_path(demo_dir: Path, frame_idx: int) -> Path:
	color_dir = demo_dir / "color"
	frame_paths = sorted(color_dir.glob("*.png"))
	if not frame_paths:
		raise FileNotFoundError(f"No PNG images found in {color_dir}")
	if frame_idx < 0 or frame_idx >= len(frame_paths):
		raise IndexError(f"frame_idx={frame_idx} out of range [0, {len(frame_paths) - 1}]")
	return frame_paths[frame_idx]


def _build_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
	overlay = image_rgb.copy()
	alpha = 0.45
	color = np.array([60, 220, 60], dtype=np.uint8)
	masked = mask.astype(bool)
	overlay[masked] = ((1.0 - alpha) * overlay[masked] + alpha * color).astype(np.uint8)

	# Draw boundary for clearer visualization.
	contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
	return overlay


def _save_layout(original_rgb: np.ndarray, bbox_rgb: np.ndarray, mask_rgb: np.ndarray, out_path: Path) -> None:
	title_h = 36
	panels = [original_rgb, bbox_rgb, mask_rgb]
	labels = ["Original", "Bounding Box", "Mask Overlay"]
	labeled_panels = []

	for panel, label in zip(panels, labels):
		header = np.full((title_h, panel.shape[1], 3), 20, dtype=np.uint8)
		cv2.putText(
			header,
			label,
			(10, 24),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)
		labeled_panels.append(np.vstack([header, panel]))

	layout = np.hstack(labeled_panels)
	Image.fromarray(layout).save(out_path)


def main() -> None:
	parser = argparse.ArgumentParser(description="Extract Franka gripper mask at a specified frame")
	parser.add_argument("--demo_dir", type=str, required=True, help="Demo directory containing color/")
	parser.add_argument("--frame_idx", type=int, default=0, help="Index in sorted color/*.png list")
	parser.add_argument("--text", type=str, default="franka gripper", help="GroundingDINO text prompt")
	parser.add_argument(
		"--method",
		type=str,
		choices=["mask", "som"],
		default="mask",
		help="Visualization method: mask (GroundingDINO+SAM) or som (Set-of-Mask)",
	)
	parser.add_argument("--som_alpha", type=float, default=0.1, help="SoM mask alpha")
	parser.add_argument("--som_level", type=int, default=2, help="Semantic-SAM level in [1,6], default 2")
	parser.add_argument(
		"--som_label_mode",
		type=str,
		choices=["Number", "Alphabet"],
		default="Number",
		help="SoM mark label mode",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default=None,
		help="Output directory, default is <demo_dir>/debug/gripper_test",
	)
	args = parser.parse_args()

	t_total_start = time.perf_counter()
	demo_dir = Path(args.demo_dir).expanduser().resolve()
	out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else demo_dir / "debug" / "gripper_test"
	out_dir.mkdir(parents=True, exist_ok=True)

	frame_path = _resolve_frame_path(demo_dir, args.frame_idx)
	image_pil = Image.open(frame_path).convert("RGB")
	image_rgb = np.asarray(image_pil)
	frame_stem = frame_path.stem

	if args.method == "som":
		t_init_start = time.perf_counter()
		try:
			som = SoMGenerator()
		except Exception as exc:
			raise RuntimeError(
				"Failed to initialize SoMGenerator. "
				"Please verify SoM dependencies and checkpoints are correctly installed."
			) from exc
		t_init = time.perf_counter() - t_init_start

		t_som_start = time.perf_counter()
		result = som.generate(
			image=image_pil,
			alpha=args.som_alpha,
			semsam_level=args.som_level,
			label_mode=args.som_label_mode,
			anno_mode=["Mask", "Mark"],
		)
		t_som = time.perf_counter() - t_som_start

		som_image = result["image"]
		som_masks = result["masks"]
		som_meta = result["metadata"]

		som_image_path = out_dir / f"{frame_stem}_som.png"
		som_masks_path = out_dir / f"{frame_stem}_som_masks.npz"
		stats_path = out_dir / f"{frame_stem}_som_stats.json"

		som_image.save(som_image_path)
		np.savez_compressed(som_masks_path, masks=som_masks)

		t_total = time.perf_counter() - t_total_start
		stats = {
			"method": "som",
			"demo_dir": str(demo_dir),
			"frame_path": str(frame_path),
			"frame_idx": args.frame_idx,
			"som": {
				"alpha": args.som_alpha,
				"model": "semantic-sam",
				"som_level": args.som_level,
				"label_mode": args.som_label_mode,
				"mask_count": int(som_masks.shape[0]),
				"used_params": result.get("used_params", {}),
				"metadata": som_meta,
			},
			"timing_sec": {
				"model_init": t_init,
				"som_inference": t_som,
				"total": t_total,
			},
			"outputs": {
				"som_image": str(som_image_path),
				"som_masks": str(som_masks_path),
			},
		}
		stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

		print("==== SoM Generation Complete ====")
		print(f"Frame: {frame_path.name} (idx={args.frame_idx})")
		print(f"Model: {result.get('used_params', {}).get('model_name', 'unknown')}")
		print(f"SoM masks: {som_masks.shape[0]}")
		print("Timing (sec):")
		print(f"  model_init    : {t_init:.3f}")
		print(f"  som_inference : {t_som:.3f}")
		print(f"  total         : {t_total:.3f}")
		print("Saved outputs:")
		print(f"  som image : {som_image_path}")
		print(f"  som masks : {som_masks_path}")
		print(f"  stats     : {stats_path}")
		return

	t_init_start = time.perf_counter()
	generator = MaskGenerator()
	t_init = time.perf_counter() - t_init_start

	t_bbox_start = time.perf_counter()
	boxes, logits, phrases = generator.get_scene_object_bboxes(
		image=image_pil,
		texts=[args.text],
		visualize=False,
	)
	t_bbox = time.perf_counter() - t_bbox_start

	t_mask_start = time.perf_counter()
	masks = generator.get_segmentation_masks(
		image=image_pil,
		boxes=boxes,
		logits=logits,
		phrases=phrases,
		visualize=False,
	)
	t_mask = time.perf_counter() - t_mask_start

	if masks.shape[0] < 1:
		raise RuntimeError("No segmentation mask returned")

	mask = masks[0].astype(np.uint8)
	bbox_annotated = annotate(image_source=image_rgb, boxes=boxes, logits=logits, phrases=phrases)[..., ::-1]
	mask_overlay = _build_mask_overlay(image_rgb, mask)

	bbox_path = out_dir / f"{frame_stem}_bbox.png"
	mask_path = out_dir / f"{frame_stem}_mask.png"
	overlay_path = out_dir / f"{frame_stem}_mask_overlay.png"
	layout_path = out_dir / f"{frame_stem}_layout.png"
	stats_path = out_dir / f"{frame_stem}_stats.json"

	Image.fromarray(bbox_annotated).save(bbox_path)
	Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
	Image.fromarray(mask_overlay).save(overlay_path)
	_save_layout(image_rgb, bbox_annotated, mask_overlay, layout_path)

	t_total = time.perf_counter() - t_total_start
	stats = {
		"demo_dir": str(demo_dir),
		"frame_path": str(frame_path),
		"frame_idx": args.frame_idx,
		"prompt": args.text,
		"phrases": phrases,
		"score": float(logits[0].item()) if logits.numel() > 0 else None,
		"mask_pixels": int(mask.sum()),
		"mask_ratio": float(mask.mean()),
		"timing_sec": {
			"model_init": t_init,
			"bbox_detection": t_bbox,
			"mask_segmentation": t_mask,
			"total": t_total,
		},
		"outputs": {
			"bbox_image": str(bbox_path),
			"mask_image": str(mask_path),
			"mask_overlay_image": str(overlay_path),
			"layout_image": str(layout_path),
		},
	}
	stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

	print("==== Gripper Mask Extraction Complete ====")
	print(f"Frame: {frame_path.name} (idx={args.frame_idx})")
	print(f"Prompt: {args.text}")
	print(f"Detected phrase: {phrases[0] if phrases else 'N/A'}")
	print(f"Detection score: {stats['score']}")
	print(f"Mask pixels: {stats['mask_pixels']} ({stats['mask_ratio'] * 100:.2f}%)")
	print("Timing (sec):")
	print(f"  model_init        : {t_init:.3f}")
	print(f"  bbox_detection    : {t_bbox:.3f}")
	print(f"  mask_segmentation : {t_mask:.3f}")
	print(f"  total             : {t_total:.3f}")
	print("Saved outputs:")
	print(f"  bbox    : {bbox_path}")
	print(f"  mask    : {mask_path}")
	print(f"  overlay : {overlay_path}")
	print(f"  layout  : {layout_path}")
	print(f"  stats   : {stats_path}")


if __name__ == "__main__":
	main()
