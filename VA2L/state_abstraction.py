from pathlib import Path
import json
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

class StateAbstraction:
	"""Prepare VLM inputs using two-frame difference-guided object segmentation."""

	def __init__(
		self,
		demo_dir: str,
		window_size: int = 15,
		object_prompt: str = "gripper",
		diff_threshold: int = 24,
		min_diff_area: int = 250,
		manipulation_backend: str = "sam",
		yolo_model_path: str = "yolo26s.pt",
	):
		self.demo_dir = Path(demo_dir)
		self.window_size = int(window_size)
		self.object_prompt = object_prompt
		self.diff_threshold = int(diff_threshold)
		self.min_diff_area = int(min_diff_area)
		self.manipulation_backend = manipulation_backend.lower()
		self.yolo_model_path = yolo_model_path
		if self.manipulation_backend not in {"sam", "yolo"}:
			raise ValueError("manipulation_backend must be 'sam' or 'yolo'.")

		self.image_paths = sorted((self.demo_dir / "color").glob("*.png"))
		if not self.image_paths:
			raise FileNotFoundError(f"No RGB images found in {self.demo_dir / 'color'}")

		# Load trajectory data
		trajectory_path = self.demo_dir / "trajectory.json"
		if not trajectory_path.exists():
			raise FileNotFoundError(f"trajectory.json not found in {self.demo_dir}")
		with open(trajectory_path, "r") as f:
			self.trajectory = json.load(f)

		self.mask_generator = None
		self.som_generator = None
		self.yolo_model = None
		if self.manipulation_backend == "yolo":
			from ultralytics import YOLO
			self.yolo_model = YOLO(self.yolo_model_path)

	def _window_indices(self, t: int) -> Tuple[int, int]:
		start = max(0, t - self.window_size)
		return start, t

	def _get_mask_generator(self):
		if self.mask_generator is None:
			from vision_module.mask_generator import MaskGenerator

			self.mask_generator = MaskGenerator()
		return self.mask_generator

	def _get_som_generator(self):
		if self.som_generator is None:
			from vision_module.som_generator import SoMGenerator

			self.som_generator = SoMGenerator()
		return self.som_generator

	def _load_rgb(self, idx: int) -> np.ndarray:
		return np.asarray(Image.open(self.image_paths[idx]).convert("RGB"))

	def _load_gripper_state(self, idx: int) -> str:
		"""Load gripper state from trajectory.json for frame index idx."""
		if idx < 0 or idx >= len(self.trajectory):
			raise IndexError(f"idx={idx} out of range [0, {len(self.trajectory) - 1}]")
		state = self.trajectory[idx].get("gripper_state", "unknown")
		return state.lower()  # Normalize to lowercase

	def _diff_bbox(self, current_rgb: np.ndarray, ref_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
		import cv2

		diff = cv2.absdiff(current_rgb, ref_rgb)
		gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
		_, binary = cv2.threshold(gray, self.diff_threshold, 255, cv2.THRESH_BINARY)
		binary = cv2.medianBlur(binary, 5)

		contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return None

		best = max(contours, key=cv2.contourArea)
		area = float(cv2.contourArea(best))
		if area < self.min_diff_area:
			return None

		x, y, w, h = cv2.boundingRect(best)
		return x, y, x + w, y + h

	@staticmethod
	def _bbox_to_norm_cxcywh(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float, float, float]:
		x1, y1, x2, y2 = bbox
		bw = max(1.0, float(x2 - x1))
		bh = max(1.0, float(y2 - y1))
		cx = float(x1 + x2) / 2.0
		cy = float(y1 + y2) / 2.0
		return cx / float(w), cy / float(h), bw / float(w), bh / float(h)

	def _segment_with_shared_bbox(
		self,
		rgb: np.ndarray,
		bbox: Tuple[int, int, int, int],
	) -> Optional[np.ndarray]:
		try:
			import torch

			img_pil = Image.fromarray(rgb)
			h, w = rgb.shape[:2]
			cx, cy, bw, bh = self._bbox_to_norm_cxcywh(bbox, w=w, h=h)
			boxes = torch.tensor([[cx, cy, bw, bh]], dtype=torch.float32)
			logits = torch.tensor([1.0], dtype=torch.float32)
			phrases = [self.object_prompt]
			masks = self._get_mask_generator().get_segmentation_masks(
				image=img_pil,
				boxes=boxes,
				logits=logits,
				phrases=phrases,
				visualize=False,
			)
			if masks.shape[0] < 1:
				return None
			return masks[0].astype(np.uint8)
		except Exception:
			return None

	@staticmethod
	def _bbox_center(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
		if bbox is None:
			return None
		x1, y1, x2, y2 = bbox
		return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))

	def _draw_result(
		self,
		image: Image.Image,
		bbox: Optional[Tuple[int, int, int, int]],
		start_pt: Optional[Tuple[int, int]],
		cur_pt: Optional[Tuple[int, int]],
	) -> None:
		draw = ImageDraw.Draw(image)
		if bbox is not None:
			x1, y1, x2, y2 = bbox
			draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=3)

		if start_pt is not None and cur_pt is not None:
			draw.line([start_pt, cur_pt], fill=(255, 80, 70), width=4)

			dx = cur_pt[0] - start_pt[0]
			dy = cur_pt[1] - start_pt[1]
			norm = float(np.hypot(dx, dy))
			if norm > 1e-6:
				ux, uy = dx / norm, dy / norm
				head_len = 12.0
				head_w = 7.0
				tip = np.array([cur_pt[0], cur_pt[1]], dtype=np.float64)
				base = tip - head_len * np.array([ux, uy], dtype=np.float64)
				perp = np.array([-uy, ux], dtype=np.float64)
				left = base + head_w * perp
				right = base - head_w * perp
				triangle = [tuple(tip.astype(int)), tuple(left.astype(int)), tuple(right.astype(int))]
				draw.polygon(triangle, fill=(255, 80, 70))

		for p in [start_pt, cur_pt]:
			if p is None:
				continue
			u, v = p
			draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=(255, 255, 255))

	@staticmethod
	def _resize_to_match(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
		if image.size == (target_w, target_h):
			return image
		return image.resize((target_w, target_h), Image.BICUBIC)

	@staticmethod
	def _format_yolo_label(name: str, confidence: float) -> str:
		return f"{name} {confidence:.2f}"

	def _draw_yolo_detection(
		self,
		image: Image.Image,
		bbox: Tuple[float, float, float, float],
	) -> None:
		draw = ImageDraw.Draw(image)
		x1, y1, x2, y2 = [float(v) for v in bbox]
		x1_i, y1_i, x2_i, y2_i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
		draw.rectangle((x1_i, y1_i, x2_i, y2_i), outline=(0, 255, 120), width=3)

	def _render_yolo_manipulation(self, current_rgb: np.ndarray) -> Tuple[Image.Image, dict]:
		if self.yolo_model is None:
			raise RuntimeError("YOLO model is not initialized.")

		image = Image.fromarray(current_rgb.copy())
		results = self.yolo_model.predict(source=current_rgb, verbose=False)
		if not results:
			return image, {"detections": 0, "yolo_used": True}

		result = results[0]
		boxes = getattr(result, "boxes", None)
		if boxes is None or len(boxes) == 0:
			return image, {"detections": 0, "yolo_used": True}

		names = result.names if hasattr(result, "names") else {}
		detection_count = 0
		detected_names = []
		for box in boxes:
			xyxy = box.xyxy[0].tolist()
			cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
			confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
			if isinstance(names, dict):
				name = names.get(cls_id, str(cls_id))
			elif isinstance(names, list) and 0 <= cls_id < len(names):
				name = names[cls_id]
			else:
				name = str(cls_id)
			detected_names.append(name)
			self._draw_yolo_detection(image=image, bbox=tuple(xyxy))
			detection_count += 1

		return image, {
			"detections": detection_count,
			"detected_objects": sorted(set(detected_names)),
			"yolo_used": True,
		}

	def _compose_instruction(self, gripper_state: str) -> str:
		"""Generate language instruction based on gripper state."""
		if gripper_state == "open":
			lines = [
				"Infer the robot's current action.",
				"Rules: Choose the best action from: pick, place, move, insert, remove, open, close. Keep it short. Do not explain.",
				"Hint: The white bounding box represents the area where motion occurred in the past 2 seconds. Grippr state: OPEN.",
				"Examples: `pick the object`, `open the drawer`"
			]
		elif gripper_state == "closed":
			lines = [
				"Infer the robot's current action.",
				"Template: <Action> <Object> [Relation] <Target>",
				"Rules: Choose the best action from: pick, place, move, insert, remove, open, close. Choose the best relation from: on, into, next to, from. Keep it short. Do not explain.",
				"Hint: The bounding box outlines the objects. Consider the relationships between the objects. Grippr state: CLOSED.",
				"Examples: `insert the pen into the mug`, `place the box next to the bottle`"
			]
		return "\n".join(lines)

	def prepare_vlm_inputs(self, t: int) -> Tuple[Image.Image, str]:
		if t < 0 or t >= len(self.image_paths):
			raise IndexError(f"t={t} out of range [0, {len(self.image_paths) - 1}]")

		idx_old, idx_cur = self._window_indices(t)
		current_rgb = self._load_rgb(t)
		old_rgb = self._load_rgb(idx_old)
		gripper_state = self._load_gripper_state(t)

		image = Image.fromarray(current_rgb.copy())

		start_time = time.perf_counter()

		if gripper_state == "open":
			# Gripper OPEN: keep original behavior.
			shared_bbox = self._diff_bbox(current_rgb=current_rgb, ref_rgb=old_rgb)
			if self.manipulation_backend == "yolo":
				old_mask = None
				cur_mask = None
			else:
				old_mask = self._segment_with_shared_bbox(old_rgb, shared_bbox) if shared_bbox is not None else None
				cur_mask = self._segment_with_shared_bbox(current_rgb, shared_bbox) if shared_bbox is not None else None
			start_center = self._bbox_center(shared_bbox)
			cur_center = self._bbox_center(shared_bbox)
			if old_mask is not None:
				ys, xs = np.where(old_mask > 0)
				if len(xs) > 0:
					start_center = (int(round(xs.mean())), int(round(ys.mean())))
			if cur_mask is not None:
				ys, xs = np.where(cur_mask > 0)
				if len(xs) > 0:
					cur_center = (int(round(xs.mean())), int(round(ys.mean())))

			self._draw_result(image=image, bbox=shared_bbox, start_pt=start_center, cur_pt=cur_center)

			self.last_timing_stats = {
				"bbox_pipeline_sec": time.perf_counter() - start_time,
				"old_index": idx_old,
				"cur_index": idx_cur,
				"gripper_state": gripper_state,
				"shared_bbox": shared_bbox,
				"start_center": start_center,
				"cur_center": cur_center,
				"old_mask_pixels": int(old_mask.sum()) if old_mask is not None else 0,
				"cur_mask_pixels": int(cur_mask.sum()) if cur_mask is not None else 0,
			}
			print(
				"[StateAbstraction] OPEN: bbox_pipeline_sec="
				f"{time.perf_counter() - start_time:.3f}, old_idx={idx_old}, cur_idx={idx_cur}, "
				f"shared_bbox={'yes' if shared_bbox is not None else 'no'}, "
				f"mask_refine={'off' if self.manipulation_backend == 'yolo' else 'on'}"
			)

		else:  # gripper_state == "close"
			if self.manipulation_backend == "yolo":
				older_idx = max(0, idx_old - self.window_size)
				older_rgb = self._load_rgb(older_idx)
				prev_bbox = self._diff_bbox(current_rgb=old_rgb, ref_rgb=older_rgb)
				cur_bbox = self._diff_bbox(current_rgb=current_rgb, ref_rgb=old_rgb)
				start_center = self._bbox_center(prev_bbox)
				cur_center = self._bbox_center(cur_bbox)
				self._draw_result(image=image, bbox=cur_bbox, start_pt=start_center, cur_pt=cur_center)
				detected_objects = self._render_yolo_manipulation(current_rgb=current_rgb)[1].get("detected_objects", [])
				self.last_timing_stats = {
					"bbox_pipeline_sec": time.perf_counter() - start_time,
					"old_index": idx_old,
					"cur_index": idx_cur,
					"gripper_state": gripper_state,
					"detected_objects": detected_objects,
					"prev_bbox": prev_bbox,
					"cur_bbox": cur_bbox,
					"start_center": start_center,
					"cur_center": cur_center,
				}
				print(
					"[StateAbstraction] CLOSE: yolo_pipeline_sec="
					f"{time.perf_counter() - start_time:.3f}, old_idx={idx_old}, cur_idx={idx_cur}, "
					f"objects={','.join(detected_objects)}"
				)
			else:
				# Gripper CLOSE: show SoM mask and mark annotations only
				som_result = self._get_som_generator().generate(
					image=Image.fromarray(current_rgb),
					alpha=0.0,
					label_mode="Number",
					anno_mode=["Mark", "Mask"],
					semsam_level=2,
				)
				image = self._resize_to_match(som_result["image"].convert("RGB"), current_rgb.shape[1], current_rgb.shape[0])

				self.last_timing_stats = {
					"bbox_pipeline_sec": time.perf_counter() - start_time,
					"old_index": idx_old,
					"cur_index": idx_cur,
					"gripper_state": gripper_state,
					"som_used": True,
				}
				print(
					"[StateAbstraction] CLOSE: som_pipeline_sec="
					f"{time.perf_counter() - start_time:.3f}, old_idx={idx_old}, cur_idx={idx_cur}"
				)

		elapsed = time.perf_counter() - start_time
		instruction = self._compose_instruction(gripper_state)
		return image, instruction


__all__ = ["StateAbstraction"]
