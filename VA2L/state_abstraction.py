from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_IGNORED_DETECTED_OBJECTS = {"table", "person"}


def _compute_diff_bbox(a_rgb: np.ndarray, b_rgb: np.ndarray, diff_threshold: int, min_diff_area: int) -> Optional[Tuple[int, int, int, int]]:
	import cv2

	diff = cv2.absdiff(a_rgb, b_rgb)
	gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
	_, binary = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)
	binary = cv2.medianBlur(binary, 5)
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if not contours:
		return None

	best = max(contours, key=cv2.contourArea)
	if float(cv2.contourArea(best)) < float(min_diff_area):
		return None

	x, y, w, h = cv2.boundingRect(best)
	return x, y, x + w, y + h


def _bbox_center(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
	if bbox is None:
		return None
	x1, y1, x2, y2 = bbox
	return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))


def _detect_yolo_objects(yolo_model: Any, image_rgb: np.ndarray) -> List[str]:
	results = yolo_model.predict(source=image_rgb, verbose=False)
	if not results:
		return []
	result = results[0]
	boxes = getattr(result, "boxes", None)
	if boxes is None or len(boxes) == 0:
		return []
	names = result.names if hasattr(result, "names") else {}
	detected_names: List[str] = []
	for box in boxes:
		cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
		if isinstance(names, dict):
			name = names.get(cls_id, str(cls_id))
		elif isinstance(names, list) and 0 <= cls_id < len(names):
			name = names[cls_id]
		else:
			name = str(cls_id)
		if name.lower() not in _IGNORED_DETECTED_OBJECTS:
			detected_names.append(name)
	return sorted(set(detected_names))


def _draw_motion_overlay(
	current_rgb: np.ndarray,
	one_sec_rgb: np.ndarray,
	two_sec_rgb: np.ndarray,
	diff_threshold: int,
	min_diff_area: int,
) -> Image.Image:
	canvas = Image.fromarray(current_rgb.copy())
	draw = ImageDraw.Draw(canvas)

	cur_bbox = _compute_diff_bbox(current_rgb, one_sec_rgb, diff_threshold, min_diff_area)
	prev_bbox = _compute_diff_bbox(one_sec_rgb, two_sec_rgb, diff_threshold, min_diff_area)
	start_pt = _bbox_center(prev_bbox)
	end_pt = _bbox_center(cur_bbox)

	if cur_bbox is not None:
		draw.rectangle(cur_bbox, outline=(255, 255, 255), width=3)
	if start_pt is not None and end_pt is not None:
		draw.line([start_pt, end_pt], fill=(255, 80, 70), width=4)
		dx = end_pt[0] - start_pt[0]
		dy = end_pt[1] - start_pt[1]
		norm = float(np.hypot(dx, dy))
		if norm > 1e-6:
			ux, uy = dx / norm, dy / norm
			head_len = 12.0
			head_w = 7.0
			tip = np.array([end_pt[0], end_pt[1]], dtype=np.float64)
			base = tip - head_len * np.array([ux, uy], dtype=np.float64)
			perp = np.array([-uy, ux], dtype=np.float64)
			left = base + head_w * perp
			right = base - head_w * perp
			triangle = [tuple(tip.astype(int)), tuple(left.astype(int)), tuple(right.astype(int))]
			draw.polygon(triangle, fill=(255, 80, 70))

	return canvas


def _compose_instruction(gripper_state: str, detected_objects: Optional[List[str]] = None) -> str:
	obj_line = ""
	if detected_objects:
		obj_text = ", ".join(detected_objects)
		obj_line = f"Detected objects: {obj_text}"

	if gripper_state == "open":
		lines = [
			"Infer the robotic arm's current action. <action><target_object>",
			"Rules: Choose the best action from: pick, open, close. Keep it short. Do not explain.",
			"Hint: The white bounding box represents the area where motion occurred in the past 2 seconds. Gripper state: OPEN.",
			"Examples: `pick the object`, `open the drawer`",
		]
		if obj_line:
			lines.insert(3, obj_line)
	else:
		lines = [
			"Infer the robotic arm's current action. <action><grasped_object><relation><target_object>",
			"Rules: Choose the best action from: pick, place, insert, open, close. Choose relation from: on, into, next to, from. Keep it short. Do not explain.",
			"Hint: The bounding boxes outline objects. Consider object relationships. Gripper state: CLOSED.",
			"Examples: `insert the pen into the mug`, `place the box next to the bottle`",
		]
		if obj_line:
			lines.insert(3, obj_line)
	return "\n".join(lines)


def build_state_abstraction_frame(
	current_rgb: np.ndarray,
	one_sec_rgb: np.ndarray,
	two_sec_rgb: np.ndarray,
	gripper_state: str,
	diff_threshold: int = 24,
	min_diff_area: int = 250,
	yolo_model: Any = None,
	wrist_rgb: Optional[np.ndarray] = None,
	prev_wrist_rgb: Optional[np.ndarray] = None,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
	rendered = _draw_motion_overlay(
		current_rgb=current_rgb,
		one_sec_rgb=one_sec_rgb,
		two_sec_rgb=two_sec_rgb,
		diff_threshold=diff_threshold,
		min_diff_area=min_diff_area,
	)

	# Object detection logic
	front_objects: List[str] = []
	wrist_objects: List[str] = []
	if yolo_model is not None:
		if gripper_state == "closed":
			front_objects = _detect_yolo_objects(yolo_model, current_rgb)
		elif gripper_state == "open" and wrist_rgb is not None and prev_wrist_rgb is not None:
			if _compute_diff_bbox(wrist_rgb, prev_wrist_rgb, diff_threshold, min_diff_area) is not None:
				wrist_objects = _detect_yolo_objects(yolo_model, wrist_rgb)
				front_objects = wrist_objects

	if gripper_state == "closed" and wrist_rgb is not None and prev_wrist_rgb is not None:
		if _compute_diff_bbox(wrist_rgb, prev_wrist_rgb, diff_threshold, min_diff_area) is not None:
			wrist_objects = _detect_yolo_objects(yolo_model, wrist_rgb)

	detected_objects = sorted(set(front_objects + wrist_objects))
	instruction = _compose_instruction(gripper_state, detected_objects=detected_objects)
	stats = {
		"detected_objects": detected_objects,
		"detections": len(detected_objects),
	}
	return rendered, instruction, stats


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

	def _compose_instruction(self, gripper_state: str) -> str:
		return _compose_instruction(gripper_state)

	def prepare_vlm_inputs(self, t: int) -> Tuple[Image.Image, str]:
		if t < 0 or t >= len(self.image_paths):
			raise IndexError(f"t={t} out of range [0, {len(self.image_paths) - 1}]")

		idx_old, idx_cur = self._window_indices(t)
		idx_one_sec = max(0, t - self.window_size)
		idx_two_sec = max(0, t - 2 * self.window_size)
		current_rgb = self._load_rgb(t)
		one_sec_rgb = self._load_rgb(idx_one_sec)
		two_sec_rgb = self._load_rgb(idx_two_sec)
		gripper_state = self._load_gripper_state(t)
		yolo_model = self.yolo_model if self.manipulation_backend == "yolo" else None
		image, instruction, stats = build_state_abstraction_frame(
			current_rgb=current_rgb,
			one_sec_rgb=one_sec_rgb,
			two_sec_rgb=two_sec_rgb,
			gripper_state=gripper_state,
			diff_threshold=self.diff_threshold,
			min_diff_area=self.min_diff_area,
			yolo_model=yolo_model,
		)
		self.last_timing_stats = {
			"old_index": idx_old,
			"cur_index": idx_cur,
			"gripper_state": gripper_state,
			**stats,
		}
		return image, instruction


def prepare_state_abstraction_from_demo(
	demo_dir: str,
	t: int,
	window_size: int = 15,
	object_prompt: str = "gripper",
	diff_threshold: int = 24,
	min_diff_area: int = 250,
	manipulation_backend: str = "sam",
	yolo_model_path: str = "yolo26s.pt",
) -> Tuple[Image.Image, str, Dict[str, Any]]:
	abstracter = StateAbstraction(
		demo_dir=demo_dir,
		window_size=window_size,
		object_prompt=object_prompt,
		diff_threshold=diff_threshold,
		min_diff_area=min_diff_area,
		manipulation_backend=manipulation_backend,
		yolo_model_path=yolo_model_path,
	)
	image, instruction = abstracter.prepare_vlm_inputs(t)
	return image, instruction, getattr(abstracter, "last_timing_stats", {})


__all__ = ["StateAbstraction", "build_state_abstraction_frame", "prepare_state_abstraction_from_demo"]
