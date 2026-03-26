from pathlib import Path
import json
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
	from vision_module.mask_generator import MaskGenerator
	from vision_module.som_generator import SoMGenerator
except ImportError:
	import sys
	workspace_root = Path(__file__).resolve().parent.parent
	if str(workspace_root) not in sys.path:
		sys.path.insert(0, str(workspace_root))
	from vision_module.mask_generator import MaskGenerator
	from vision_module.som_generator import SoMGenerator


class StateAbstraction:
	"""Prepare VLM inputs using two-frame difference-guided object segmentation."""

	def __init__(
		self,
		demo_dir: str,
		window_size: int = 15,
		object_prompt: str = "gripper",
		diff_threshold: int = 24,
		min_diff_area: int = 250,
	):
		self.demo_dir = Path(demo_dir)
		self.window_size = int(window_size)
		self.object_prompt = object_prompt
		self.diff_threshold = int(diff_threshold)
		self.min_diff_area = int(min_diff_area)

		self.image_paths = sorted((self.demo_dir / "color").glob("*.png"))
		if not self.image_paths:
			raise FileNotFoundError(f"No RGB images found in {self.demo_dir / 'color'}")

		# Load trajectory data
		trajectory_path = self.demo_dir / "trajectory.json"
		if not trajectory_path.exists():
			raise FileNotFoundError(f"trajectory.json not found in {self.demo_dir}")
		with open(trajectory_path, "r") as f:
			self.trajectory = json.load(f)

		self.mask_generator = MaskGenerator()
		self.som_generator = SoMGenerator()

	def _window_indices(self, t: int) -> Tuple[int, int]:
		start = max(0, t - self.window_size)
		return start, t

	def _load_rgb(self, idx: int) -> np.ndarray:
		return np.asarray(Image.open(self.image_paths[idx]).convert("RGB"))

	def _load_gripper_state(self, idx: int) -> str:
		"""Load gripper state from trajectory.json for frame index idx."""
		if idx < 0 or idx >= len(self.trajectory):
			raise IndexError(f"idx={idx} out of range [0, {len(self.trajectory) - 1}]")
		state = self.trajectory[idx].get("gripper_state", "unknown")
		return state.lower()  # Normalize to lowercase

	def _diff_bbox(self, current_rgb: np.ndarray, ref_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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
			img_pil = Image.fromarray(rgb)
			h, w = rgb.shape[:2]
			cx, cy, bw, bh = self._bbox_to_norm_cxcywh(bbox, w=w, h=h)
			boxes = torch.tensor([[cx, cy, bw, bh]], dtype=torch.float32)
			logits = torch.tensor([1.0], dtype=torch.float32)
			phrases = [self.object_prompt]
			masks = self.mask_generator.get_segmentation_masks(
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

	def _compose_instruction(self, gripper_state: str) -> str:
		"""Generate language instruction based on gripper state."""
		if gripper_state == "open":
			lines = [
				"Infer the task ONLY based on:",
				"1) frame difference trajectory (motion region in last 2s)",
				"2) direction of gripper motion (indicated by arrow)",
				"3) gripper state: OPEN",
				"Hint: Determine what object or region is being approached or moved toward."
			]
		elif gripper_state == "closed":
			lines = [
				"Infer the task ONLY based on:",
				"1) objects with mark annotations in the scene",
				"2) grasped object and its position relative to other objects",
				"3) gripper state: CLOSED",
				"Hint: Identify the affordance between the grasped object and other objects."
			]
		else:
			lines = [
				"Infer the task ONLY based on:",
				"1) motion region and objects with mark annotations",
				"2) relative position between gripper and objects",
				"3) gripper state: UNKNOWN",
				"Hint: Determine the task from available visual cues."
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
			# Gripper OPEN: show frame difference bbox + trajectory line and arrow only
			shared_bbox = self._diff_bbox(current_rgb=current_rgb, ref_rgb=old_rgb)
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
				f"shared_bbox={'yes' if shared_bbox is not None else 'no'}"
			)

		else:  # gripper_state == "close"
			# Gripper CLOSE: show SoM mask and mark annotations only
			som_result = self.som_generator.generate(
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
