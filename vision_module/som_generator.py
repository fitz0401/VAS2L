from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


class SoMGenerator:
	"""SoM generator based on Semantic-SAM only (default level=2)."""

	def __init__(
		self,
		som_root: Optional[str] = None,
		device: Optional[str] = None,
	):
		self.som_root = (
			Path(som_root).expanduser().resolve()
			if som_root
			else Path(__file__).resolve().parent.parent
		)
		if not self.som_root.exists():
			raise FileNotFoundError(f"SoM root not found: {self.som_root}")

		if str(self.som_root) not in sys.path:
			sys.path.insert(0, str(self.som_root))
		module_root = Path(__file__).resolve().parent
		if str(module_root) not in sys.path:
			sys.path.insert(0, str(module_root))

		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.semsam_ckpt = self.som_root / "ckpts" / "swinl_only_sam_many2many.pth"
		self.semsam_cfg = self.som_root / "configs" / "semantic_sam_only_sa-1b_swinL.yaml"

		for p in [self.semsam_ckpt, self.semsam_cfg]:
			if not p.exists():
				raise FileNotFoundError(f"Required SoM file not found: {p}")

		self._semsam_model = None

	@staticmethod
	def _to_label_mode(label_mode: str) -> str:
		mode = str(label_mode).strip().lower()
		return "a" if mode.startswith("a") else "1"

	def _ensure_semsam_model(self):
		if self._semsam_model is not None:
			return
		from semantic_sam.BaseModel import BaseModel
		from semantic_sam import build_model
		from semantic_sam.utils.arguments import load_opt_from_config_file

		opt = load_opt_from_config_file(str(self.semsam_cfg))
		model = BaseModel(opt, build_model(opt)).from_pretrained(str(self.semsam_ckpt)).eval()
		if self.device.startswith("cuda"):
			model = model.cuda()
		self._semsam_model = model

	def generate(
		self,
		image: Image.Image,
		alpha: float = 0.1,
		label_mode: str = "Number",
		anno_mode: Optional[Sequence[str]] = None,
		semsam_level: int = 2,
	) -> Dict[str, object]:
		"""Generate SoM output with Semantic-SAM."""
		image_rgb = image.convert("RGB")
		anno = list(anno_mode) if anno_mode is not None else ["Mask", "Mark"]
		label_token = self._to_label_mode(label_mode)
		level = int(semsam_level)
		if level < 1 or level > 6:
			raise ValueError("semsam_level must be in [1, 6]")

		text_size = 640
		hole_scale, island_scale = 100, 100
		text, text_part, text_thresh = "", "", "0.0"
		semantic = False

		self._ensure_semsam_model()
		from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

		level_list = [level]
		if self.device.startswith("cuda"):
			with torch.autocast(device_type="cuda", dtype=torch.float16):
				out_np, anns = inference_semsam_m2m_auto(
					self._semsam_model,
					image_rgb,
					level_list,
					text,
					text_part,
					text_thresh,
					text_size,
					hole_scale,
					island_scale,
					semantic,
					label_mode=label_token,
					alpha=float(alpha),
					anno_mode=anno,
				)
		else:
			out_np, anns = inference_semsam_m2m_auto(
				self._semsam_model,
				image_rgb,
				level_list,
				text,
				text_part,
				text_thresh,
				text_size,
				hole_scale,
				island_scale,
				semantic,
				label_mode=label_token,
				alpha=float(alpha),
				anno_mode=anno,
			)

		masks: List[np.ndarray] = []
		metadata: List[Dict[str, object]] = []
		for i, ann in enumerate(anns, start=1):
			seg = ann.get("segmentation", None)
			if seg is None:
				continue
			seg_u8 = np.asarray(seg, dtype=np.uint8)
			masks.append(seg_u8)
			metadata.append(
				{
					"id": i,
					"area": int(ann.get("area", int(seg_u8.sum()))),
					"bbox_xywh": [float(v) for v in ann.get("bbox", [0, 0, 0, 0])],
				}
			)

		if masks:
			masks_stacked = np.stack(masks, axis=0)
		else:
			h, w = np.asarray(image_rgb).shape[:2]
			masks_stacked = np.zeros((0, h, w), dtype=np.uint8)

		return {
			"image": Image.fromarray(out_np),
			"masks": masks_stacked,
			"metadata": metadata,
			"used_params": {
				"model_name": "semantic-sam",
				"object_level": True,
				"semantic_sam_level": [level],
			},
		}


__all__ = ["SoMGenerator"]
