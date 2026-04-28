from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import Dataset


@dataclass
class EpisodeSample:
	episode_index: int
	video_path: Path
	segment_index: int
	frame_idx: int
	keyframe_idx: int
	label: int
	transition_type: str


class SuccessFrameDataset(Dataset):
	"""
	Frame-level dataset for success classification.

	Rules implemented from user requirements:
	1. Keyframe is derived from action[:, gripper_action_index] transition 0 -> 1.
	2. Positive frames are frames within keyframe-1s to keyframe (configurable).
	3. Only frames before/equal keyframe are kept (no post-close frames).
	4. Only wrist_image_left videos are used.
	"""

	def __init__(self, config_path: str | Path):
		self.config_path = Path(config_path)
		self.cfg = self._load_config(self.config_path)

		self.repo_root = self._infer_repo_root(self.config_path)
		dataset_cfg = self.cfg["dataset"]
		label_cfg = self.cfg["labeling"]
		dl_cfg = self.cfg["dataloader"]

		self.dataset_root = (self.repo_root / dataset_cfg["root"]).resolve()
		self.data_dir = (self.dataset_root / dataset_cfg["data_dir"]).resolve()
		self.wrist_video_dir = (self.dataset_root / dataset_cfg["wrist_video_dir"]).resolve()

		self.fps = float(dataset_cfg.get("fps", 15))
		self.parquet_pattern = dataset_cfg.get("parquet_pattern", "episode_*.parquet")
		self.video_pattern = dataset_cfg.get("video_pattern", "episode_*.mp4")

		self.gripper_action_index = int(label_cfg.get("gripper_action_index", -1))
		self.label_mode = str(label_cfg.get("label_mode", "close")).strip().lower()
		self.transition_from = float(label_cfg.get("transition_from", 0))
		self.transition_to = float(label_cfg.get("transition_to", 1))
		self.transition_threshold = float(label_cfg.get("transition_threshold", 0.5))
		self.positive_seconds_before_keyframe = float(label_cfg.get("positive_seconds_before_keyframe", 1.0))
		self.include_keyframe_in_positive = bool(label_cfg.get("include_keyframe_in_positive", True))
		self.keep_only_before_keyframe = bool(label_cfg.get("keep_only_before_keyframe", True))
		self.skip_no_transition_episode = bool(label_cfg.get("skip_no_transition_episode", True))

		resize_hw = dl_cfg.get("resize_hw", [224, 224])
		self.resize_hw = (int(resize_hw[0]), int(resize_hw[1]))
		self.normalize = bool(dl_cfg.get("normalize", True))
		self.rgb = bool(dl_cfg.get("rgb", True))

		self.samples: List[EpisodeSample] = []
		self.positive_count = 0
		self.negative_count = 0
		self._build_index()
		self.total_count = len(self.samples)
		self.positive_ratio = self.positive_count / max(self.total_count, 1)
		self.negative_ratio = self.negative_count / max(self.total_count, 1)

	@staticmethod
	def _load_config(config_path: Path) -> Dict:
		with config_path.open("r", encoding="utf-8") as f:
			cfg = yaml.safe_load(f) or {}
		for k in ["dataset", "labeling", "dataloader"]:
			if k not in cfg:
				raise ValueError(f"Missing '{k}' section in config: {config_path}")
		return cfg

	@staticmethod
	def _infer_repo_root(config_path: Path) -> Path:
		# Expect config under <repo>/success_module/config.yaml
		return config_path.resolve().parent.parent

	def _episode_index_from_name(self, p: Path) -> int:
		# episode_000123.parquet -> 123
		stem = p.stem
		return int(stem.split("_")[-1])

	def _video_path_for_episode(self, episode_index: int) -> Path:
		return self.wrist_video_dir / f"episode_{episode_index:06d}.mp4"

	def _read_gripper_signal(self, parquet_path: Path) -> np.ndarray:
		table = pq.read_table(parquet_path, columns=["actions"])
		actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
		if actions.ndim != 2:
			raise ValueError(f"Unexpected actions shape in {parquet_path}: {actions.shape}")
		idx = self.gripper_action_index
		if idx < 0:
			idx = actions.shape[1] + idx
		if idx < 0 or idx >= actions.shape[1]:
			raise IndexError(f"gripper_action_index out of range: {self.gripper_action_index}")
		return actions[:, idx]

	def _find_keyframes(self, signal: np.ndarray, transition_from: float, transition_to: float) -> List[int]:
		keyframes: List[int] = []
		for i in range(1, len(signal)):
			prev_v = float(signal[i - 1])
			curr_v = float(signal[i])
			if abs(prev_v - transition_from) <= self.transition_threshold and abs(curr_v - transition_to) <= self.transition_threshold:
				keyframes.append(i)
		return keyframes

	def _select_keyframes(self, signal: np.ndarray) -> List[Tuple[int, str]]:
		if self.label_mode == "close":
			return [(idx, "close") for idx in self._find_keyframes(signal, self.transition_from, self.transition_to)[:1]]
		if self.label_mode == "open":
			return [(idx, "open") for idx in self._find_keyframes(signal, self.transition_to, self.transition_from)[:1]]
		if self.label_mode == "both":
			close_keyframes = [(idx, "close") for idx in self._find_keyframes(signal, self.transition_from, self.transition_to)]
			open_keyframes = [(idx, "open") for idx in self._find_keyframes(signal, self.transition_to, self.transition_from)]
			merged = sorted(close_keyframes + open_keyframes, key=lambda item: item[0])
			unique: List[Tuple[int, str]] = []
			seen = set()
			for keyframe_idx, transition_type in merged:
				if keyframe_idx in seen:
					continue
				seen.add(keyframe_idx)
				unique.append((keyframe_idx, transition_type))
			return unique
		raise ValueError(f"Unsupported label_mode: {self.label_mode}. Expected close/open/both.")

	def _build_index(self) -> None:
		parquet_files = sorted(self.data_dir.glob(self.parquet_pattern))
		if not parquet_files:
			raise FileNotFoundError(f"No parquet files found under {self.data_dir} with pattern {self.parquet_pattern}")

		pos_window = max(1, int(round(self.positive_seconds_before_keyframe * self.fps)))

		for parquet_path in parquet_files:
			episode_index = self._episode_index_from_name(parquet_path)
			video_path = self._video_path_for_episode(episode_index)

			if not video_path.exists():
				continue

			signal = self._read_gripper_signal(parquet_path)
			keyframes = self._select_keyframes(signal)
			if not keyframes:
				if self.skip_no_transition_episode:
					continue
				keyframes = [(len(signal) - 1, "close")]

			for segment_index, (keyframe, transition_type) in enumerate(keyframes):
				segment_start = 0 if segment_index == 0 else keyframes[segment_index - 1][0] + 1
				segment_end = keyframe if self.keep_only_before_keyframe or self.label_mode == "both" else len(signal) - 1
				segment_end = min(segment_end, len(signal) - 1)
				if segment_start > segment_end:
					continue

				start_pos = max(segment_start, keyframe - pos_window)
				end_pos = keyframe if self.include_keyframe_in_positive else keyframe - 1
				end_pos = max(end_pos, -1)

				for fi in range(segment_start, segment_end + 1):
					label = int(start_pos <= fi <= end_pos)
					self.samples.append(
						EpisodeSample(
							episode_index=episode_index,
							video_path=video_path,
							segment_index=segment_index,
							frame_idx=fi,
							keyframe_idx=keyframe,
							label=label,
							transition_type=transition_type,
						)
					)
					if label == 1:
						self.positive_count += 1
					else:
						self.negative_count += 1

		if not self.samples:
			raise RuntimeError("No samples built. Check config paths and transition settings.")

	def __len__(self) -> int:
		return len(self.samples)

	def _load_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
		cap = cv2.VideoCapture(str(video_path))
		if not cap.isOpened():
			raise RuntimeError(f"Failed to open video: {video_path}")
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
		ok, frame = cap.read()
		cap.release()
		if not ok:
			raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

		frame = cv2.resize(frame, (self.resize_hw[1], self.resize_hw[0]), interpolation=cv2.INTER_AREA)
		if self.rgb:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = frame.astype(np.float32)
		if self.normalize:
			frame /= 255.0
		frame = np.transpose(frame, (2, 0, 1))
		return frame

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
		s = self.samples[index]
		img = self._load_frame(s.video_path, s.frame_idx)
		img_t = torch.from_numpy(img).float()
		label_t = torch.tensor(s.label, dtype=torch.long)
		meta = {
			"episode_index": s.episode_index,
			"segment_index": s.segment_index,
			"frame_idx": s.frame_idx,
			"keyframe_idx": s.keyframe_idx,
			"transition_type": s.transition_type,
			"video_path": str(s.video_path),
		}
		return img_t, label_t, meta


def build_dataset(config_path: str | Path = "success_module/config.yaml") -> SuccessFrameDataset:
	return SuccessFrameDataset(config_path)
