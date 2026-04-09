"""Load and inspect DROID RLDS (TFDS) episodes and export demo artifacts.

Example:
	python -m VA2L.utils.load_droid \
		--dataset-dir /home/ze/VAS2L/dataset/droid_100/1.0.0 \
		--episode-index 0 \
		--save-video /home/ze/VAS2L/dataset/droid_100/demo_ep0.mp4 \
		--save-metadata /home/ze/VAS2L/dataset/droid_100/demo_ep0_meta.json \
		--save-gripper /home/ze/VAS2L/dataset/droid_100/demo_ep0_gripper.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _decode_if_bytes(value: Any) -> Any:
	if isinstance(value, (bytes, bytearray)):
		return value.decode("utf-8", errors="ignore")
	return value


def _to_jsonable(value: Any) -> Any:
	if isinstance(value, dict):
		return {k: _to_jsonable(v) for k, v in value.items()}
	if isinstance(value, list):
		return [_to_jsonable(v) for v in value]
	if isinstance(value, tuple):
		return [_to_jsonable(v) for v in value]
	if isinstance(value, (bytes, bytearray)):
		return _decode_if_bytes(value)
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	return value


def _to_step_list(steps: Any) -> List[Dict[str, Any]]:
	# RLDS "steps" can arrive as a generator-like object when using tfds.as_numpy.
	if isinstance(steps, list):
		return steps
	if isinstance(steps, tuple):
		return list(steps)
	if hasattr(steps, "as_numpy_iterator"):
		return list(steps.as_numpy_iterator())
	if isinstance(steps, np.ndarray):
		return list(steps.tolist())
	return list(steps)


def _find_image_candidates(observation: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
	candidates: List[Tuple[str, np.ndarray]] = []
	for key, value in observation.items():
		if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in (1, 3, 4):
			if value.dtype == np.uint8:
				candidates.append((key, value))
	return candidates


def _normalize_image_for_save(image: np.ndarray) -> np.ndarray:
	if image.ndim != 3:
		raise ValueError(f"Expected image ndim=3, got ndim={image.ndim}.")
	if image.dtype != np.uint8:
		raise ValueError(f"Expected image dtype=uint8, got dtype={image.dtype}.")
	if image.shape[-1] == 1:
		return np.repeat(image, 3, axis=-1)
	if image.shape[-1] == 4:
		return image[..., :3]
	if image.shape[-1] != 3:
		raise ValueError(f"Expected image channel in (1, 3, 4), got {image.shape[-1]}.")
	return image


def _load_one_episode(builder: Any, split: str, episode_index: int) -> Dict[str, Any]:
	tfds = importlib.import_module("tensorflow_datasets")

	ds = builder.as_dataset(split=split)
	sampled = ds.skip(episode_index).take(1)
	try:
		episode = next(iter(tfds.as_numpy(sampled)))
	except StopIteration as exc:
		raise IndexError(
			f"No episode found at index {episode_index} in split '{split}'."
		) from exc
	return episode


def _save_image(image: np.ndarray, output_path: Path) -> None:
	from PIL import Image

	output_path.parent.mkdir(parents=True, exist_ok=True)
	if image.shape[-1] == 1:
		image = image[..., 0]
	Image.fromarray(image).save(output_path)


def _save_video(frames: List[np.ndarray], output_path: Path, fps: int) -> None:
	if not frames:
		raise RuntimeError("No frames to write for video export.")
	if fps <= 0:
		raise ValueError("fps must be > 0.")

	try:
		imageio = importlib.import_module("imageio.v2")
	except ImportError as exc:
		raise ImportError(
			"imageio is required for video export. Install with: pip install imageio imageio-ffmpeg"
		) from exc

	output_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		with imageio.get_writer(str(output_path), format="FFMPEG", fps=fps, codec="libx264") as writer:
			for frame in frames:
				writer.append_data(frame)
	except Exception as exc:
		raise RuntimeError(
			f"Failed to save video to {output_path}. "
			"Please ensure ffmpeg is available (pip install imageio-ffmpeg)."
		) from exc


def _resolve_image_key(steps: List[Dict[str, Any]], image_key: str | None) -> str:
	if not steps:
		raise RuntimeError("Episode has no steps.")

	first_obs = steps[0].get("observation", {})
	image_candidates = _find_image_candidates(first_obs)
	if not image_candidates:
		raise RuntimeError("No image tensors found in first step['observation'].")

	if image_key is None:
		return image_candidates[0][0]

	if image_key not in first_obs:
		available = ", ".join(sorted(k for k, _ in image_candidates))
		raise KeyError(f"image-key '{image_key}' not found. Available: {available}")

	image = first_obs[image_key]
	if not (isinstance(image, np.ndarray) and image.ndim == 3):
		raise ValueError(f"observation['{image_key}'] is not a valid image tensor.")
	return image_key


def _extract_instructions(step: Dict[str, Any]) -> Dict[str, str]:
	return {
		"language_instruction": str(_decode_if_bytes(step.get("language_instruction", b""))),
		"language_instruction_2": str(_decode_if_bytes(step.get("language_instruction_2", b""))),
		"language_instruction_3": str(_decode_if_bytes(step.get("language_instruction_3", b""))),
	}


def _extract_gripper_position(step: Dict[str, Any]) -> float:
	obs = step.get("observation", {})
	value = obs.get("gripper_position", None)
	if value is None:
		return float("nan")

	arr = np.asarray(value)
	if arr.size == 0:
		return float("nan")
	return float(arr.reshape(-1)[0])


def _save_gripper_csv(gripper_positions: List[float], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["step_index", "gripper_position"])
		for idx, value in enumerate(gripper_positions):
			writer.writerow([idx, value])


def _collect_vector_observations(steps: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
	if not steps:
		return {}

	first_obs = steps[0].get("observation", {})
	vector_keys: List[str] = []
	for key, value in first_obs.items():
		if isinstance(value, np.ndarray):
			if not (value.ndim == 3 and value.shape[-1] in (1, 3, 4) and value.dtype == np.uint8):
				vector_keys.append(key)

	collected: Dict[str, np.ndarray] = {}
	for key in vector_keys:
		items = [np.asarray(step.get("observation", {}).get(key)) for step in steps]
		collected[key] = np.stack(items, axis=0)
	return collected


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Load DROID RLDS episodes and export frame/video/metadata artifacts."
	)
	parser.add_argument(
		"--dataset-dir",
		type=Path,
		default=Path("dataset/droid_100/1.0.0"),
		help="Directory containing dataset_info.json/features.json and TFRecord shards.",
	)
	parser.add_argument("--split", type=str, default="train", help="Dataset split name.")
	parser.add_argument("--episode-index", type=int, default=0, help="Episode index in split.")
	parser.add_argument("--step-index", type=int, default=0, help="Step index inside episode.")
	parser.add_argument(
		"--image-key",
		type=str,
		default=None,
		help="Observation image key, e.g. exterior_image_1_left. Defaults to first detected image key.",
	)
	parser.add_argument(
		"--save-image",
		type=Path,
		default=Path("dataset/droid_100/sample_frame.jpg"),
		help="Where to save the extracted image frame.",
	)
	parser.add_argument(
		"--save-video",
		type=Path,
		default=Path("dataset/droid_100/demo_episode.mp4"),
		help="Where to save the full episode video from observation image frames.",
	)
	parser.add_argument("--fps", type=int, default=30, help="FPS for exported demo video.")
	parser.add_argument(
		"--save-metadata",
		type=Path,
		default=Path("dataset/droid_100/demo_episode_metadata.json"),
		help="Where to save episode metadata + all language instructions.",
	)
	parser.add_argument(
		"--save-gripper",
		type=Path,
		default=Path("dataset/droid_100/demo_episode_gripper.csv"),
		help="Where to save per-step gripper_position CSV.",
	)
	parser.add_argument(
		"--save-observations",
		type=Path,
		default=Path("dataset/droid_100/demo_episode_observations.npz"),
		help="Where to save non-image observation tensors as NPZ.",
	)
	parser.add_argument(
		"--no-video",
		action="store_true",
		help="Disable full-episode video export.",
	)
	args = parser.parse_args()

	try:
		tfds = importlib.import_module("tensorflow_datasets")
	except ImportError as exc:
		raise ImportError(
			"tensorflow-datasets is required. Install with: pip install tensorflow-datasets"
		) from exc

	try:
		builder = tfds.builder_from_directory(builder_dir=str(args.dataset_dir))
	except Exception as exc:
		raise RuntimeError(
			f"Failed to build TFDS dataset from {args.dataset_dir}. "
			"Please ensure this directory has dataset_info.json and TFRecord shards."
		) from exc

	episode = _load_one_episode(builder=builder, split=args.split, episode_index=args.episode_index)

	metadata = episode.get("episode_metadata", {})
	print("=== Episode metadata ===")
	for key, value in metadata.items():
		print(f"{key}: {_decode_if_bytes(value)}")

	steps = _to_step_list(episode.get("steps", []))
	if not steps:
		raise RuntimeError("Episode has no steps.")
	if args.step_index < 0 or args.step_index >= len(steps):
		raise IndexError(f"step-index {args.step_index} is out of range [0, {len(steps) - 1}].")

	selected_key = _resolve_image_key(steps=steps, image_key=args.image_key)

	step = steps[args.step_index]
	instruction_dict = _extract_instructions(step)
	print("=== Episode summary ===")
	print(f"split: {args.split}")
	print(f"episode_index: {args.episode_index}")
	print(f"num_steps: {len(steps)}")
	print(f"step_index: {args.step_index}")
	print(f"language_instruction: {instruction_dict['language_instruction']}")
	print(f"selected_image_key: {selected_key}")

	observation = step.get("observation", {})
	image = np.asarray(observation[selected_key])

	_save_image(image=image, output_path=args.save_image)
	print("=== Image export ===")
	print(f"image_shape: {tuple(image.shape)}")
	print(f"saved_to: {args.save_image}")

	all_instructions = [_extract_instructions(s) for s in steps]
	gripper_positions = [_extract_gripper_position(s) for s in steps]
	vector_obs = _collect_vector_observations(steps)

	if not args.no_video:
		frames: List[np.ndarray] = []
		for idx, s in enumerate(steps):
			obs = s.get("observation", {})
			if selected_key not in obs:
				raise KeyError(f"step {idx} has no observation['{selected_key}']")
			frame = _normalize_image_for_save(np.asarray(obs[selected_key]))
			frames.append(frame)
		_save_video(frames=frames, output_path=args.save_video, fps=args.fps)
		print("=== Video export ===")
		print(f"num_frames: {len(frames)}")
		print(f"fps: {args.fps}")
		print(f"saved_to: {args.save_video}")

	metadata_json = {
		"split": args.split,
		"episode_index": args.episode_index,
		"num_steps": len(steps),
		"selected_image_key": selected_key,
		"episode_metadata": _to_jsonable(metadata),
		"observation_keys": sorted(list(steps[0].get("observation", {}).keys())),
		"language_instructions_all": all_instructions,
	}
	args.save_metadata.parent.mkdir(parents=True, exist_ok=True)
	with args.save_metadata.open("w", encoding="utf-8") as f:
		json.dump(metadata_json, f, ensure_ascii=False, indent=2)
	print("=== Metadata export ===")
	print(f"saved_to: {args.save_metadata}")

	_save_gripper_csv(gripper_positions=gripper_positions, output_path=args.save_gripper)
	print("=== Gripper export ===")
	print(f"saved_to: {args.save_gripper}")

	args.save_observations.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(args.save_observations, **vector_obs)
	print("=== Observation export ===")
	print(f"keys: {', '.join(sorted(vector_obs.keys()))}")
	print(f"saved_to: {args.save_observations}")


if __name__ == "__main__":
	main()
