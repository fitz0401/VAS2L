from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from VA2L.vlm_inference import VLMInference


def _decompose_instruction_to_subtasks(instruction: str) -> List[str]:
	text = instruction.strip()
	if not text:
		return []
	text = re.sub(r"\s+", " ", text)
	parts = re.split(r"\bthen\b|\band then\b|\band\b|,|;", text, flags=re.IGNORECASE)
	clean = []
	for p in parts:
		t = p.strip(" .")
		if t:
			clean.append(t)
	return clean if clean else [text]


def _decompose_instruction_with_qwen(vlm: VLMInference, instruction: str, dummy_image: Image.Image) -> List[str]:
	text = instruction.strip()
	if not text:
		return []
	prompt = (
		"You are an expert in robot task decomposition.\n"
		"Split the instruction into executable robotic-arm sub tasks while keeping semantics.\n"
		"Example:\n"
		"Input: Put the marker in the pot\n"
		"Output: pick up the marker | place the marker in the pot\n"
		"Rules:\n"
		"1. Keep each sub task short and imperative.\n"
		"2. Preserve object and relation semantics.\n"
		"3. Return ONE line only, separated by ' | '.\n"
		"4. If the instruction is atomic, return it as one sub task.\n"
		f"Input: {text}\n"
		"Output:"
	)
	resp = vlm.infer(dummy_image, prompt).strip()
	# VLMInference appends concise constraints; keep parser tolerant.
	resp = resp.replace("\n", " ").strip()
	parts = [p.strip(" .") for p in resp.split("|")]
	out = [p for p in parts if p]
	return out if out else _decompose_instruction_to_subtasks(text)


def build_subtask_gt_sets(
	episode_gt_sets: Dict[int, List[str]],
	decompose_fn: Optional[Callable[[str], List[str]]] = None,
) -> Dict[int, List[str]]:
	decompose = decompose_fn or _decompose_instruction_to_subtasks
	out: Dict[int, List[str]] = {}
	for ep_idx, instruction_set in episode_gt_sets.items():
		seen = set()
		subtasks: List[str] = []
		for instruction in instruction_set:
			for sub in decompose(instruction):
				key = sub.lower().strip()
				if key and key not in seen:
					seen.add(key)
					subtasks.append(sub)
		out[int(ep_idx)] = subtasks
	return out


def save_subtask_gt_sets(path: Path, subtask_gt_sets: Dict[int, List[str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {str(k): v for k, v in subtask_gt_sets.items()}
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_subtask_gt_sets(path: Path) -> Dict[int, List[str]]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	out: Dict[int, List[str]] = {}
	for k, v in payload.items():
		try:
			ep_idx = int(k)
		except (TypeError, ValueError):
			continue
		if isinstance(v, list):
			out[ep_idx] = [str(x).strip() for x in v if str(x).strip()]
	return out


def _decode_if_bytes(value: Any) -> Any:
	if isinstance(value, (bytes, bytearray)):
		return value.decode("utf-8", errors="ignore")
	return value


def _to_step_list(steps: Any) -> List[Dict[str, Any]]:
	if isinstance(steps, list):
		return steps
	if isinstance(steps, tuple):
		return list(steps)
	if hasattr(steps, "as_numpy_iterator"):
		return list(steps.as_numpy_iterator())
	return list(steps)


def _extract_episode_gt_set(steps: List[Dict[str, Any]]) -> List[str]:
	if not steps:
		return []
	first = steps[0]
	vals = [
		_decode_if_bytes(first.get("language_instruction", "")),
		_decode_if_bytes(first.get("language_instruction_2", "")),
		_decode_if_bytes(first.get("language_instruction_3", "")),
	]
	seen = set()
	out: List[str] = []
	for v in vals:
		t = str(v).strip()
		if t and t not in seen:
			seen.add(t)
			out.append(t)
	return out


def _iter_episodes(builder: Any, split: str, tfds: Any, max_episodes: int) -> Iterable[Tuple[int, Dict[str, Any]]]:
	ds = builder.as_dataset(split=split)
	for ep_idx, episode in enumerate(tfds.as_numpy(ds)):
		if ep_idx >= max_episodes:
			break
		yield ep_idx, episode


def main() -> None:
	parser = argparse.ArgumentParser(description="Build DROID-100 subtask GT set JSON from long-horizon instructions.")
	parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/droid_100/1.0.0"))
	parser.add_argument("--split", type=str, default="train")
	parser.add_argument("--max-episodes", type=int, default=100)
	parser.add_argument("--output", type=Path, default=Path("dataset/droid_100/subtask_gt_set.json"))
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--model", type=str, choices=["qwen-vl-4b", "qwen-vl-2b"], default="qwen-vl-4b")
	parser.add_argument("--precision", type=str, choices=["auto", "fp16", "bf16", "fp32"], default="auto")
	parser.add_argument("--fallback-rule-only", action="store_true", help="Skip Qwen and use rule-based splitting only.")
	args = parser.parse_args()

	import importlib
	tfds = importlib.import_module("tensorflow_datasets")
	builder = tfds.builder_from_directory(builder_dir=str(args.dataset_dir))

	episode_gt_sets: Dict[int, List[str]] = {}
	for ep_idx, episode in _iter_episodes(builder, args.split, tfds, args.max_episodes):
		steps = _to_step_list(episode.get("steps", []))
		if not steps:
			continue
		episode_gt_sets[ep_idx] = _extract_episode_gt_set(steps)

	decompose_fn: Callable[[str], List[str]]
	if args.fallback_rule_only:
		decompose_fn = _decompose_instruction_to_subtasks
	else:
		vlm = VLMInference(
			model=args.model,
			model_id=None,
			device=args.device,
			precision=args.precision,
		)
		dummy_image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
		cache: Dict[str, List[str]] = {}

		def decompose_fn(instruction: str) -> List[str]:
			key = instruction.strip()
			if not key:
				return []
			if key not in cache:
				cache[key] = _decompose_instruction_with_qwen(vlm, key, dummy_image)
			return cache[key]

	subtask_gt_sets = build_subtask_gt_sets(episode_gt_sets, decompose_fn=decompose_fn)
	save_subtask_gt_sets(args.output, subtask_gt_sets)
	print(f"saved: {args.output}")
	print(f"episodes: {len(subtask_gt_sets)}")


if __name__ == "__main__":
	main()
