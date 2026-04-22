from __future__ import annotations

import argparse
import csv
import importlib
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from VA2L.state_abstraction import build_state_abstraction_frame
from VA2L.utils.evaluate_droid_results import evaluate_experiment_results, save_experiment_results
from VA2L.utils.preprocess_droid_subtasks import build_subtask_gt_sets, load_subtask_gt_sets, save_subtask_gt_sets


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
	if isinstance(steps, np.ndarray):
		return list(steps.tolist())
	return list(steps)


def _find_image_key(first_step: Dict[str, Any], explicit_key: Optional[str]) -> str:
	obs = first_step.get("observation", {})
	if explicit_key is not None:
		if explicit_key not in obs:
			raise KeyError(f"image-key '{explicit_key}' not in observation.")
		return explicit_key

	for key, value in obs.items():
		if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in (1, 3, 4):
			if value.dtype == np.uint8:
				return key
	raise RuntimeError("No image tensor found in observation.")


def _find_wrist_key(first_step: Dict[str, Any], explicit_key: Optional[str]) -> Optional[str]:
	obs = first_step.get("observation", {})
	if explicit_key is not None:
		return explicit_key if explicit_key in obs else None
	for key, value in obs.items():
		if "wrist" in key and isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in (1, 3, 4):
			if value.dtype == np.uint8:
				return key
	return None


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
	if image.dtype != np.uint8:
		image = image.astype(np.uint8)
	if image.ndim != 3:
		raise ValueError("Expected HxWxC image.")
	if image.shape[-1] == 1:
		return np.repeat(image, 3, axis=-1)
	if image.shape[-1] == 4:
		return image[..., :3]
	return image


class GripperStateTracker:
	def __init__(self, stable_frames: int, stable_eps: float):
		self.stable_frames = max(1, stable_frames)
		self.stable_eps = stable_eps
		self._last_nonzero: Optional[float] = None
		self._stable_count = 0

	def update(self, gripper_position: float) -> str:
		if abs(gripper_position) <= self.stable_eps:
			self._last_nonzero = None
			self._stable_count = 0
			return "open"

		if self._last_nonzero is None:
			self._last_nonzero = gripper_position
			self._stable_count = 1
			return "open"

		if abs(gripper_position - self._last_nonzero) <= self.stable_eps:
			self._stable_count += 1
		else:
			self._last_nonzero = gripper_position
			self._stable_count = 1

		return "closed" if self._stable_count >= self.stable_frames else "open"


def _extract_gripper(step: Dict[str, Any]) -> float:
	obs = step.get("observation", {})
	value = obs.get("gripper_position", 0.0)
	arr = np.asarray(value)
	if arr.size == 0:
		return 0.0
	return float(arr.reshape(-1)[0])


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


def _load_vlm(args: argparse.Namespace):
	from VA2L.vlm_inference import VLMInference
	return VLMInference(
		model=args.model,
		model_id=None,
		device=args.device,
		precision="auto",
	)


def _load_yolo(args: argparse.Namespace):
	from ultralytics import YOLO
	return YOLO(args.yolo_model_path)


def _make_vlm_image_saver(args: argparse.Namespace):
	if args.debug is None:
		return None, None, None

	args.vlm_input_dir.mkdir(parents=True, exist_ok=True)
	q: queue.Queue = queue.Queue(maxsize=max(64, args.vlm_save_queue_size))
	stats = {"dropped": 0}

	def _worker() -> None:
		while True:
			item = q.get()
			if item is None:
				return
			episode_index, frame_index, image = item
			path = args.vlm_input_dir / f"ep{episode_index:04d}_f{frame_index:06d}.png"
			image.save(path)

	t = threading.Thread(target=_worker, daemon=True)
	t.start()
	return q, t, stats


def _enqueue_vlm_image(save_q: Optional[queue.Queue], stats: Optional[Dict[str, int]], episode_index: int, frame_index: int, image: Image.Image) -> None:
	if save_q is None or stats is None:
		return
	try:
		save_q.put_nowait((episode_index, frame_index, image.copy()))
	except queue.Full:
		stats["dropped"] = int(stats.get("dropped", 0)) + 1


def _judge_action_with_qwen(vlm: Any, pred_action: str, gt_set: List[str]) -> bool:
	if not pred_action.strip() or not gt_set:
		return False

	prompt = (
		"You are an evaluator for robot action prediction.\n"
		f"Ground-truth intent set: {gt_set}\n"
		f"Predicted action: {pred_action}\n"
		"Does predicted action semantically match ANY ground-truth intent in the set? "
		"Reply with exactly one token: CORRECT or WRONG."
	)
	dummy = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
	resp = vlm.infer(dummy, prompt).strip().lower()
	return resp.startswith("correct") or resp == "yes"


def _judge_detected_objects_with_qwen(vlm: Any, detected_objects: List[str], gt_set: List[str]) -> int:
	"""Use Qwen to judge whether detected objects semantically match objects in the task set."""
	if not detected_objects or not gt_set:
		return 0

	prompt = (
		"You are evaluating object relevance for a robot task.\n"
		f"Task descriptions: {gt_set}\n"
		f"Detected objects: {detected_objects}\n"
		"Question: Is at least one detected object semantically the same as an object mentioned or required in the task descriptions?\n"
		"Reply with exactly one token: 1 or 0."
	)
	dummy = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
	resp = vlm.infer(dummy, prompt).strip().lower()
	return 1 if resp.startswith("1") or resp.startswith("yes") or resp.startswith("true") else 0


def _save_debug_frame(
	debug_dir: Path,
	episode_index: int,
	frame_index: int,
	image: Image.Image,
	wrist_image: Optional[Image.Image],
	detected_objects: List[str],
	pred_action: str,
) -> None:
	"""Save query image, wrist image, detected objects, and VLM output for a debug frame."""
	frame_dir = debug_dir / f"ep{episode_index:04d}" / f"f{frame_index:06d}"
	frame_dir.mkdir(parents=True, exist_ok=True)
	image.save(frame_dir / "query.png")
	if wrist_image is not None:
		wrist_image.save(frame_dir / "wrist.png")
	(frame_dir / "detected_objects.txt").write_text(
		", ".join(detected_objects) if detected_objects else "none",
		encoding="utf-8",
	)
	(frame_dir / "pred_action.txt").write_text(pred_action, encoding="utf-8")


def annotate_csv_with_success(
	output_csv: Path,
	episode_subtask_gt_sets: Dict[int, List[str]],
	vlm: Any,
) -> Tuple[int, int, float]:
	rows: List[Dict[str, str]] = []
	with output_csv.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		fieldnames = list(reader.fieldnames or [])
		if "is_correct" not in fieldnames:
			fieldnames.append("is_correct")
		for row in reader:
			has_pred = str(row.get("has_prediction", "0")).strip() == "1"
			if not has_pred:
				row["is_correct"] = ""
				rows.append(row)
				continue

			ep_idx = int(row.get("episode_index", "0"))
			pred_action = row.get("pred_action", "")
			ok = _judge_action_with_qwen(vlm=vlm, pred_action=pred_action, gt_set=episode_subtask_gt_sets.get(ep_idx, []))
			row["is_correct"] = "1" if ok else "0"
			rows.append(row)

	with output_csv.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	total_pred = sum(1 for r in rows if str(r.get("has_prediction", "0")).strip() == "1")
	total_ok = sum(1 for r in rows if str(r.get("is_correct", "")).strip() == "1")
	rate = (total_ok / total_pred) if total_pred > 0 else 0.0
	return total_ok, total_pred, rate


def run_episode_online(
	episode_index: int,
	steps: List[Dict[str, Any]],
	episode_subtask_gt_set: List[str],
	image_key: str,
	args: argparse.Namespace,
	vlm: Any,
	yolo_model: Any,
	vlm_save_q: Optional[queue.Queue],
	vlm_save_stats: Optional[Dict[str, int]],
	debug_episode: Optional[int] = None,
	debug_output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
	dt = 1.0 / max(1e-6, args.playback_hz)
	history_window = max(1, int(round(2.0 * args.playback_hz)))
	frame_count = len(steps) if args.max_frames is None else min(len(steps), args.max_frames)
	if frame_count <= 0:
		return []

	gripper_tracker = GripperStateTracker(stable_frames=args.stable_frames, stable_eps=args.stable_eps)
	frames: List[np.ndarray] = []
	wrist_frames: List[Optional[np.ndarray]] = []
	for i in range(frame_count):
		step = steps[i]
		frames.append(_normalize_rgb(np.asarray(step["observation"][image_key])))
		if args.wrist_key is not None:
			wrist_value = step.get("observation", {}).get(args.wrist_key, None)
			wrist_frames.append(_normalize_rgb(np.asarray(wrist_value)) if wrist_value is not None else None)
		else:
			wrist_frames.append(None)

	shared = {
		"latest": None,
		"done": False,
		"play_index": -1,
	}
	cond = threading.Condition()
	pred_by_frame: Dict[int, Dict[str, Any]] = {}

	stream_start = time.perf_counter()

	def playback_worker() -> None:
		for frame_idx in range(frame_count):
			scheduled_ts = stream_start + frame_idx * dt
			sleep_sec = scheduled_ts - time.perf_counter()
			if sleep_sec > 0:
				time.sleep(sleep_sec)
			with cond:
				shared["latest"] = (frame_idx, scheduled_ts)
				shared["play_index"] = frame_idx
				cond.notify_all()

		with cond:
			shared["done"] = True
			cond.notify_all()

	def inference_worker() -> None:
		last_consumed_idx = -1
		while True:
			with cond:
				while True:
					latest = shared["latest"]
					done = bool(shared["done"])
					if latest is not None and latest[0] != last_consumed_idx:
						frame_idx, scheduled_ts = latest
						break
					if done:
						return
					cond.wait()

			frame = frames[frame_idx]
			wrist_frame = wrist_frames[frame_idx]
			step = steps[frame_idx]
			t_infer_start = time.perf_counter()

			gripper_pos = _extract_gripper(step)
			gripper_state = gripper_tracker.update(gripper_pos)
			one_sec = max(1, int(round(args.playback_hz)))
			idx_one_sec = max(0, frame_idx - one_sec)
			idx_two_sec = max(0, frame_idx - 2 * one_sec)
			one_sec_frame = frames[idx_one_sec]
			two_sec_frame = frames[idx_two_sec]
			prev_wrist_idx = max(0, last_consumed_idx)
			prev_wrist_frame = wrist_frames[prev_wrist_idx]
			rendered, prompt, stats = build_state_abstraction_frame(
				current_rgb=frame,
				one_sec_rgb=one_sec_frame,
				two_sec_rgb=two_sec_frame,
				gripper_state=gripper_state,
				diff_threshold=args.diff_threshold,
				min_diff_area=args.min_diff_area,
				yolo_model=yolo_model,
				wrist_rgb=wrist_frame,
				prev_wrist_rgb=prev_wrist_frame,
			)
			detected_objects = stats.get("detected_objects", [])
			det_count = int(stats.get("detections", 0))
			_enqueue_vlm_image(
				save_q=vlm_save_q,
				stats=vlm_save_stats,
				episode_index=episode_index,
				frame_index=frame_idx,
				image=rendered,
			)
			action = vlm.infer(rendered, prompt)

			t_done = time.perf_counter()
			with cond:
				play_idx_snapshot = int(shared["play_index"])

			pred_by_frame[frame_idx] = {
				"predict_wall_sec": round(t_done - stream_start, 6),
				"latency_sec": round(max(0.0, t_done - scheduled_ts), 6),
				"latency_frames": max(0, play_idx_snapshot - frame_idx),
				"processing_sec": round(t_done - t_infer_start, 6),
				"gripper_position": gripper_pos,
				"gripper_state": gripper_state,
				"detections": det_count,
				"detected_objects": ", ".join(detected_objects),
				"detected_in_task": _judge_detected_objects_with_qwen(vlm, detected_objects, episode_subtask_gt_set),
				"pred_action": action.strip(),
			}
			if debug_episode is not None and episode_index == debug_episode and debug_output_dir is not None:
				wrist_pil = Image.fromarray(wrist_frame) if wrist_frame is not None else None
				_save_debug_frame(
					debug_dir=debug_output_dir,
					episode_index=episode_index,
					frame_index=frame_idx,
					image=rendered,
					wrist_image=wrist_pil,
					detected_objects=detected_objects,
					pred_action=action.strip(),
				)
			last_consumed_idx = frame_idx

	t_play = threading.Thread(target=playback_worker, daemon=True)
	t_inf = threading.Thread(target=inference_worker, daemon=True)
	t_play.start()
	t_inf.start()
	t_play.join()
	t_inf.join()

	records: List[Dict[str, Any]] = []
	for frame_idx in range(frame_count):
		record = {
			"episode_index": episode_index,
			"frame_index": frame_idx,
			"stream_timestamp_sec": round(frame_idx * dt, 6),
			"scheduled_wall_sec": round(frame_idx * dt, 6),
			"predict_wall_sec": "",
			"latency_sec": "",
			"latency_frames": "",
			"processing_sec": "",
			"gripper_position": "",
			"gripper_state": "",
			"backend": "yolo",
			"detections": "",
			"detected_objects": "",
			"detected_in_task": 0,
			"pred_action": "",
			"has_prediction": 0,
			"history_window_frames": history_window,
		}
		if frame_idx in pred_by_frame:
			record.update(pred_by_frame[frame_idx])
			record["has_prediction"] = 1
		records.append(record)

	return records


def write_records(records: List[Dict[str, Any]], output_csv: Path) -> None:
	if not records:
		return
	output_csv.parent.mkdir(parents=True, exist_ok=True)

	fieldnames = [
		"episode_index",
		"frame_index",
		"stream_timestamp_sec",
		"scheduled_wall_sec",
		"predict_wall_sec",
		"latency_sec",
		"latency_frames",
		"processing_sec",
		"gripper_position",
		"gripper_state",
		"backend",
		"detections",
		"detected_objects",
		"detected_in_task",
		"pred_action",
		"has_prediction",
		"history_window_frames",
	]

	with output_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(records)


def main() -> None:
	parser = argparse.ArgumentParser(description="DROID-100 online benchmark (YOLO + Qwen VLM).")
	parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/droid_100/1.0.0"))
	parser.add_argument("--split", type=str, default="train")
	parser.add_argument("--max-episodes", type=int, default=100)
	parser.add_argument("--max-frames", type=int, default=None)
	parser.add_argument("--image-key", type=str, default=None)
	parser.add_argument("--wrist-key", type=str, default=None)
	parser.add_argument("--playback-hz", type=float, default=15.0)
	parser.add_argument("--diff-threshold", type=int, default=24)
	parser.add_argument("--min-diff-area", type=int, default=250)
	parser.add_argument("--stable-frames", type=int, default=5)
	parser.add_argument("--stable-eps", type=float, default=1e-6)
	parser.add_argument("--yolo-model-path", type=str, default="yolov8l-worldv2.pt")
	parser.add_argument("--model", type=str, choices=["qwen-vl-4b", "qwen-vl-2b"], default="qwen-vl-4b")
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument(
		"--debug",
		type=int,
		default=None,
		help="If specified, run only this episode index and save query images, detected objects, and VLM outputs for all predicted frames.",
	)
	parser.add_argument("--vlm-input-dir", type=Path, default=Path("dataset/droid_100/benchmark_vlm_inputs"))
	parser.add_argument("--vlm-save-queue-size", type=int, default=1024)
	parser.add_argument("--evaluate", action="store_true", help="Use Qwen to judge prediction correctness and append is_correct column to CSV.")
	parser.add_argument("--subtask-gt-path", type=Path, default=Path("dataset/droid_100/subtask_gt_set.json"))
	parser.add_argument("--output-csv", type=Path, default=Path("dataset/droid_100/benchmark_results.csv"))
	args = parser.parse_args()

	try:
		tfds = importlib.import_module("tensorflow_datasets")
	except ImportError as exc:
		raise ImportError("tensorflow-datasets is required. Install with: pip install tensorflow-datasets") from exc

	builder = tfds.builder_from_directory(builder_dir=str(args.dataset_dir))

	vlm = _load_vlm(args)
	yolo_model = _load_yolo(args)
	episode_subtask_gt_sets: Dict[int, List[str]] = load_subtask_gt_sets(args.subtask_gt_path) if args.subtask_gt_path.exists() else {}
	debug_output_dir = None
	if args.debug is not None:
		debug_output_dir = Path("dataset/droid_100/debug_frames")
		debug_output_dir.mkdir(parents=True, exist_ok=True)
		args.max_episodes = args.debug + 1
	vlm_save_q, vlm_save_thread, vlm_save_stats = _make_vlm_image_saver(args)

	all_records: List[Dict[str, Any]] = []
	episode_gt_sets: Dict[int, List[str]] = {}
	for ep_idx, episode in _iter_episodes(builder, args.split, tfds, max_episodes=args.max_episodes):
		if args.debug is not None and ep_idx != args.debug:
			continue
		steps = _to_step_list(episode.get("steps", []))
		if not steps:
			continue
		episode_gt_sets[ep_idx] = _extract_episode_gt_set(steps)
		episode_subtask_gt_set = episode_subtask_gt_sets.get(ep_idx)
		if episode_subtask_gt_set is None:
			episode_subtask_gt_set = build_subtask_gt_sets({ep_idx: episode_gt_sets[ep_idx]}).get(ep_idx, episode_gt_sets[ep_idx])
			episode_subtask_gt_sets[ep_idx] = episode_subtask_gt_set

		image_key = _find_image_key(steps[0], explicit_key=args.image_key)
		args.wrist_key = _find_wrist_key(steps[0], explicit_key=args.wrist_key)
		history_window = max(1, int(round(2.0 * args.playback_hz)))
		print(
			f"[Episode {ep_idx}] steps={len(steps)}, image_key={image_key}, "
			f"playback_hz={args.playback_hz:.3f}, history_window_frames={history_window}"
		)

		records = run_episode_online(
			episode_index=ep_idx,
			steps=steps,
			episode_subtask_gt_set=episode_subtask_gt_set,
			image_key=image_key,
			args=args,
			vlm=vlm,
			yolo_model=yolo_model,
			vlm_save_q=vlm_save_q,
			vlm_save_stats=vlm_save_stats,
			debug_episode=args.debug,
			debug_output_dir=debug_output_dir,
		)
		all_records.extend(records)
		pred_count = sum(int(r.get("has_prediction", 0)) for r in records)
		print(f"[Episode {ep_idx}] frames={len(records)}, predictions={pred_count}, dropped={len(records) - pred_count}")

	if not args.subtask_gt_path.exists():
		save_subtask_gt_sets(args.subtask_gt_path, episode_subtask_gt_sets)

	save_experiment_results(all_records, output_csv=args.output_csv)
	if vlm_save_q is not None:
		vlm_save_q.put(None)
	if vlm_save_thread is not None:
		vlm_save_thread.join()

	success_ok, success_total, success_rate = (0, 0, 0.0)
	if args.evaluate:
		eval_stats = evaluate_experiment_results(
			output_csv=args.output_csv,
			episode_subtask_gt_sets=episode_subtask_gt_sets,
			vlm=vlm,
			object_col="detected_objects",
		)
		success_ok = int(eval_stats["task_correct"])
		success_total = int(eval_stats["task_total"])
		success_rate = float(eval_stats["task_rate"])

	print("=== Benchmark done ===")
	if args.debug is not None:
		print(f"debug_mode: episode {args.debug}")
		print(f"debug_frames_dir: {debug_output_dir}")
	else:
		print(f"episodes: {args.max_episodes}")
	print(f"records: {len(all_records)}")
	obj_judge_total = len(all_records)
	obj_judge_hit = sum(int(r.get("detected_in_task", 0)) for r in all_records)
	obj_judge_pct = (100.0 * obj_judge_hit / obj_judge_total) if obj_judge_total > 0 else 0.0
	print(f"object_detect_in_task: {obj_judge_hit}/{obj_judge_total} = {obj_judge_pct:.2f}%")
	if args.evaluate:
		print(f"success: {success_ok}/{success_total} = {success_rate:.4f}")
	print(f"saved_to: {args.output_csv}")


if __name__ == "__main__":
	main()
