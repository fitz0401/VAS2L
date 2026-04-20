from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def save_experiment_results(records: List[Dict[str, Any]], output_csv: Path) -> None:
	if not records:
		return
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = list(records[0].keys())
	with output_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(records)


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


def _read_rows(output_csv: Path) -> Tuple[List[Dict[str, str]], List[str]]:
	with output_csv.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)
		fieldnames = list(reader.fieldnames or [])
	return rows, fieldnames


def _write_rows(output_csv: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def evaluate_experiment_results(
	output_csv: Path,
	episode_subtask_gt_sets: Dict[int, List[str]],
	vlm: Any,
	task_pred_col: str = "pred_action",
	object_col: Optional[str] = None,
	episode_index_col: str = "episode_index",
	frame_index_col: str = "frame_index",
) -> Dict[str, float]:
	rows, fieldnames = _read_rows(output_csv)
	if not rows:
		return {
			"task_correct": 0.0,
			"task_total": 0.0,
			"task_rate": 0.0,
			"object_correct": 0.0,
			"object_total": 0.0,
			"object_rate": 0.0,
		}

	task_col = "task_is_correct"
	obj_col = "object_is_correct"
	if task_col not in fieldnames:
		fieldnames.append(task_col)
	if object_col is not None and obj_col not in fieldnames:
		fieldnames.append(obj_col)

	seen_object_episode = set()
	task_correct = 0
	task_total = 0
	object_correct = 0
	object_total = 0

	for row in rows:
		ep_idx = int(row.get(episode_index_col, "0") or 0)
		gt_set = episode_subtask_gt_sets.get(ep_idx, [])
		pred_action = row.get(task_pred_col, "")
		if pred_action.strip():
			task_total += 1
			task_ok = _judge_action_with_qwen(vlm, pred_action=pred_action, gt_set=gt_set)
			row[task_col] = "1" if task_ok else "0"
			task_correct += int(task_ok)
		else:
			row[task_col] = ""

		if object_col is not None:
			if ep_idx not in seen_object_episode:
				seen_object_episode.add(ep_idx)
				object_text = row.get(object_col, "")
				object_list = [x.strip() for x in object_text.split("|") if x.strip()]
				if object_list:
					object_total += 1
					obj_ok = _judge_detected_objects_with_qwen(vlm, detected_objects=object_list, gt_set=gt_set)
					row[obj_col] = str(obj_ok)
					object_correct += obj_ok
				else:
					row[obj_col] = ""
			else:
				row[obj_col] = ""

	_write_rows(output_csv, rows, fieldnames)

	return {
		"task_correct": float(task_correct),
		"task_total": float(task_total),
		"task_rate": (float(task_correct) / float(task_total)) if task_total > 0 else 0.0,
		"object_correct": float(object_correct),
		"object_total": float(object_total),
		"object_rate": (float(object_correct) / float(object_total)) if object_total > 0 else 0.0,
	}
