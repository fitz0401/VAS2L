#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "$PYTHON_BIN" -c "import torch, cv2, pyarrow" >/dev/null 2>&1; then
  if [ -x "/home/ze/miniconda3/envs/vas2l/bin/python" ]; then
    PYTHON_BIN="/home/ze/miniconda3/envs/vas2l/bin/python"
  fi
fi

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import random

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from success_module.success_model import build_model, load_checkpoint, predict_image, preprocess_bgr_image

repo_root = Path('/home/ze/VAS2L')
config_path = repo_root / 'success_module' / 'config.yaml'
cfg = yaml.safe_load(config_path.read_text())
label_cfg = cfg['labeling']
positive_seconds_before_keyframe = float(label_cfg.get('positive_seconds_before_keyframe', 2.0))
gipper_action_index = int(label_cfg.get('gripper_action_index', -1))

dataset_root = repo_root / 'dataset' / 'open_drawer_droid'
video_root = dataset_root / 'videos' / 'chunk-000'
data_root = dataset_root / 'data' / 'chunk-000'
info_path = dataset_root / 'meta' / 'info.json'
ckpt_path = repo_root / 'ckpts' / 'success_model' / 'best.pt'

if not info_path.exists():
	raise FileNotFoundError(f'Missing {info_path}')
if not ckpt_path.exists():
	raise FileNotFoundError(f'Missing checkpoint {ckpt_path}')

info = json.loads(info_path.read_text())
fps = int(info.get('fps', 15))

video_paths = sorted([p for p in video_root.glob('**/*.mp4') if 'wrist' in str(p)])
if not video_paths:
	raise RuntimeError(f'No wrist videos found under {video_root}')

rng = random.Random(42)
video_path = rng.choice(video_paths)
video_key = video_path.stem
episode_index = int(video_key.split('_')[-1])
parquet_path = data_root / f'episode_{episode_index:06d}.parquet'

if not parquet_path.exists():
	raise FileNotFoundError(f'Missing parquet for episode {episode_index}: {parquet_path}')

table = pq.read_table(parquet_path, columns=['actions'])
actions = np.asarray(table['actions'].to_pylist(), dtype=np.float32)
if actions.ndim != 2:
	raise ValueError(f'Invalid actions shape: {actions.shape}')
idx = gipper_action_index
if idx < 0:
	idx = actions.shape[1] + idx
gripper = actions[:, idx]

transitions = np.where((gripper[:-1] < 0.5) & (gripper[1:] > 0.5))[0]
if len(transitions) == 0:
	raise RuntimeError(f'No 0->1 gripper transition found in {parquet_path}')
keyframe_idx = int(transitions[0] + 1)

positive_frames = max(1, int(round(positive_seconds_before_keyframe * fps)))
window_start = max(0, keyframe_idx - positive_frames)
frame_indices = list(range(window_start, keyframe_idx + 1))

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
	raise RuntimeError(f'Cannot open video: {video_path}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(pretrained=False, device=device)
if ckpt_path.exists():
	checkpoint = load_checkpoint(model, ckpt_path, map_location=device)
	epoch = checkpoint.get('epoch', 'unknown')
else:
	epoch = 'N/A'
model.eval()

print(f'[INFO] dataset_root={dataset_root}')
print(f'[INFO] video={video_path.name}')
print(f'[INFO] parquet={parquet_path.name}')
print(f'[INFO] fps={fps}')
print(f'[INFO] positive_seconds_before_keyframe={positive_seconds_before_keyframe}')
print(f'[INFO] checkpoint={ckpt_path.name} epoch={epoch}')
print(f'[INFO] keyframe_idx={keyframe_idx} gripper_transition={gripper[keyframe_idx-1]:.3f}->{gripper[keyframe_idx]:.3f}')
print(f'[INFO] output_frames={frame_indices[0]}..{frame_indices[-1]} (count={len(frame_indices)})')
print('[INFO] frame_idx | gripper | pred | prob_0 | prob_1 | note')

for frame_idx in frame_indices:
	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
	ok, frame = cap.read()
	if not ok:
		print(f'[WARN] Failed to read frame {frame_idx}')
		continue
	image_tensor = preprocess_bgr_image(frame)
	result = predict_image(model, image_tensor, device=device)
	note = 'positive' if frame_idx >= window_start else 'negative'
	print(f'{frame_idx:7d} | {gripper[frame_idx]:7.3f} | {result["pred"]:4d} | {result["prob_0"]:.4f} | {result["prob_1"]:.4f} | {note}')

cap.release()
print('[INFO] Test completed successfully')
PY
