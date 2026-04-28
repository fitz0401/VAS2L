from __future__ import annotations

import argparse
import random
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml

from success_module.success_dataloader import SuccessFrameDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one sampled video and print 0/1 label ranges")
    parser.add_argument("--config", type=str, default="success_module/config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for video sampling")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())

    dataset = SuccessFrameDataset(config_path)
    video_paths = sorted({s.video_path for s in dataset.samples})
    if not video_paths:
        raise RuntimeError("No videos found in dataset")

    print(f"[INFO] label_mode: {dataset.label_mode}")
    print(f"[INFO] positive samples: {dataset.positive_count}")
    print(f"[INFO] negative samples: {dataset.negative_count}")
    print(f"[INFO] positive ratio: {dataset.positive_ratio:.4f}")
    print(f"[INFO] negative ratio: {dataset.negative_ratio:.4f}")

    rng = random.Random(args.seed)
    video_path = rng.choice(video_paths)
    samples = [s for s in dataset.samples if s.video_path == video_path]
    samples = sorted(samples, key=lambda x: (x.segment_index, x.frame_idx))

    segments = {}
    for sample in samples:
        segments.setdefault(sample.segment_index, []).append(sample)

    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Dataset root: {cfg['dataset']['root']}")
    print(f"[INFO] Sampled video: {video_path}")
    print(f"[INFO] Total labeled frames in sampled video: {len(samples)}")

    for segment_index in sorted(segments):
        segment_samples = segments[segment_index]
        positive = [s.frame_idx for s in segment_samples if s.label == 1]
        negative = [s.frame_idx for s in segment_samples if s.label == 0]
        keyframe_idx = segment_samples[0].keyframe_idx
        transition_type = segment_samples[0].transition_type
        print(f"[INFO] segment {segment_index} ({transition_type}) keyframe: {keyframe_idx}")
        if positive:
            start_pos = min(positive)
            end_pos = max(positive)
            print(f"[INFO]   label 1 range: {start_pos} -> {end_pos} ({start_pos / dataset.fps:.3f}s -> {end_pos / dataset.fps:.3f}s)")
        else:
            print("[INFO]   label 1 range: none")
        if negative:
            start_neg = min(negative)
            end_neg = max(negative)
            print(f"[INFO]   label 0 range: {start_neg} -> {end_neg} ({start_neg / dataset.fps:.3f}s -> {end_neg / dataset.fps:.3f}s)")
        else:
            print("[INFO]   label 0 range: none")


if __name__ == "__main__":
    main()
