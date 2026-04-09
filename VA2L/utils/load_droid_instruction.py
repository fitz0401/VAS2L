"""Export DROID episode instructions to a text file.

The DROID RLDS episodes store the same instruction strings at every step within
an episode, so this script reads only the first step of each episode and writes
the three instruction fields once per episode.

Example:
    python -m VA2L.utils.load_droid_instruction \
        --dataset-dir /home/ze/VAS2L/dataset/droid_100/1.0.0 \
        --save-txt /home/ze/VAS2L/dataset/droid_100/all_episode_instructions.txt
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
    return _decode_if_bytes(value)


def _extract_instructions(step: Dict[str, Any]) -> Dict[str, str]:
    return {
        "language_instruction": str(_decode_if_bytes(step.get("language_instruction", ""))),
        "language_instruction_2": str(_decode_if_bytes(step.get("language_instruction_2", ""))),
        "language_instruction_3": str(_decode_if_bytes(step.get("language_instruction_3", ""))),
    }


def _first_step_from_episode(episode: Dict[str, Any], tfds: Any) -> Dict[str, Any]:
    steps = episode.get("steps", [])

    if hasattr(steps, "take"):
        try:
            return next(iter(tfds.as_numpy(steps.take(1))))
        except StopIteration as exc:
            raise RuntimeError("Episode has no steps.") from exc

    if hasattr(steps, "as_numpy_iterator"):
        try:
            return next(iter(steps.as_numpy_iterator()))
        except StopIteration as exc:
            raise RuntimeError("Episode has no steps.") from exc

    if isinstance(steps, Iterable):
        step_list = list(steps)
        if not step_list:
            raise RuntimeError("Episode has no steps.")
        return step_list[0]

    raise RuntimeError("Unsupported steps container in episode record.")


def _iter_episodes(builder: Any, split: str, tfds: Any) -> Iterable[Dict[str, Any]]:
    dataset = builder.as_dataset(split=split)
    return tfds.as_numpy(dataset)


def _format_episode_block(
    episode_index: int,
    episode_metadata: Dict[str, Any],
    instructions: Dict[str, str],
) -> str:
    lines = [
        f"episode_index: {episode_index}",
        f"file_path: {_decode_if_bytes(episode_metadata.get('file_path', ''))}",
        f"recording_folderpath: {_decode_if_bytes(episode_metadata.get('recording_folderpath', ''))}",
        f"language_instruction: {instructions['language_instruction']}",
        f"language_instruction_2: {instructions['language_instruction_2']}",
        f"language_instruction_3: {instructions['language_instruction_3']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all DROID episode instructions into one txt file."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/droid_100/1.0.0"),
        help="Directory containing dataset_info.json/features.json and TFRecord shards.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split name.")
    parser.add_argument(
        "--save-txt",
        type=Path,
        default=Path("dataset/droid_100/all_episode_instructions.txt"),
        help="Where to save the per-episode instruction summary.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episodes to export.",
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

    if args.split not in builder.info.splits:
        available_splits = ", ".join(sorted(builder.info.splits.keys()))
        raise KeyError(f"split '{args.split}' not found. Available splits: {available_splits}")

    split_info = builder.info.splits[args.split]
    total_episodes = int(split_info.num_examples)
    if total_episodes <= 0:
        raise RuntimeError(f"Split '{args.split}' contains no episodes.")

    episode_limit = total_episodes if args.max_episodes is None else min(total_episodes, args.max_episodes)

    lines: List[str] = []
    lines.append(f"dataset_dir: {args.dataset_dir}")
    lines.append(f"split: {args.split}")
    lines.append(f"total_episodes: {total_episodes}")
    lines.append(f"exported_episodes: {episode_limit}")
    lines.append("")

    for episode_index, episode in enumerate(_iter_episodes(builder=builder, split=args.split, tfds=tfds)):
        if episode_index >= episode_limit:
            break

        first_step = _first_step_from_episode(episode, tfds=tfds)
        instructions = _extract_instructions(first_step)
        episode_metadata = _to_jsonable(episode.get("episode_metadata", {}))

        lines.append(
            _format_episode_block(
                episode_index=episode_index,
                episode_metadata=episode_metadata,
                instructions=instructions,
            )
        )
        lines.append("-")

    args.save_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.save_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print("=== Instruction export ===")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"split: {args.split}")
    print(f"episodes_exported: {episode_limit}")
    print(f"saved_to: {args.save_txt}")


if __name__ == "__main__":
    main()