from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from typing import List, Optional
from VA2L.vlm_inference import VLMInference


TEMPLATES = [
    "pick the {object}",
    "place the {object} into the {target}",
    "insert the {object} into the {target}",
    "move the {object} next to the {target}",
    "open the {object}",
    "close the {object}",
    "fold the {object}",
]

DEFAULT_DETECTED_OBJECTS = [
    "green grape",
    "white drawer",
    "yellow cup",
    "red mug",
    "marker pen",
]


_SELECTION_PROMPT = """You are a robotics instruction parser.

Task:
Given a user instruction, select object/target from detected-object candidates, then do a strict selection from fixed task templates.

You MUST select object_id and target_id from this list: {object_choices}

You MUST select exactly one template ID from this list:
0: pick the {object}
1: place the {object} into the {target}
2: insert the {object} into the {target}
3: move the {object} next to the {target}
4: open the {object}
5: close the {object}
6: fold the {object}

Selection hint:
- Objects in same category are allowed (e.g. pen -> marker).
- Use semantic similarity between instruction and templates.
- Consider affordance and interaction commonsense (e.g., containers -> into, articulated objects -> open/close).

Output format:
Return JSON only, no extra text:
{"template_id": <int>, "object_id": <int>, "target_id": <int>}

Rules:
- If selected template does not use target, set target_id to -1.
- If object/target is semantically inconsistent with all detected candidates, output template_id=-1, object_id=-1, target_id=-1.
"""


def _build_selection_prompt(instruction: str, detected_objects: List[str]) -> str:
    object_choices = "\n".join(f"{idx}: {name}" for idx, name in enumerate(detected_objects))
    prompt = _SELECTION_PROMPT.replace("{object_choices}", object_choices)
    return f"{prompt}\nUser instruction: {instruction.strip()}\nJSON:"


def _extract_json_blob(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else ""


def _render_template(template_id: int, obj: str, target: str | None) -> str:
    if template_id < 0 or template_id >= len(TEMPLATES):
        raise ValueError("Invalid template_id")
    template = TEMPLATES[template_id]

    if "{target}" in template:
        if not target:
            raise ValueError("Target is required for selected template")
        return template.format(object=obj.strip(), target=target.strip())
    return template.format(object=obj.strip())


def _parse_detected_objects(text: str) -> List[str]:
    items = [re.sub(r"\s+", " ", part).strip(" .\t") for part in re.split(r",|;|\n|\|", text)]
    out: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        lowered = item.lower()
        if lowered in {"none", "n/a", "na"}:
            continue
        if lowered not in seen:
            seen.add(lowered)
            out.append(item)
    return out


def _run_online_detected_objects(
    device_id: str,
    width: int,
    height: int,
    fps: int,
    model: str,
    timeout_sec: float,
) -> List[str]:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    pipeline = rs.pipeline()
    config = rs.config()
    if device_id:
        config.enable_device(device_id)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    frame_rgb = None
    try:
        pipeline.start(config)
        for _ in range(20):
            pipeline.wait_for_frames()

        frameset = pipeline.wait_for_frames()
        color_frame = frameset.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to read color frame from camera")

        bgr = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        pipeline.stop()

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode camera frame")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    prompt_text = (
        "You are a tabletop object detector for robot manipulation. "
        "Ignore robot arm/gripper/body, table, environment background, walls, floor, and room context. "
        "Output exactly one line only: comma-separated object descriptions with color, such as 'red mug, blue box'. "
        "If no valid tabletop object, output exactly: none"
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        timeout=max(1.0, float(timeout_sec)),
    )

    t_start = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ],
            }
        ],
        extra_body={"enable_thinking": False},
    )
    elapsed = time.perf_counter() - t_start

    raw_text = "none"
    for item in getattr(response, "output", []):
        if getattr(item, "type", "") != "message":
            continue
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if text:
                raw_text = str(text).strip()
                break
        if raw_text != "none":
            break

    first_line = raw_text.splitlines()[0] if raw_text else "none"
    objects = _parse_detected_objects(first_line)
    print(f"[online-detected] elapsed={elapsed:.3f}s raw={raw_text}")
    return objects


def rephrase_instruction(
    instruction: str,
    detected_objects: Optional[List[str]] = None,
    model: str = "qwen-vl-4b",
    model_id: str | None = None,
    device: str = "cuda:0",
    precision: str = "auto",
    vlm: Optional[VLMInference] = None,
) -> str:
    """Map a free-form instruction to a fixed template plus slot filling."""
    cleaned_instruction = instruction.strip()
    if not cleaned_instruction:
        return ""

    object_candidates = detected_objects or DEFAULT_DETECTED_OBJECTS
    if not object_candidates:
        return "none"

    rewriter = vlm or VLMInference(
        model=model,
        model_id=model_id,
        device=device,
        precision=precision,
    )
    raw = rewriter.infer_text(_build_selection_prompt(cleaned_instruction, object_candidates)).strip()

    try:
        payload = json.loads(_extract_json_blob(raw))
        template_id = int(payload["template_id"])
        object_id = int(payload["object_id"])
        target_id = int(payload.get("target_id", -1))

        if template_id == -1 or object_id == -1:
            return "none"
        if template_id < 0 or template_id >= len(TEMPLATES):
            return "none"
        if object_id < 0 or object_id >= len(object_candidates):
            return "none"

        obj = object_candidates[object_id]        
        target = None
        if "{target}" in TEMPLATES[template_id]:
            if target_id < 0 or target_id >= len(object_candidates):
                return "none"
            target = object_candidates[target_id]
        return _render_template(template_id, obj, target)
    except Exception:
        return "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rephrase a language instruction into Droid-style text")
    parser.add_argument("instruction", nargs="?", default=None, help="Input instruction to rephrase")
    parser.add_argument("--device", type=str, default="cuda:1", help="Model device")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-vl-4b",
        choices=["qwen-vl-4b", "qwen-vl-8b", "qwen-vl-2b"],
        help="Model to use for rephrase",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Model precision",
    )
    parser.add_argument(
        "--detected-objects",
        type=str,
        default=", ".join(DEFAULT_DETECTED_OBJECTS),
        help="Comma-separated detected object candidates",
    )
    parser.add_argument(
        "--online-detected",
        action="store_true",
        help="Use camera + online VLM to replace detected object candidates",
    )
    parser.add_argument("--online-object-model", type=str, default="qwen3.5-plus", help="Online VLM model name")
    parser.add_argument("--device-id", type=str, default="f1421698", help="RealSense device id")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera fps")
    parser.add_argument("--online-timeout-sec", type=float, default=20.0, help="Online VLM timeout in seconds")
    args = parser.parse_args()

    instruction = args.instruction
    if instruction is None:
        instruction = sys.stdin.read().strip()

    detected_objects = _parse_detected_objects(args.detected_objects)
    if not detected_objects:
        detected_objects = DEFAULT_DETECTED_OBJECTS.copy()

    if args.online_detected:
        try:
            online_objects = _run_online_detected_objects(
                device_id=args.device_id,
                width=args.width,
                height=args.height,
                fps=args.fps,
                model=args.online_object_model,
                timeout_sec=args.online_timeout_sec,
            )
            if online_objects:
                detected_objects = online_objects
            else:
                detected_objects = ["none"]
        except Exception as exc:
            print(f"[online-detected] warning: {exc}. fallback to defaults.", file=sys.stderr)
            if not detected_objects:
                detected_objects = DEFAULT_DETECTED_OBJECTS.copy()

    result = rephrase_instruction(
        instruction=instruction,
        detected_objects=detected_objects,
        model=args.model,
        device=args.device,
        precision=args.precision,
    )
    print(result)


if __name__ == "__main__":
    main()