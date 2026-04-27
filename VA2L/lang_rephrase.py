from __future__ import annotations

import argparse
import json
import re
import sys
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
    "yellow teacup",
    "blue cup",
    "marker pen",
    "red apple"
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
- Consider spelling and phonetic similarity for object names (e.g. jar -> drawer, cup -> cap, grip -> grape, gray -> grape, door -> drawer).
- Use semantic similarity between instruction and templates (e.g. pen -> marker, grasp -> pick).
- Consider affordance and interaction commonsense (e.g., containers -> into, articulated objects -> open/close).

Output format:
Return JSON only, no extra text:
{"template_id": <int>, "object_id": <int>, "target_id": <int>}

Rules:
- If selected template does not use target, set target_id to -1.
- "template_id" and "object_id" MUST be valid indices in the template and object lists, respectively.
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
    args = parser.parse_args()

    instruction = args.instruction
    if instruction is None:
        instruction = sys.stdin.read().strip()

    detected_objects = _parse_detected_objects(args.detected_objects)
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