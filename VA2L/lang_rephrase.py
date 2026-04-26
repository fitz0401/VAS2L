from __future__ import annotations

import argparse
import sys
from typing import Optional
from VA2L.vlm_inference import VLMInference


_REPHRASE_PROMPT = """Your goal is to convert user input into the specific style aligned with the training data.

### Rules:
1. Task Validation: If the user's instruction is NOT related to physical robot manipulation (e.g., general questions, chat, math, code, or abstract requests), output exactly: "[NOT_A_MANIPULATION_TASK]".
2. Imperative Style: Remove all politeness (please, can you, etc.).
3. Avoid Making Up New Elements: Do not make up new objects, relations, or actions that are not in the original instruction.
4. Preserve Grounding Details: Keep essential visual adjectives (color, material, texture) if provided (e.g., "the red bowl" should not become just "the bowl").
5. Strict Patterns: Fit the instruction into the dataset patterns. Prioritize the template: [Verb] + [Object with descriptors] + [Spatial Relation] + [Target/Location].

### Dataset Patterns:
- Put [Object] in/on/into [Target]
- Pick up [Object] from [Source]
- Move [Object] [Direction/Relative Position]
- Open/Close [Articulated Object like drawer/cabinet/lid]
- Wipe [Target] with [Tool]
- Pour [Content] from [Source] into [Target]
- Fold/Unfold [Deformable Object like towel/cloth]
- Turn on/off [Switch/Appliance]

### Example Rewrites:
- "Could you please put that green marker into the plastic cup?" -> "Put the green marker in the plastic cup"
- "I need you to move the sponge to the left side of the sink." -> "Move the sponge to the left of the sink"
- "Can you open the top drawer of the desk?" -> "Open the top drawer"
- "Tell me a joke about robots." -> "[NOT_A_MANIPULATION_TASK]"
- "What is the capital of France?" -> "[NOT_A_MANIPULATION_TASK]"
- "Pick up the checkered shirt and place it on the bed." -> "Put the checkered shirt on the bed"

Now, rewrite the following instruction:
"""

def _build_rephrase_prompt(instruction: str) -> str:
    return f"{_REPHRASE_PROMPT}\nUser instruction: {instruction.strip()}\nRewritten instruction:"


def rephrase_instruction(
    instruction: str,
    model: str = "qwen-vl-4b",
    model_id: str | None = None,
    device: str = "cuda:0",
    precision: str = "auto",
    vlm: Optional[VLMInference] = None,
) -> str:
    """Rephrase a free-form instruction into a Droid-style command."""
    cleaned_instruction = instruction.strip()
    if not cleaned_instruction:
        return ""

    rewriter = vlm or VLMInference(
        model=model,
        model_id=model_id,
        device=device,
        precision=precision,
    )
    return rewriter.infer_text(_build_rephrase_prompt(cleaned_instruction)).strip()


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
    args = parser.parse_args()

    instruction = args.instruction
    if instruction is None:
        instruction = sys.stdin.read().strip()

    result = rephrase_instruction(
        instruction=instruction,
        model=args.model,
        device=args.device,
        precision=args.precision,
    )
    print(result)


if __name__ == "__main__":
    main()