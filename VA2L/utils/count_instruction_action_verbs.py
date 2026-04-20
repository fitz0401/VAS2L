from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

INSTRUCTION_KEYS = (
    "language_instruction:",
    "language_instruction_2:",
    "language_instruction_3:",
)

# Common phrasal/action forms in manipulation instructions.
MULTIWORD_VERBS = (
    "pick up",
    "put down",
    "take out",
    "turn off",
    "turn on",
    "move away",
)

# Small stop set to avoid counting connectors as verbs.
STOP_WORDS = {
    "the",
    "a",
    "an",
    "to",
    "and",
    "then",
    "from",
    "of",
    "on",
    "in",
    "into",
    "inside",
    "outside",
    "with",
    "by",
    "up",
    "down",
    "left",
    "right",
    "middle",
    "center",
    "top",
    "bottom",
}


def _extract_instruction_texts(lines: Sequence[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        for key in INSTRUCTION_KEYS:
            if stripped.startswith(key):
                text = stripped[len(key) :].strip()
                if text:
                    out.append(text)
                break
    return out


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _find_multiword_verbs(text: str) -> tuple[str, List[tuple[int, int]]]:
    spans: List[tuple[int, int]] = []
    tokens: List[str] = []
    for phrase in MULTIWORD_VERBS:
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            spans.append((match.start(), match.end()))
            tokens.append(phrase.replace(" ", "_"))

    return " ".join(tokens), spans


def _is_inside_any(index: int, spans: Iterable[tuple[int, int]]) -> bool:
    for start, end in spans:
        if start <= index < end:
            return True
    return False


def _extract_action_verbs(text: str) -> List[str]:
    normalized = _normalize_text(text)
    multiword_joined, spans = _find_multiword_verbs(normalized)

    # Remove punctuation but keep word boundaries for regex token extraction.
    cleaned = re.sub(r"[^a-z\s]", " ", normalized)
    words = [w for w in cleaned.split() if w and w not in STOP_WORDS]

    verbs: List[str] = []
    if multiword_joined:
        verbs.extend(multiword_joined.split())

    for match in re.finditer(r"\b[a-z]+\b", normalized):
        idx = match.start()
        word = match.group(0)

        if _is_inside_any(idx, spans):
            continue
        if word in STOP_WORDS:
            continue

        # Heuristic: keep typical imperative/action words seen in DROID instructions.
        # Includes simple base verbs and action-like forms.
        if re.fullmatch(r"(put|pick|place|move|take|get|open|close|turn|slide|press|flip|remove|arrange|use|wipe|lift|center|fold|insert)", word):
            verbs.append(word)

    # Fallback: if nothing matched, use the first token as a proxy action word.
    if not verbs and words:
        verbs.append(words[0])

    return verbs


def analyze_instruction_file(path: Path) -> tuple[Counter, int, int]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    instructions = _extract_instruction_texts(lines)

    verb_counter: Counter[str] = Counter()
    instruction_with_verb = 0

    for instruction in instructions:
        verbs = _extract_action_verbs(instruction)
        if verbs:
            instruction_with_verb += 1
            verb_counter.update(verbs)

    return verb_counter, len(instructions), instruction_with_verb


def main() -> None:
    parser = argparse.ArgumentParser(description="Count action verbs from language instructions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/droid_100/all_episode_instructions.txt"),
        help="Path to all_episode_instructions.txt",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Number of verbs to print")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    counter, total_instructions, instructions_with_verb = analyze_instruction_file(args.input)
    total_verbs = sum(counter.values())

    print(f"input: {args.input}")
    print(f"total_instructions: {total_instructions}")
    print(f"instructions_with_action_verb: {instructions_with_verb}")
    print(f"total_action_verb_tokens: {total_verbs}")
    print()
    print("verb,count,ratio")

    if total_verbs == 0:
        return

    for verb, cnt in counter.most_common(max(1, args.top_k)):
        ratio = cnt / total_verbs
        print(f"{verb},{cnt},{ratio:.4f}")


if __name__ == "__main__":
    main()
