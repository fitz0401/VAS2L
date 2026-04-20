"""Recognize Anything (RAM++) wrapper used by VA2L pipelines."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image

# RAM's vendored BERT code expects old transformers symbols and does not require TF.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


DEFAULT_RAM_CKPT = Path("ckpts/ram_plus_swin_large_14m.pth")


def _split_tags(tag_text: str) -> List[str]:
    if not tag_text:
        return []
    parts = re.split(r"\s*\|\s*|\s*,\s*|\s*;\s*", str(tag_text).strip())
    out = []
    seen = set()
    for p in parts:
        t = p.strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


class RAMRecognizer:
    """Small reusable RAM++ recognizer with normalized tag output."""

    def __init__(self, pretrained: str | Path = DEFAULT_RAM_CKPT, image_size: int = 384, device: str | None = None):
        self._patch_transformers_for_ram()
        from ram import get_transform
        from ram.models import ram_plus

        self.pretrained = str(pretrained)
        self.image_size = int(image_size)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.transform = get_transform(image_size=self.image_size)
        self.model = ram_plus(pretrained=self.pretrained, image_size=self.image_size, vit="swin_l")
        self.model.eval()
        self.model = self.model.to(self.device)
        self._inference = None

    @staticmethod
    def _patch_transformers_for_ram() -> None:
        """Backfill symbols removed from modeling_utils in newer transformers."""
        try:
            import transformers.modeling_utils as modeling_utils
            import transformers.pytorch_utils as pytorch_utils
        except Exception:
            return

        for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
            if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))

    def recognize(self, image: Image.Image) -> Tuple[List[str], str, str]:
        """Return normalized tags and raw english/chinese RAM outputs."""
        if self._inference is None:
            from ram import inference_ram as inference
            self._inference = inference
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        res = self._inference(tensor, self.model)
        english = str(res[0]) if len(res) > 0 else ""
        chinese = str(res[1]) if len(res) > 1 else ""
        tags = _split_tags(english)
        return tags, english, chinese


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize objects from one image using RAM++.")
    parser.add_argument("--image", type=Path, default=Path("images/demo/demo1.jpg"), help="Input image path.")
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_RAM_CKPT, help="RAM++ checkpoint path.")
    parser.add_argument("--image-size", type=int, default=384, help="Input image size.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda or cpu.")
    args = parser.parse_args()

    recognizer = RAMRecognizer(pretrained=args.pretrained, image_size=args.image_size, device=args.device)
    image = Image.open(args.image).convert("RGB")
    tags, raw_en, raw_zh = recognizer.recognize(image)
    print(f"image: {args.image}")
    print(f"tags: {tags}")
    print(f"raw_en: {raw_en}")
    print(f"raw_zh: {raw_zh}")


if __name__ == "__main__":
    main()