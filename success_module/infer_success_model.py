from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import torch

from success_module.success_model import build_model, load_checkpoint, predict_image, preprocess_bgr_image


def load_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return preprocess_bgr_image(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer success model on a single image")
    parser.add_argument("--ckpt", type=str, default="ckpts/success_model/best.pt")
    parser.add_argument("--image", type=str, required=True, help="Image path to run inference on")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=True, device=device)
    checkpoint = load_checkpoint(model, Path(args.ckpt).resolve(), map_location=device)
    model.eval()

    print(f"[INFO] Loaded checkpoint: {args.ckpt}")
    print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"[INFO] Device: {device}")

    image_path = Path(args.image).resolve()
    image_tensor = load_image(image_path)
    result = predict_image(model, image_tensor, device=device)
    print(f"[RESULT] image={image_path}")
    print(result)


if __name__ == "__main__":
    main()
