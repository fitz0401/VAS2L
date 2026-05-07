from __future__ import annotations

"""
success_augmentation.py
-----------------------
Augmentation pipelines for robot wrist-camera success classification.

Design constraints:
  - NO horizontal flip  (changes left/right semantic meaning for arm)
  - NO heavy geometric warp (wrist camera has fixed perspective)
  - YES colour/lighting jitter (real-world lighting variation)
  - YES mild crop/scale (slight camera-mount vibration)
  - YES occlusion simulation (fingers, objects partially in frame)
  - YES blur/noise (focus variation, sensor noise)
  - Augmentation applied to TRAIN only; test uses clean normalize-only transform

Usage
-----
In SuccessFrameDataset.__init__, pass augment=True for train subset:

    full_dataset = SuccessFrameDataset(config_path, augment=False)
    train_dataset = SuccessFrameDataset(config_path, augment=True)

Or patch __getitem__ to accept a transform (see bottom of file for monkey-patch helper).
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torchvision.tv_tensors import Image as TVImage
import numpy as np


# ---------------------------------------------------------------------------
# ImageNet normalisation constants (same as ResNet18 pretrain)
# ---------------------------------------------------------------------------
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Clean transform (test / inference — never augment)
# ---------------------------------------------------------------------------
def build_test_transform(resize_hw: tuple[int, int] = (224, 224)) -> T.Compose:
    """
    Deterministic pipeline: resize → normalise.
    Input:  float32 tensor (C, H, W) in [0, 1]   (as returned by _load_frame)
    Output: float32 tensor (C, H, W), ImageNet-normalised
    """
    return T.Compose([
        T.Resize(resize_hw, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


# ---------------------------------------------------------------------------
# Train augmentation transform
# ---------------------------------------------------------------------------
def build_train_transform(
    resize_hw: tuple[int, int] = (224, 224),
    # ── colour ──────────────────────────────────────────────────────────────
    color_jitter_prob: float = 0.8,
    brightness: float = 0.3,
    contrast:   float = 0.3,
    saturation: float = 0.2,
    hue:        float = 0.05,
    grayscale_prob: float = 0.05,   # rare; preserves colour cues most of the time
    # ── geometry (mild only) ────────────────────────────────────────────────
    random_crop_scale: tuple[float, float] = (0.80, 1.00),  # max 20 % crop
    random_crop_ratio: tuple[float, float] = (0.90, 1.10),  # slight aspect jitter
    # ── occlusion ───────────────────────────────────────────────────────────
    erasing_prob: float = 0.30,
    erasing_scale: tuple[float, float] = (0.02, 0.08),  # small patches only
    erasing_ratio: tuple[float, float] = (0.3, 3.0),
    erasing_value: float = 0.0,     # black patch (or use 'random')
    # ── blur / noise ────────────────────────────────────────────────────────
    blur_prob: float = 0.20,
    blur_kernel: int = 5,
    blur_sigma: tuple[float, float] = (0.1, 1.2),
    noise_std: float = 0.015,       # additive Gaussian noise std (in [0,1] space)
    noise_prob: float = 0.30,
) -> T.Compose:
    """
    Stochastic pipeline for training frames.
    Input:  float32 tensor (C, H, W) in [0, 1]
    Output: float32 tensor (C, H, W), ImageNet-normalised

    Key omissions (intentional):
      • No horizontal flip  → arm chirality is meaningful
      • No rotation > 0°    → wrist cam is rigidly mounted
      • No vertical flip    → gravity direction is a cue
    """
    transforms = [
        # 1. Mild random crop (simulates slight camera shake / crop variation)
        T.RandomResizedCrop(
            size=resize_hw,
            scale=random_crop_scale,
            ratio=random_crop_ratio,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),

        # 2. Colour jitter (lighting, white-balance variation)
        T.RandomApply([
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        ], p=color_jitter_prob),

        # 3. Occasional grayscale (forces texture-based reasoning)
        T.RandomGrayscale(p=grayscale_prob),

        # 4. Gaussian blur (focus variation, motion blur approximation)
        T.RandomApply([
            T.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
        ], p=blur_prob),

        # 5. ImageNet normalisation  ← must come before RandomErasing
        T.Normalize(mean=_MEAN, std=_STD),

        # 6. Random erasing (simulates partial occlusion by fingers / objects)
        #    Applied AFTER normalise so erasing_value=0 → near-black patch
        T.RandomErasing(
            p=erasing_prob,
            scale=erasing_scale,
            ratio=erasing_ratio,
            value=erasing_value,
        ),
    ]

    if noise_prob > 0 and noise_std > 0:
        transforms.append(_GaussianNoise(std=noise_std, p=noise_prob))

    return T.Compose(transforms)


# ---------------------------------------------------------------------------
# Custom transform: additive Gaussian noise (not in torchvision.transforms.v2)
# ---------------------------------------------------------------------------
class _GaussianNoise(torch.nn.Module):
    """
    Add N(0, std) noise to a float tensor.
    Applied in normalised space, so keep std small (0.01–0.03 is typical).
    """
    def __init__(self, std: float = 0.015, p: float = 0.3):
        super().__init__()
        self.std = std
        self.p   = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            x = x + torch.randn_like(x) * self.std
        return x


# ---------------------------------------------------------------------------
# Multi-frame variant (for temporal stacking)
# ---------------------------------------------------------------------------
def build_train_transform_multiframe(
    num_frames: int = 3,
    resize_hw: tuple[int, int] = (224, 224),
    **kwargs,
) -> "_ConsistentMultiFrameTransform":
    """
    Wraps build_train_transform so the *same* spatial augmentation
    (crop, erase) is applied consistently across all frames in a stack,
    while colour jitter and noise are applied independently per frame.

    Input:  float32 tensor (num_frames, C, H, W) in [0, 1]
    Output: float32 tensor (num_frames, C, H, W), normalised
    """
    return _ConsistentMultiFrameTransform(
        num_frames=num_frames,
        resize_hw=resize_hw,
        **kwargs,
    )


class _ConsistentMultiFrameTransform(torch.nn.Module):
    """
    Spatial ops (crop, erase) use a shared random state across frames.
    Per-pixel ops (colour, noise) are independent.
    """
    def __init__(self, num_frames: int, resize_hw: tuple[int, int], **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.resize_hw  = resize_hw
        self.crop_scale = kwargs.get("random_crop_scale", (0.80, 1.00))
        self.crop_ratio = kwargs.get("random_crop_ratio", (0.90, 1.10))
        self.erasing_p  = kwargs.get("erasing_prob", 0.30)
        self.erasing_scale = kwargs.get("erasing_scale", (0.02, 0.08))
        self.erasing_ratio = kwargs.get("erasing_ratio", (0.3, 3.0))

        # Per-frame colour/noise pipeline (no spatial ops)
        self._per_frame = T.Compose([
            T.RandomApply([
                T.ColorJitter(
                    brightness=kwargs.get("brightness", 0.3),
                    contrast=kwargs.get("contrast", 0.3),
                    saturation=kwargs.get("saturation", 0.2),
                    hue=kwargs.get("hue", 0.05),
                )
            ], p=kwargs.get("color_jitter_prob", 0.8)),
            T.RandomGrayscale(p=kwargs.get("grayscale_prob", 0.05)),
            T.RandomApply([
                T.GaussianBlur(
                    kernel_size=kwargs.get("blur_kernel", 5),
                    sigma=kwargs.get("blur_sigma", (0.1, 1.2)),
                )
            ], p=kwargs.get("blur_prob", 0.20)),
            T.Normalize(mean=_MEAN, std=_STD),
        ])
        self._noise = _GaussianNoise(
            std=kwargs.get("noise_std", 0.015),
            p=kwargs.get("noise_prob", 0.30),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T, C, H, W) float32 in [0,1]"""
        T_frames, C, H, W = frames.shape

        # ── Shared crop parameters ──────────────────────────────────────────
        i, j, h, w = T.RandomResizedCrop.get_params(
            frames[0],
            scale=self.crop_scale,
            ratio=self.crop_ratio,
        )

        # ── Shared erase parameters (decided once) ──────────────────────────
        do_erase = torch.rand(1).item() < self.erasing_p
        if do_erase:
            ei, ej, eh, ew, ev = T.RandomErasing.get_params(
                frames[0],
                scale=self.erasing_scale,
                ratio=self.erasing_ratio,
                value=[0.0],
            )

        out = []
        for t in range(T_frames):
            f = frames[t]
            # Shared spatial crop
            f = TF.resized_crop(
                f, i, j, h, w,
                size=list(self.resize_hw),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )
            # Independent colour/blur/noise
            f = self._per_frame(f)
            f = self._noise(f)
            # Shared erase
            if do_erase:
                f = TF.erase(f, ei, ej, eh, ew, ev)
            out.append(f)

        return torch.stack(out, dim=0)  # (T, C, H, W)


# ---------------------------------------------------------------------------
# Integration helper: patch SuccessFrameDataset without subclassing
# ---------------------------------------------------------------------------
def patch_dataset_with_augmentation(
    dataset,
    augment: bool = True,
    num_frames: int = 1,
    resize_hw: tuple[int, int] = (224, 224),
):
    """
    Monkey-patches dataset.__getitem__ to apply augmentation.

    Usage:
        train_set = Subset(dataset, train_indices)
        patch_dataset_with_augmentation(train_set.dataset, augment=True)
        # test set — call again with augment=False to use clean transform
        test_set  = Subset(dataset, test_indices)
        patch_dataset_with_augmentation(test_set.dataset, augment=False)

    NOTE: If train and test share the *same* dataset object (as in the current
    codebase), you need to subclass or use a wrapper instead:

        train_loader = DataLoader(
            AugmentedSubset(dataset, train_indices, augment=True), ...)
        test_loader  = DataLoader(
            AugmentedSubset(dataset, test_indices,  augment=False), ...)
    """
    if num_frames > 1:
        transform = (
            build_train_transform_multiframe(num_frames=num_frames, resize_hw=resize_hw)
            if augment
            else None   # handled inside AugmentedSubset
        )
    else:
        transform = (
            build_train_transform(resize_hw=resize_hw)
            if augment
            else build_test_transform(resize_hw=resize_hw)
        )
    dataset._augment_transform = transform


# ---------------------------------------------------------------------------
# Recommended: AugmentedSubset wrapper (clean train/test separation)
# ---------------------------------------------------------------------------
from torch.utils.data import Dataset as _Dataset, Subset


class AugmentedSubset(_Dataset):
    """
    Wraps a Subset and applies different transforms to train vs test
    *without* modifying the underlying dataset object.

    Example (drop-in replacement for make_loader in train_model.py):

        train_loader = DataLoader(
            AugmentedSubset(dataset, train_indices, augment=True),
            batch_size=32, shuffle=True, num_workers=4)

        test_loader = DataLoader(
            AugmentedSubset(dataset, test_indices, augment=False),
            batch_size=32, shuffle=False, num_workers=4)
    """

    def __init__(
        self,
        dataset,
        indices: list[int],
        augment: bool = False,
        num_frames: int = 1,
        resize_hw: tuple[int, int] = (224, 224),
    ):
        self.dataset   = dataset
        self.indices   = indices
        self.num_frames = num_frames

        if num_frames > 1:
            self.transform = (
                build_train_transform_multiframe(num_frames=num_frames, resize_hw=resize_hw)
                if augment
                else None  # multi-frame test: only normalise each frame
            )
            self.frame_norm = build_test_transform(resize_hw=resize_hw)
        else:
            self.transform = (
                build_train_transform(resize_hw=resize_hw)
                if augment
                else build_test_transform(resize_hw=resize_hw)
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img, label, meta = self.dataset[real_idx]
        # img: (C,H,W) for single-frame, (T,C,H,W) for multi-frame

        if self.num_frames > 1:
            if self.transform is not None:
                img = self.transform(img)       # consistent spatial + per-frame colour
            else:
                # test: only normalise
                T_frames = img.shape[0]
                img = torch.stack([
                    self.frame_norm(img[t]) for t in range(T_frames)
                ], dim=0)
        else:
            img = self.transform(img)

        return img, label, meta


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing augmentation pipelines...")

    # Single-frame train
    tf_train = build_train_transform()
    tf_test  = build_test_transform()

    dummy = torch.rand(3, 240, 320)   # CHW, [0,1], simulates _load_frame output
    out_train = tf_train(dummy)
    out_test  = tf_test(dummy)
    print(f"  Single-frame train output: {out_train.shape}  "
          f"min={out_train.min():.3f}  max={out_train.max():.3f}")
    print(f"  Single-frame test  output: {out_test.shape}   "
          f"min={out_test.min():.3f}   max={out_test.max():.3f}")

    # Multi-frame train
    tf_multi = build_train_transform_multiframe(num_frames=3)
    dummy_multi = torch.rand(3, 3, 240, 320)   # (T, C, H, W)
    out_multi = tf_multi(dummy_multi)
    print(f"  Multi-frame train  output: {out_multi.shape}  "
          f"min={out_multi.min():.3f}  max={out_multi.max():.3f}")

    print("All checks passed.")