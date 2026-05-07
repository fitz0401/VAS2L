from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

if __package__ in (None, ""):
	sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18
from sklearn.metrics import average_precision_score
from success_module.data_augmentation import AugmentedSubset
from success_module.success_dataloader import SuccessFrameDataset


class SuccessClassifier(nn.Module):
	def __init__(self, num_classes: int = 2, pretrained: bool = True):
		super().__init__()
		weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
		self.backbone = resnet18(weights=weights)
		in_features = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(in_features, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.backbone(x)


def build_model(pretrained: bool = True, num_classes: int = 2, device: torch.device | None = None) -> SuccessClassifier:
	model = SuccessClassifier(num_classes=num_classes, pretrained=pretrained)
	if device is not None:
		model = model.to(device)
	return model


def load_checkpoint(model: nn.Module, ckpt_path: str | Path, map_location: str | torch.device = "cpu") -> Dict:
	checkpoint = torch.load(Path(ckpt_path), map_location=map_location)
	state_dict = checkpoint.get("model_state_dict", checkpoint)
	model.load_state_dict(state_dict)
	return checkpoint


def preprocess_bgr_image(image_bgr: np.ndarray, resize_hw: Tuple[int, int] = (224, 224), normalize: bool = True, rgb: bool = True) -> torch.Tensor:
	image = cv2.resize(image_bgr, (int(resize_hw[1]), int(resize_hw[0])), interpolation=cv2.INTER_AREA)
	if rgb:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.float32)
	if normalize:
		image /= 255.0
	image = np.transpose(image, (2, 0, 1))
	return torch.from_numpy(image).float()


@torch.no_grad()
def predict_image(model: nn.Module, image_tensor: torch.Tensor, device: torch.device | None = None) -> Dict[str, float | int]:
	model.eval()
	if image_tensor.ndim == 3:
		image_tensor = image_tensor.unsqueeze(0)
	if device is not None:
		image_tensor = image_tensor.to(device)
	logits = model(image_tensor)
	probs = torch.softmax(logits, dim=1)[0]
	pred = int(torch.argmax(probs).item())
	return {
		"pred": pred,
		"prob_0": float(probs[0].item()),
		"prob_1": float(probs[1].item()),
	}


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def split_indices_by_video(dataset: SuccessFrameDataset, test_size: float, seed: int) -> Tuple[List[int], List[int]]:
	unique_videos = sorted({str(s.video_path) for s in dataset.samples})
	if not unique_videos:
		raise ValueError("Need at least 1 video to create train/test split.")
	if test_size <= 0:
		return list(range(len(dataset.samples))), []
	if len(unique_videos) < 2:
		raise ValueError("Need at least 2 videos to create a non-empty test split.")

	rng = random.Random(seed)
	rng.shuffle(unique_videos)

	n_test = max(1, int(round(len(unique_videos) * test_size)))
	n_test = min(n_test, len(unique_videos) - 1)
	test_videos = set(unique_videos[:n_test])

	train_indices: List[int] = []
	test_indices: List[int] = []
	for i, s in enumerate(dataset.samples):
		if str(s.video_path) in test_videos:
			test_indices.append(i)
		else:
			train_indices.append(i)

	if not train_indices or not test_indices:
		raise RuntimeError("Invalid split produced empty train or test indices.")
	return train_indices, test_indices


def make_loader(dataset: SuccessFrameDataset, indices: Sequence[int], batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
	subset = Subset(dataset, list(indices))
	return DataLoader(
		subset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer | None, device: torch.device) -> Dict[str, float]:
	train_mode = optimizer is not None
	model.train(mode=train_mode)

	total_loss = 0.0
	total = 0
	correct = 0
	all_labels: List[int] = []
	all_probs: List[float] = []

	for images, labels, _ in loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		logits = model(images)
		loss = criterion(logits, labels)

		if train_mode:
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()

		total_loss += float(loss.item()) * images.size(0)
		preds = torch.argmax(logits, dim=1)
		correct += int((preds == labels).sum().item())
		total += images.size(0)
		probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
		all_probs.extend(float(p) for p in probs)
		all_labels.extend(labels.detach().cpu().tolist())

	avg_loss = total_loss / max(total, 1)
	acc = correct / max(total, 1)
	if len(set(all_labels)) < 2:
		ap = float(all_labels[0]) if all_labels else 0.0
	else:
		ap = float(average_precision_score(all_labels, all_probs))
	return {"loss": avg_loss, "acc": acc, "ap": ap}


def load_config(config_path: Path) -> Dict:
	with config_path.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}
	if "train" not in cfg:
		raise ValueError(f"Missing 'train' section in {config_path}")
	return cfg


def main() -> None:
	parser = argparse.ArgumentParser(description="Train success classifier (resnet18 + linear head)")
	parser.add_argument("--config", type=str, default="success_module/config.yaml")
	parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
	args = parser.parse_args()

	config_path = Path(args.config).resolve()
	cfg = load_config(config_path)
	train_cfg = cfg["train"]

	set_seed(int(train_cfg.get("random_seed", 42)))

	dataset = SuccessFrameDataset(config_path)
	train_indices, test_indices = split_indices_by_video(
		dataset=dataset,
		test_size=float(train_cfg.get("test_size", 0.2)),
		seed=int(train_cfg.get("random_seed", 42)),
	)

	batch_size = int(train_cfg.get("batch_size", 32))
	num_workers = int(train_cfg.get("num_workers", 4))
	enable_augmentation = bool(train_cfg.get("enable_augmentation", False))
	if enable_augmentation:
		print("[INFO] Data augmentation enabled for training set.")
		train_loader = DataLoader(
			AugmentedSubset(dataset, train_indices, augment=True),
			batch_size=batch_size, shuffle=True,
			num_workers=num_workers, pin_memory=True,
		)
		test_loader = DataLoader(
			AugmentedSubset(dataset, test_indices, augment=False),
			batch_size=batch_size, shuffle=False,
			num_workers=num_workers, pin_memory=True,
		) if test_indices else None
	else:
		train_loader = make_loader(dataset, train_indices, batch_size, num_workers, shuffle=True)
		test_loader = make_loader(dataset, test_indices, batch_size, num_workers, shuffle=False) if test_indices else None
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SuccessClassifier(num_classes=2, pretrained=(not args.no_pretrained)).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(
		model.parameters(),
		lr=float(train_cfg.get("learning_rate", 1.0e-4)),
		weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
	)

	epochs = int(train_cfg.get("epochs", 10))
	patience = int(train_cfg.get("early_stop_patience", 3))
	best_test_ap = -1.0
	no_improve_epochs = 0

	save_dir = (config_path.parent.parent / str(train_cfg.get("save_dir", "ckpts/success_model"))).resolve()
	save_dir.mkdir(parents=True, exist_ok=True)
	best_ckpt = save_dir / "best.pt"
	last_ckpt = save_dir / "last.pt"
	metrics_path = save_dir / "metrics.json"

	history: List[Dict] = []
	train_video_indices = sorted({int(Path(dataset.samples[i].video_path).stem.split("_")[-1]) for i in train_indices})
	test_video_indices = sorted({int(Path(dataset.samples[i].video_path).stem.split("_")[-1]) for i in test_indices})

	print(f"[INFO] Dataset frames: {len(dataset)}")
	print(f"[INFO] Positive samples: {dataset.positive_count}")
	print(f"[INFO] Negative samples: {dataset.negative_count}")
	print(f"[INFO] Positive ratio: {dataset.positive_ratio:.4f}")
	print(f"[INFO] Negative ratio: {dataset.negative_ratio:.4f}")
	print(f"[INFO] Train frames: {len(train_indices)} | Test frames: {len(test_indices)}")
	print(f"[INFO] Train videos: {len(train_video_indices)} | indices={train_video_indices}")
	print(f"[INFO] Test videos: {len(test_video_indices)} | indices={test_video_indices}")
	print(f"[INFO] Device: {device}")

	for epoch in range(1, epochs + 1):
		train_stats = run_epoch(model, train_loader, criterion, optimizer, device)
		test_stats = run_epoch(model, test_loader, criterion, optimizer=None, device=device) if test_loader is not None else None

		record = {
			"epoch": epoch,
			"train_loss": train_stats["loss"],
			"train_acc": train_stats["acc"],
			"train_ap": train_stats["ap"],
			"test_loss": None if test_stats is None else test_stats["loss"],
			"test_acc": None if test_stats is None else test_stats["acc"],
			"test_ap": None if test_stats is None else test_stats["ap"],
		}
		history.append(record)

		if test_stats is None:
			print(f"[E{epoch:03d}] train_loss={record['train_loss']:.4f} train_acc={record['train_acc']:.4f} train_ap={record['train_ap']:.4f}")
		else:
			print(
				f"[E{epoch:03d}] train_loss={record['train_loss']:.4f} train_acc={record['train_acc']:.4f} train_ap={record['train_ap']:.4f} "
				f"test_loss={record['test_loss']:.4f} test_acc={record['test_acc']:.4f} test_ap={record['test_ap']:.4f}"
			)

		if test_stats is not None and record["test_ap"] > best_test_ap:
			best_test_ap = record["test_ap"]
			no_improve_epochs = 0
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"config": cfg,
					"epoch": epoch,
					"test_ap": best_test_ap,
				},
				best_ckpt,
			)
		elif test_stats is not None:
			no_improve_epochs += 1

		torch.save(
			{
				"model_state_dict": model.state_dict(),
				"config": cfg,
				"epoch": epoch,
				"test_acc": record["test_acc"],
			},
			last_ckpt,
		)

		if test_stats is not None and no_improve_epochs >= patience:
			print(f"[INFO] Early stopping triggered after {no_improve_epochs} epochs without test AP improvement.")
			break

	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(
			{
				"best_test_ap": best_test_ap,
				"early_stop_patience": patience,
				"epochs_trained": len(history),
				"stopped_early": test_stats is not None and no_improve_epochs >= patience,
				"has_test_split": test_stats is not None,
				"history": history,
				"train_size": len(train_indices),
				"test_size": len(test_indices),
				"positive_count": dataset.positive_count,
				"negative_count": dataset.negative_count,
				"positive_ratio": dataset.positive_ratio,
				"negative_ratio": dataset.negative_ratio,
				"train_videos": sorted({str(dataset.samples[i].video_path) for i in train_indices}),
				"test_videos": sorted({str(dataset.samples[i].video_path) for i in test_indices}),
			},
			f,
			indent=2,
			ensure_ascii=False,
		)

	print(f"[INFO] Best checkpoint: {best_ckpt}")
	print(f"[INFO] Last checkpoint: {last_ckpt}")
	print(f"[INFO] Metrics: {metrics_path}")
	print(f"[INFO] Best test AP: {best_test_ap:.4f}")
	print(f"[INFO] Early stop patience: {patience}")
	print(f"[INFO] Trained epochs: {len(history)}")
	print(f"[INFO] Stopped early: {no_improve_epochs >= patience}")


if __name__ == "__main__":
	main()
