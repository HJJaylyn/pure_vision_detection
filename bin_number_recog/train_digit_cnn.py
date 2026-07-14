#!/usr/bin/env python3
"""Train a small CNN for cropped bin-number digit images."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cnn_digit_model import DIGIT_LABELS, INPUT_SIZE, TinyDigitCNN, normalize_digit_image, require_torch

require_torch()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class DigitSample:
    path: Path
    label: int


def collect_samples_from_folders(root: Path) -> list[DigitSample]:
    samples: list[DigitSample] = []
    for digit, label in zip(DIGIT_LABELS, range(len(DIGIT_LABELS))):
        folder = root / digit
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(DigitSample(path=path, label=label))
    return samples


def collect_samples_from_csv(csv_path: Path, base_dir: Path | None = None) -> list[DigitSample]:
    samples: list[DigitSample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_value = row.get("image") or row.get("path") or row.get("file")
            label_value = row.get("label") or row.get("digit")
            if not image_value or label_value not in DIGIT_LABELS:
                continue
            path = Path(image_value)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            samples.append(DigitSample(path=path, label=DIGIT_LABELS.index(label_value)))
    return samples


def augment_digit(arr: np.ndarray) -> np.ndarray:
    """Light augmentation for real camera variation."""
    image = (arr[0] * 255).astype(np.uint8)
    h, w = image.shape

    angle = random.uniform(-7.0, 7.0)
    scale = random.uniform(0.92, 1.08)
    tx = random.uniform(-2.0, 2.0)
    ty = random.uniform(-2.0, 2.0)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += (tx, ty)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    alpha = random.uniform(0.75, 1.25)
    beta = random.uniform(-14, 14)
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.25:
        k = random.choice((3, 5))
        image = cv2.GaussianBlur(image, (k, k), 0)
    if random.random() < 0.25:
        noise = np.random.normal(0, 6, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image.astype(np.float32)[None, :, :] / 255.0


class DigitDataset(Dataset):
    def __init__(self, samples: list[DigitSample], augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"cannot read image: {sample.path}")
        arr = normalize_digit_image(image, input_size=INPUT_SIZE)
        if self.augment:
            arr = augment_digit(arr)
        return torch.from_numpy(arr), torch.tensor(sample.label, dtype=torch.long)


def evaluate(model, loader, device: str) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += images.size(0)
    return total_loss / max(1, count), correct / max(1, count)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TinyDigitCNN on cropped single-digit images.")
    parser.add_argument("--data-root", type=Path, help="Dataset folder with 0/..9/ subfolders.")
    parser.add_argument("--csv", type=Path, help="CSV with image/path/file and label/digit columns.")
    parser.add_argument("--csv-base-dir", type=Path, help="Base dir for relative CSV image paths.")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint path, e.g. models/digit_cnn.pt.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentation.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples: list[DigitSample] = []
    if args.data_root:
        samples.extend(collect_samples_from_folders(args.data_root))
    if args.csv:
        samples.extend(collect_samples_from_csv(args.csv, args.csv_base_dir))
    if not samples:
        raise SystemExit("No training samples found. Use --data-root or --csv.")

    random.shuffle(samples)
    dataset = DigitDataset(samples, augment=not args.no_augment)
    val_count = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) >= 10 else 0
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(args.seed)
    if val_count:
        train_dataset, val_dataset = random_split(dataset, [train_count, val_count], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )

    model = TinyDigitCNN(num_classes=len(DIGIT_LABELS)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None

    print(f"samples={len(samples)} train={train_count} val={val_count} device={args.device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += images.size(0)

        train_loss = total_loss / max(1, count)
        train_acc = correct / max(1, count)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, args.device)
        else:
            val_loss, val_acc = train_loss, train_acc

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state if best_state is not None else model.state_dict(),
            "labels": DIGIT_LABELS,
            "input_size": INPUT_SIZE,
            "sample_count": len(samples),
            "best_val_acc": best_acc,
        },
        args.output,
    )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
