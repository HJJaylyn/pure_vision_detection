#!/usr/bin/env python3
"""Small CNN utilities for white-on-black digit recognition."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - handled by caller for deployment.
    torch = None
    nn = None
    F = None


INPUT_SIZE = (32, 48)  # width, height
DIGIT_LABELS = tuple("0123456789")


if nn is not None:

    class TinyDigitCNN(nn.Module):
        """A small CPU-friendly digit classifier for cropped bin-label digits."""

        def __init__(self, num_classes: int = 10) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.15),
                nn.Linear(96, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

else:
    TinyDigitCNN = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for CNN digit recognition. Install torch in this environment.")


def normalize_digit_image(image: np.ndarray, input_size: tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    """Convert a digit crop into a normalized 1xHxW float tensor array."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Accept either white-on-black crops or binary masks. Keep the convention:
    # bright foreground digit on dark background.
    if gray.mean() > 127:
        # If the crop is mostly bright, it may be an inverted OCR image.
        gray = 255 - gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() > 127:
        binary = 255 - binary

    ys, xs = np.where(binary > 0)
    target_w, target_h = input_size
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    if len(xs) == 0:
        return canvas.astype(np.float32)[None, :, :] / 255.0

    crop = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = min((target_w - 6) / crop.shape[1], (target_h - 6) / crop.shape[0])
    new_w = max(1, int(crop.shape[1] * scale))
    new_h = max(1, int(crop.shape[0] * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas.astype(np.float32)[None, :, :] / 255.0


def load_digit_cnn(checkpoint_path: str | Path, device: str = "cpu"):
    """Load a trained TinyDigitCNN checkpoint."""
    require_torch()
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    labels = tuple(checkpoint.get("labels", DIGIT_LABELS))
    input_size = tuple(checkpoint.get("input_size", INPUT_SIZE))
    model = TinyDigitCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "labels": labels,
        "input_size": input_size,
        "device": device,
        "checkpoint": str(checkpoint_path),
    }


def predict_digit(image: np.ndarray, bundle) -> tuple[str, float]:
    """Predict one digit crop."""
    require_torch()
    arr = normalize_digit_image(image, input_size=bundle["input_size"])
    tensor = torch.from_numpy(arr).unsqueeze(0).to(bundle["device"])
    with torch.no_grad():
        logits = bundle["model"](tensor)
        probs = F.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return bundle["labels"][int(idx)], float(conf.item())


def predict_digit_sequence(digit_images: list[np.ndarray], bundle) -> tuple[str, float, list[float]]:
    """Predict a left-to-right sequence of segmented digit crops."""
    if not digit_images:
        return "", 0.0, []
    chars: list[str] = []
    confidences: list[float] = []
    for digit_image in digit_images:
        digit, confidence = predict_digit(digit_image, bundle)
        chars.append(digit)
        confidences.append(confidence)
    return "".join(chars), min(confidences), confidences
