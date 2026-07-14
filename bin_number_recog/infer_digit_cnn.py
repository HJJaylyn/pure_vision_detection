#!/usr/bin/env python3
"""Run TinyDigitCNN on cropped digit or black-panel images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cnn_digit_model import load_digit_cnn, predict_digit, predict_digit_sequence, require_torch
from recognize_bin_labels import split_digits


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def collect_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            paths.append(path)
    return paths


def infer_digit_image(path: Path, bundle) -> tuple[str, float]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return predict_digit(image, bundle)


def infer_panel_image(path: Path, bundle, max_digits: int) -> tuple[str, float, int]:
    panel = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if panel is None:
        raise RuntimeError(f"cannot read image: {path}")
    digit_images, boxes, _ = split_digits(panel, max_digits=max_digits)
    value, confidence, _ = predict_digit_sequence(digit_images, bundle)
    return value, confidence, len(boxes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer bin-number digits with a trained TinyDigitCNN.")
    parser.add_argument("inputs", nargs="+", help="Image files or folders.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("digit", "panel"), default="panel")
    parser.add_argument("--max-digits", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    require_torch()
    bundle = load_digit_cnn(args.checkpoint, device=args.device)
    for path in collect_inputs(args.inputs):
        if args.mode == "digit":
            value, confidence = infer_digit_image(path, bundle)
            print(f"{path}\tvalue={value}\tconf={confidence:.3f}")
        else:
            value, confidence, digit_count = infer_panel_image(path, bundle, args.max_digits)
            print(f"{path}\tvalue={value or 'N/A'}\tconf={confidence:.3f}\tdigits={digit_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
