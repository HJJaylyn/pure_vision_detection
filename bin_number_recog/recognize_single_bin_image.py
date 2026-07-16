#!/usr/bin/env python3
"""Recognize one bin-number label image with the finalized PaddleOCR pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from evaluate_paddle_text_recognition import best_panel, prepare_paddle_panel, recognize_panel


DEFAULT_IMAGE = Path("/workspace/huangjie/Franka/data/img/right_tcp_20260715_212141_294.jpg")


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize one bin-number label image.")
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--digit-count", type=int, default=None, help="Optional known digit count, e.g. 4.")
    parser.add_argument("--min-blur", type=float, default=20.0)
    parser.add_argument("--min-label-quality", type=float, default=0.55)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        result = {"ok": False, "code": "ERR_IMAGE_READ", "file": str(args.image)}
        print(json.dumps(result, ensure_ascii=False))
        return 2

    expected_hint = "0" * args.digit_count if args.digit_count else None
    try:
        payload, code, quality_code = best_panel(
            args.image,
            image,
            expected_hint,
            args.min_blur,
            args.min_label_quality,
        )
    except Exception as exc:
        result = {
            "ok": False,
            "code": "ERR_PIPELINE",
            "file": str(args.image),
            "error": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False))
        return 3

    if payload is None:
        result = {
            "ok": False,
            "code": code,
            "file": str(args.image),
            "quality_code": quality_code or "",
        }
        print(json.dumps(result, ensure_ascii=False))
        return 1

    panel = prepare_paddle_panel(payload["panel"], digit_count=args.digit_count)
    try:
        from paddleocr import TextRecognition

        recognizer = TextRecognition()
        pred, score, raw_text = recognize_panel(recognizer, panel)
    except Exception as exc:
        result = {
            "ok": False,
            "code": "ERR_OCR_ENGINE",
            "file": str(args.image),
            "rotation": payload["rotation"],
            "error": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False))
        return 4

    if not pred:
        result_code = "ERR_OCR_NO_DIGITS"
        ok = False
    else:
        result_code = "OK"
        ok = True

    result = {
        "ok": ok,
        "code": result_code,
        "file": str(args.image),
        "bin_number": pred,
        "raw_text": raw_text,
        "confidence": round(float(score), 6),
        "rotation": payload["rotation"],
        "quality_code": quality_code or "",
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
