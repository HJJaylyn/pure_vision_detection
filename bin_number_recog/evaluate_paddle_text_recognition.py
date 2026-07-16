#!/usr/bin/env python3
"""Evaluate PaddleOCR TextRecognition on cropped bin-number panels.

The dataset layout is expected to be:

    dataset_dir/
      67/
        000000.png
      9090/
        000000.png

Each folder name is used as the ground-truth digit string. The script first
uses the conservative CV pipeline to crop the black number panel, then runs
PaddleOCR TextRecognition on that cropped panel.
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

from extract_digit_crops import (
    CANONICAL_PANEL_SIZE,
    CANONICAL_TEXT_RIGHT,
    IMAGE_SUFFIXES,
    WIDE_DIGIT_ROI_RATIOS,
    _black_rect_score,
    balanced_gray,
    black_panel_edge_continuity_score,
    auto_balance_brightness,
    compute_panel_split,
    find_black_digit_panel,
    find_black_rect_edge_quad,
    local_equalized_gray,
    normalize_panel_size,
    order_points,
    panel_selection_score,
    tighten_black_rect_bounds,
    trim_panel_whitespace,
    warp_quad_to_panel,
)
from recognize_bin_labels import (
    assess_digit_panel_quality,
    crop_bin_roi_from_warp,
    preprocess_label_image,
    rotate_warp,
)


def iter_images(dataset_dir: Path):
    for label_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        for path in sorted(p for p in label_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            yield label_dir.name, path


def best_panel(path: Path, image, expected: str | None, min_blur: float, min_label_quality: float):
    preprocessed = preprocess_label_image(
        path,
        image,
        min_blur=min_blur,
        min_label_quality=min_label_quality,
    )
    if preprocessed["error"] is not None:
        return None, preprocessed["error"]["code"], None

    best = None
    best_payload = None
    digit_count = len(expected) if expected else None
    for rotation in (0, 90, 180, 270):
        oriented = rotate_warp(preprocessed["warped"], rotation)
        search_roi, _ = crop_bin_roi_from_warp(oriented, WIDE_DIGIT_ROI_RATIOS)
        panel, box, found = find_black_digit_panel(search_roi, digit_count=digit_count)
        if not found:
            continue

        quality_code = assess_digit_panel_quality(panel)
        score = panel_selection_score(panel, digit_count)
        if quality_code is not None:
            # Prefer a low-quality true panel over a clean wrong subject.
            score -= 0.05

        candidate = (score, rotation)
        if best is None or candidate > best:
            best = candidate
            best_payload = {
                "panel": panel,
                "rotation": rotation,
                "box": box,
                "quality_code": quality_code,
                "score": score,
            }

    if best_payload is None:
        return None, "ERR_BLACK_PANEL_NOT_FOUND", None
    return best_payload, "OK", best_payload["quality_code"]


OCR_DIGIT_CONFUSIONS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "Z": "2",
        "z": "2",
    }
)


def normalize_ocr_digits(text: str) -> str:
    """Keep digits after mapping common OCR letter confusions for digit-only labels."""
    normalized = text.translate(OCR_DIGIT_CONFUSIONS)
    return re.sub(r"\D", "", normalized)


def recognize_panel(recognizer, panel):
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        cv2.imwrite(tmp.name, panel)
        result = recognizer.predict(tmp.name)

    best_text = ""
    best_raw_text = ""
    best_score = 0.0
    for item in result or []:
        text = str(item.get("rec_text", ""))
        score = float(item.get("rec_score", 0.0))
        digits = normalize_ocr_digits(text)
        if digits and score >= best_score:
            best_text = digits
            best_raw_text = text
            best_score = score
    return best_text, best_score, best_raw_text


def trim_horizontal_white_margins(panel):
    """Remove white label margins to the left/right of the black panel.

    This intentionally does not trim top/bottom and does not crop to digit
    strokes. It only finds columns that belong to the black background, so the
    black panel's own right-side padding is preserved while leaked white label
    border is removed.
    """
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    height, width = gray.shape
    y0 = max(0, int(round(height * 0.08)))
    y1 = min(height, int(round(height * 0.92)))
    core = gray[y0:y1]

    dark = core < 120
    col_dark_ratio = dark.mean(axis=0)
    cols = np.where(col_dark_ratio > 0.18)[0]
    if not cols.size:
        return panel

    x0 = int(cols.min())
    x1 = int(cols.max()) + 1
    pad = max(2, int(round((x1 - x0) * 0.025)))
    x0 = max(0, x0 - pad)
    x1 = min(width, x1 + pad)

    # Be conservative: if the computed crop would remove most of the panel, it
    # is probably reacting to text/glare rather than the physical black block.
    if x1 - x0 < width * 0.45:
        return panel
    return panel[:, x0:x1]


def trim_vertical_white_margins(panel):
    """Remove white margins above/below the black panel without changing x."""
    gray = balanced_gray(panel)
    height, width = gray.shape
    if height < 12 or width < 30:
        return panel

    # Prefer the same quadrilateral edge detector used by the ROI black-panel
    # stage. In overexposed crops the leaked white strip can still be darker
    # than the fixed threshold, making a row projection cover the whole image.
    edge_quad = find_black_rect_edge_quad(panel, max_area_ratio=0.95)
    if edge_quad is not None:
        edge_x, edge_y, edge_w, edge_h = cv2.boundingRect(edge_quad.astype("float32"))
        if edge_h >= height * 0.38 and edge_h < height * 0.98:
            pad = max(1, int(round(height * 0.015)))
            y0 = max(0, edge_y - pad)
            y1 = min(height, edge_y + edge_h + pad)
            if y1 - y0 >= height * 0.38:
                return panel[y0:y1, :]

    dark = (gray < 130).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    row_ratio = dark.mean(axis=1)
    active = row_ratio > 0.30
    if not active.any():
        return panel

    # Use the longest contiguous run. Small text or barcode fragments in a
    # leaked white margin must not become the top/bottom boundary.
    padded = np.pad(active.astype("uint8"), (1, 1))
    starts = np.where((padded[1:-1] == 1) & (padded[:-2] == 0))[0]
    ends = np.where((padded[1:-1] == 1) & (padded[2:] == 0))[0]
    if starts.size == 0 or ends.size == 0:
        return panel
    runs = [(int(end - start), int(start), int(end)) for start, end in zip(starts, ends)]
    _, y0, y1 = max(runs, key=lambda item: item[0])
    if y1 - y0 < height * 0.38:
        return panel

    pad = max(1, int(round(height * 0.015)))
    y0 = max(0, y0 - pad)
    y1 = min(height, y1 + pad)
    return panel[y0:y1, :]


def digit_stroke_presence(panel) -> float:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    height, width = gray.shape
    y0 = max(0, int(round(height * 0.12)))
    y1 = min(height, int(round(height * 0.92)))
    core = gray[y0:y1]
    if core.size == 0:
        return 0.0
    threshold = max(125, min(185, float(np.percentile(core, 88))))
    bright = core > threshold
    col_count = bright.sum(axis=0).astype(np.float32)
    active_cols = float((col_count > max(1.2, core.shape[0] * 0.035)).sum())
    bright_ratio = float(bright.mean())
    contrast = float(np.percentile(core, 92) - np.percentile(core, 8)) / 255.0
    return active_cols / max(1.0, width) + bright_ratio * 2.0 + contrast * 0.35


def black_background_score(panel) -> float:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    height, width = gray.shape
    y0 = max(0, int(round(height * 0.08)))
    y1 = min(height, int(round(height * 0.94)))
    core = gray[y0:y1]
    if core.size == 0:
        return 0.0
    dark = core < 120
    col_dark = dark.mean(axis=0)
    row_dark = dark.mean(axis=1)
    return float(dark.mean()) + 0.35 * float((col_dark > 0.35).mean()) + 0.25 * float((row_dark > 0.35).mean())


def black_rectangle_metrics(panel):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    height, width = gray.shape
    if height <= 0 or width <= 0:
        return {
            "ratio": 0.0,
            "dark_ratio": 0.0,
            "continuous_cols": 0.0,
            "continuous_rows": 0.0,
        }
    y0 = max(0, int(round(height * 0.08)))
    y1 = min(height, int(round(height * 0.94)))
    core = gray[y0:y1]
    dark = core < 120
    col_dark = dark.mean(axis=0)
    row_dark = dark.mean(axis=1)
    return {
        "ratio": width / float(max(1, height)),
        "dark_ratio": float(dark.mean()),
        "continuous_cols": float((col_dark > 0.45).mean()),
        "continuous_rows": float((row_dark > 0.45).mean()),
    }


def is_strict_black_rectangle(panel):
    metrics = black_rectangle_metrics(panel)
    return bool(
        metrics["ratio"] >= 1.45
        and metrics["dark_ratio"] >= 0.54
        and metrics["continuous_cols"] >= 0.56
        and metrics["continuous_rows"] >= 0.48
    )


def warp_black_rect_from_contour(image, contour):
    rect = cv2.minAreaRect(contour)
    box = order_points(cv2.boxPoints(rect))
    tl, tr, br, bl = box
    out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if out_w < 12 or out_h < 8:
        return None
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(box.astype("float32"), dst)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    return normalize_panel_size(trim_panel_whitespace(warped))


def refine_inner_black_panel(panel):
    """Find the real black number rectangle inside an already cropped panel.

    The first detector sometimes includes the label's black/pink outer border
    or a dark strip beside the true number block. This second pass works in the
    cropped panel coordinate frame and looks again for a horizontal black
    rectangle, then normalizes that rectangle before OCR.
    """
    gray_raw = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    gray = gray_raw
    height, width = gray.shape
    if height < 12 or width < 30:
        return panel

    # The ROI detector already found and rectified the physical black panel.
    # Do not let a second threshold/contour pass replace a strong first result:
    # on samples such as 45 and 0123 that pass can mistake a digit stroke or an
    # outer label edge for the panel and remove the leading digit. Only refine
    # panels whose geometry is visibly weak.
    existing_ratio = width / float(max(1, height))
    existing_black = _black_rect_score(panel)
    existing_edge = black_panel_edge_continuity_score(panel)
    if (
        1.65 <= existing_ratio <= 3.20
        and existing_black >= 0.52
        and existing_edge >= 0.45
    ):
        return panel

    # The second pass must be about the physical black long rectangle, not any
    # arbitrary darker texture. Use a strict raw-dark mask; CLAHE can help
    # debug visualization, but here it can make half-white/half-black regions
    # look deceptively valid.
    dark = (gray_raw < 115).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((13, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.36 or h < height * 0.34:
            continue
        ratio = w / float(max(1, h))
        if not (1.25 <= ratio <= 6.8):
            continue
        roi = gray_raw[y : y + h, x : x + w]
        roi_dark = roi < 115
        dark_ratio = float(roi_dark.mean())
        if dark_ratio < 0.54:
            continue
        row_dark_ratio = roi_dark.mean(axis=1)
        col_dark_ratio = roi_dark.mean(axis=0)
        continuous_rows = float((row_dark_ratio > 0.45).mean())
        continuous_cols = float((col_dark_ratio > 0.45).mean())
        if continuous_rows < 0.48 or continuous_cols < 0.56:
            continue

        area_ratio = (w * h) / float(max(1, width * height))
        center_y = (y + h * 0.5) / float(max(1, height))
        # Prefer the broad horizontal block, not a vertical outer border.
        score = (
            w
            * h
            * min(1.0, ratio / 2.4)
            * min(1.0, dark_ratio + 0.25)
            * (0.6 * continuous_cols + 0.4 * continuous_rows)
            * min(1.0, max(0.25, area_ratio / 0.45))
            * min(1.0, max(0.35, 1.25 - abs(center_y - 0.5)))
        )
        candidates.append((score, x, y, w, h, contour))

    if not candidates:
        return panel

    _, x, y, w, h, selected_contour = max(candidates, key=lambda item: item[0])
    x, y, w, h = tighten_black_rect_bounds(gray_raw, x, y, w, h, threshold=115)

    # If the selected block is already essentially the whole panel, keep the
    # original image. Re-cropping in that case only adds interpolation noise.
    if x <= width * 0.03 and y <= height * 0.10 and x + w >= width * 0.97 and y + h >= height * 0.90:
        return panel

    pad_x = max(1, int(round(w * 0.01)))
    pad_y = 0
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)

    original_score = digit_stroke_presence(panel)
    axis_refined = panel[y0:y1, x0:x1]
    local = panel[y:y + h, x:x + w]
    local_gray = gray_raw[y:y + h, x:x + w]
    edge_box = find_black_rect_edge_quad(local)
    local_contours = []
    if edge_box is not None:
        refined = warp_quad_to_panel(local, edge_box)
    else:
        local_dark = (local_gray < 115).astype("uint8")
        local_dark = cv2.morphologyEx(local_dark, cv2.MORPH_CLOSE, np.ones((13, 5), np.uint8))
        local_contours, _ = cv2.findContours(local_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined = None
    if refined is None and local_contours:
        local_contour = max(local_contours, key=cv2.contourArea)
        refined = warp_black_rect_from_contour(local, local_contour)
    if refined is None:
        refined = warp_black_rect_from_contour(panel, selected_contour)
    if refined is None:
        refined = axis_refined
    # One more strict whitespace trim catches a residual white slit between
    # the outer border and the true black digit panel.
    refined = normalize_panel_size(trim_panel_whitespace(refined))
    refined_score = digit_stroke_presence(refined)
    original_black = black_background_score(panel)
    refined_black = black_background_score(refined)
    if not is_strict_black_rectangle(refined):
        return panel
    if refined_score < original_score * 0.72:
        return panel
    if refined_black < 0.45 or refined_black < original_black * 0.72:
        return panel
    return refined


def _slot_text_right(width: int, digit_count: int | None) -> int | None:
    if digit_count is None or digit_count not in CANONICAL_TEXT_RIGHT:
        return None
    scale = width / 200.0
    right = CANONICAL_TEXT_RIGHT[digit_count]
    if digit_count == 1:
        right = min(right, 44)
    return min(width, max(1, int(round(right * scale))))


def black_background_bounds(panel):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    height, width = gray.shape
    y0 = max(0, int(round(height * 0.10)))
    y1 = min(height, int(round(height * 0.94)))
    core = gray[y0:y1]
    if core.size == 0:
        return 0, width
    dark = core < 120
    col_dark_ratio = dark.mean(axis=0)
    cols = np.where(col_dark_ratio > 0.45)[0]
    if not cols.size:
        cols = np.where(col_dark_ratio > 0.34)[0]
    if not cols.size:
        return 0, width
    breaks = np.where(np.diff(cols) > 1)[0]
    starts = np.r_[cols[0], cols[breaks + 1]]
    ends = np.r_[cols[breaks], cols[-1]]
    runs = [(int(end - start + 1), int(start), int(end)) for start, end in zip(starts, ends)]
    # Use the longest continuous black-background run. This avoids treating a
    # leaked pink/white border or a dark text stroke before the real block as
    # the black panel's left edge.
    _, x0, x1_inclusive = max(runs, key=lambda item: item[0])
    x1 = x1_inclusive + 1
    if x1 - x0 < width * 0.30:
        return 0, width
    return x0, x1


def crop_digit_text_region_horizontally(panel, digit_count: int | None = None):
    """Crop the whole digit string horizontally using white-stroke projection.

    This is not per-digit segmentation. It only removes excessive black padding
    around the full digit string while keeping a small black margin for OCR.
    If digit_count is known, fixed slot priors constrain the right boundary.
    """
    if digit_count is None or digit_count < 1 or digit_count > 4:
        return panel

    # The slot priors only make sense after the physical black rectangle has
    # already been cleaned and normalized. Do the slot/projection computation in
    # that canonical coordinate frame, then crop once. Do not resize after this
    # crop, otherwise single digits get stretched or squashed before OCR.
    if panel.shape[1] != CANONICAL_PANEL_SIZE[0] or panel.shape[0] != CANONICAL_PANEL_SIZE[1]:
        canonical = normalize_panel_size(panel)
    else:
        canonical = panel

    cuts, _, _, runs, _, smooth, core_height = compute_panel_split(canonical, digit_count)

    x0 = 0
    if digit_count > 1 and len(cuts) >= digit_count + 1:
        x1 = int(cuts[-1])
    else:
        x1 = int(round(CANONICAL_TEXT_RIGHT.get(digit_count, CANONICAL_PANEL_SIZE[0]) * canonical.shape[1] / 200.0))

    # Projection may show that the last visible stroke extends a little farther
    # than the learned slot prior, especially with glare or anti-aliased edges.
    # Let it extend the right boundary slightly, but never let projection alone
    # pull the crop all the way into the panel's right padding.
    if runs:
        last_end = int(runs[min(len(runs), digit_count) - 1][1])
        safety = max(4, int(round(canonical.shape[1] * 0.025)))
        x1 = max(x1, last_end + 1 + safety)

    right_limit = int(round(CANONICAL_TEXT_RIGHT.get(digit_count, CANONICAL_PANEL_SIZE[0]) * canonical.shape[1] / 200.0))
    right_slack = max(6, int(round(canonical.shape[1] * 0.05)))
    x1 = min(canonical.shape[1], x1, right_limit + right_slack)

    min_width = max(18, int(round(canonical.shape[1] * 0.16)))
    if x1 - x0 < min_width:
        return canonical
    return canonical[:, x0:x1]


def prepare_paddle_panel(panel, digit_count: int | None = None):
    """Build the final image sent to PaddleOCR recognition.

    This performs conservative cleanup:
    1. Remove white label margins that leaked to the left/right.
    2. Re-detect the true black rectangle inside the cropped panel, which
       removes accidental label borders.
    3. Crop the full digit string horizontally by projection, preserving black
       context and never splitting individual digits.
    """
    panel = trim_horizontal_white_margins(panel)
    panel = trim_vertical_white_margins(panel)
    panel = normalize_panel_size(panel)
    panel = refine_inner_black_panel(panel)
    panel = crop_digit_text_region_horizontally(panel, digit_count=digit_count)
    return auto_balance_brightness(panel)


def prepare_paddle_panel_stages(panel, digit_count: int | None = None):
    """Return intermediate panels for debugging OCR input preparation."""
    first_panel = panel
    after_first_trim = normalize_panel_size(
        trim_vertical_white_margins(trim_horizontal_white_margins(first_panel))
    )
    after_second_refine = refine_inner_black_panel(after_first_trim)
    final_panel = auto_balance_brightness(crop_digit_text_region_horizontally(after_second_refine, digit_count=digit_count))
    return {
        "first_black_panel": first_panel,
        "after_first_trim": after_first_trim,
        "after_second_refine": after_second_refine,
        "final_ocr_input": final_panel,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-evaluate PaddleOCR TextRecognition on bin labels.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--error-log", type=Path, default=None, help="Optional TSV file containing only failed rows.")
    parser.add_argument("--save-panel-dir", type=Path, default=None, help="Optional folder to save cropped panels.")
    parser.add_argument("--min-blur", type=float, default=35.0)
    parser.add_argument("--min-label-quality", type=float, default=0.55)
    args = parser.parse_args()

    from paddleocr import TextRecognition

    recognizer = TextRecognition()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.save_panel_dir is not None:
        args.save_panel_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    correct = 0
    readable = 0

    for expected, path in iter_images(args.dataset_dir):
        total += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append(
                {
                    "expected": expected,
                    "file": str(path),
                    "pred": "",
                    "raw_text": "",
                    "score": 0.0,
                    "ok": False,
                    "code": "ERR_IMAGE_READ",
                    "quality_code": "",
                    "rotation": "",
                    "panel_path": "",
                }
            )
            continue

        payload, code, quality_code = best_panel(path, image, expected, args.min_blur, args.min_label_quality)
        panel_path = ""
        pred = ""
        raw_text = ""
        score = 0.0
        if payload is not None:
            panel = prepare_paddle_panel(payload["panel"], digit_count=len(expected))
            if args.save_panel_dir is not None:
                panel_dir = args.save_panel_dir / expected
                panel_dir.mkdir(parents=True, exist_ok=True)
                panel_path = str(panel_dir / f"{path.stem}_rot{payload['rotation']}.png")
                cv2.imwrite(panel_path, panel)
            pred, score, raw_text = recognize_panel(recognizer, panel)
            if pred:
                readable += 1

        ok = pred == expected
        correct += int(ok)
        rows.append(
            {
                "expected": expected,
                "file": str(path),
                "pred": pred,
                "raw_text": raw_text,
                "score": round(score, 6),
                "ok": ok,
                "code": code,
                "quality_code": quality_code or "",
                "rotation": "" if payload is None else payload["rotation"],
                "panel_path": panel_path,
            }
        )
        raw_suffix = f"\traw={raw_text}" if raw_text and raw_text != pred else ""
        print(f"{expected}\t{path.name}\tpred={pred or '-'}\tscore={score:.4f}\tok={ok}\tcode={code}{raw_suffix}")

    fieldnames = [
        "expected",
        "file",
        "pred",
        "raw_text",
        "score",
        "ok",
        "code",
        "quality_code",
        "rotation",
        "panel_path",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.error_log is not None:
        args.error_log.parent.mkdir(parents=True, exist_ok=True)
        failed_rows = [row for row in rows if not row["ok"]]
        with args.error_log.open("w", encoding="utf-8") as f:
            f.write("\t".join(fieldnames) + "\n")
            for row in failed_rows:
                f.write("\t".join(str(row[name]) for name in fieldnames) + "\n")

    print()
    print(f"total={total}")
    print(f"readable={readable}")
    print(f"correct={correct}")
    print(f"accuracy={correct / total:.4f}" if total else "accuracy=nan")
    print(f"csv={args.output_csv}")
    if args.error_log is not None:
        print(f"error_log={args.error_log}")
    if args.save_panel_dir is not None:
        print(f"panels={args.save_panel_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
