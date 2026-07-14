#!/usr/bin/env python3
"""Generate problem-only debug images for bin-number segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from extract_digit_crops import (
    IMAGE_SUFFIXES,
    WIDE_DIGIT_ROI_RATIOS,
    compute_panel_split,
    find_black_digit_panel,
    _slot_split_score,
)
from recognize_bin_labels import (
    assess_digit_panel_quality,
    crop_bin_roi_from_warp,
    preprocess_label_image,
    rotate_warp,
)


def _iter_label_images(dataset_dir: Path):
    for label_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        for path in sorted(p for p in label_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            yield label_dir.name, path


def _best_panel(path: Path, image, label: str, min_blur: float, min_label_quality: float):
    preprocessed = preprocess_label_image(
        path,
        image,
        min_blur=min_blur,
        min_label_quality=min_label_quality,
    )
    if preprocessed["error"] is not None:
        return None, preprocessed["error"]["code"]

    best = None
    best_payload = None
    for rotation in (0, 90, 180, 270):
        oriented = rotate_warp(preprocessed["warped"], rotation)
        search_roi, _ = crop_bin_roi_from_warp(oriented, WIDE_DIGIT_ROI_RATIOS)
        panel, box, found = find_black_digit_panel(search_roi, digit_count=len(label))
        if not found:
            continue

        quality_code = assess_digit_panel_quality(panel)
        score = _slot_split_score(panel, len(label))
        if quality_code is not None:
            # A real but low-quality number panel should still beat a clean
            # wrong subject such as table shadow. Keep the quality code as a
            # warning after orientation selection instead of making it a hard
            # loser during selection.
            score -= 0.05
        candidate = (score, rotation)
        if best is None or candidate > best:
            best = candidate
            best_payload = {
                "rotation": rotation,
                "search_roi": search_roi,
                "panel": panel,
                "box": box,
                "quality_code": quality_code,
            }

    if best_payload is None:
        return None, "ERR_BLACK_PANEL_NOT_FOUND"
    return best_payload, "OK"


def _segmentation_warnings(panel, label: str) -> list[str]:
    cuts, crop_y0, crop_y1, runs, slot_priors, smooth, core_height = compute_panel_split(panel, len(label))
    warnings = []
    valley_threshold = max(1.2, core_height * 0.04)

    for index, prior in enumerate(slot_priors):
        radius = max(5, int(round(panel.shape[1] * 0.045)))
        lo = max(0, prior - radius)
        hi = min(panel.shape[1] - 1, prior + radius)
        if hi <= lo:
            warnings.append(f"WARN_SLOT_{index + 1}_EMPTY_WINDOW")
            continue
        if float(smooth[lo : hi + 1].min()) > valley_threshold:
            warnings.append(f"WARN_SLOT_{index + 1}_NO_CLEAR_VALLEY")

    for index, digit in enumerate(label):
        x0 = max(0, cuts[index])
        x1 = min(panel.shape[1], cuts[index + 1])
        crop = panel[crop_y0:crop_y1, x0:x1]
        if crop.size == 0:
            warnings.append(f"WARN_CROP_{index}_{digit}_EMPTY")
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        threshold = max(110, float(np.percentile(gray, 82)))
        bright_ratio = float((gray > threshold).mean())
        if crop.shape[1] < 24:
            warnings.append(f"WARN_CROP_{index}_{digit}_NARROW_{crop.shape[1]}px")
        if bright_ratio < 0.012:
            warnings.append(f"WARN_CROP_{index}_{digit}_LOW_STROKE_{bright_ratio:.3f}")

    return warnings


def _fit_width(image, width: int):
    if image.shape[1] == width:
        return image
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def _draw_debug(label: str, path: Path, payload: dict, warnings: list[str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    search_roi = payload["search_roi"]
    panel = payload["panel"]
    box = payload["box"]
    rotation = payload["rotation"]

    top = search_roi.copy()
    x, y, w, h = box
    cv2.rectangle(top, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(
        top,
        f"{label} {path.stem} rot={rotation}",
        (4, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

    cuts, _, _, runs, slot_priors, smooth, core_height = compute_panel_split(panel, len(label))
    valley_threshold = max(1.2, core_height * 0.04)

    scale = 3
    mid = cv2.resize(panel, (panel.shape[1] * scale, panel.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    overlay = mid.copy()

    # Yellow slot-prior search bands. These are intentionally thick because
    # they are the physical slot prior, not a final cut line.
    for prior in slot_priors:
        radius = max(5, int(round(panel.shape[1] * 0.045)))
        x0 = max(0, prior - radius) * scale
        x1 = min(panel.shape[1] - 1, prior + radius) * scale
        cv2.rectangle(overlay, (x0, 0), (x1, mid.shape[0] - 1), (0, 255, 255), -1)
    mid = cv2.addWeighted(overlay, 0.22, mid, 0.78, 0)

    for start, end, _ in runs:
        cv2.rectangle(mid, (start * scale, 0), ((end + 1) * scale, mid.shape[0] - 1), (0, 255, 0), 2)
    for cut in cuts[1:-1]:
        cv2.line(mid, (cut * scale, 0), (cut * scale, mid.shape[0] - 1), (0, 0, 255), 3)
    for cut in (cuts[0], cuts[-1]):
        cv2.line(mid, (cut * scale, 0), (cut * scale, mid.shape[0] - 1), (255, 0, 0), 4)

    plot_h = 230
    plot_w = panel.shape[1] * scale
    plot = np.full((plot_h, plot_w, 3), 255, np.uint8)
    if smooth.size:
        plot_smooth = np.repeat(smooth, scale)
        value_scale = (plot_h - 18) / max(1.0, float(plot_smooth.max()))
        points = [
            (xx, max(0, min(plot_h - 1, plot_h - 1 - int(round(float(value) * value_scale)))))
            for xx, value in enumerate(plot_smooth)
        ]
        for point_a, point_b in zip(points, points[1:]):
            cv2.line(plot, point_a, point_b, (0, 0, 0), 1)
        valley_y = plot_h - 1 - int(round(valley_threshold * value_scale))
        cv2.line(plot, (0, valley_y), (plot_w - 1, valley_y), (180, 180, 180), 1)

    for prior in slot_priors:
        radius = max(5, int(round(panel.shape[1] * 0.045)))
        x0 = max(0, prior - radius) * scale
        x1 = min(panel.shape[1] - 1, prior + radius) * scale
        cv2.rectangle(plot, (x0, 0), (x1, plot_h - 1), (0, 255, 255), 1)
    for start, end, _ in runs:
        cv2.rectangle(plot, (start * scale, 0), ((end + 1) * scale, plot_h - 1), (0, 255, 0), 2)
    for cut in cuts[1:-1]:
        cv2.line(plot, (cut * scale, 0), (cut * scale, plot_h - 1), (0, 0, 255), 3)
    for cut in (cuts[0], cuts[-1]):
        cv2.line(plot, (cut * scale, 0), (cut * scale, plot_h - 1), (255, 0, 0), 4)

    warning_text = "; ".join(warnings[:3])
    if len(warnings) > 3:
        warning_text += f"; +{len(warnings) - 3}"
    if warning_text:
        cv2.putText(plot, warning_text[:90], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    width = max(top.shape[1], mid.shape[1], plot.shape[1])
    canvas = np.vstack([_fit_width(top, width), _fit_width(mid, width), _fit_width(plot, width)])
    cv2.imwrite(str(output_path), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create problem-only debug images for bin-number segmentation.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Folder containing label subfolders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for problem debug images.")
    parser.add_argument("--min-blur", type=float, default=35.0)
    parser.add_argument("--min-label-quality", type=float, default=0.55)
    parser.add_argument(
        "--include-ok",
        action="store_true",
        help="Also write debug images for samples without warnings.",
    )
    args = parser.parse_args()

    debug_dir = args.output_dir / "debug_boundaries"
    records = []
    total = 0
    written = 0

    for label, path in _iter_label_images(args.dataset_dir):
        total += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            records.append((label, path.name, "ERR_IMAGE_READ"))
            continue

        payload, code = _best_panel(path, image, label, args.min_blur, args.min_label_quality)
        if payload is None:
            records.append((label, path.name, code))
            continue
        if payload["quality_code"] is not None:
            warnings = [payload["quality_code"]]
        else:
            warnings = _segmentation_warnings(payload["panel"], label)

        if warnings or args.include_ok:
            _draw_debug(
                label,
                path,
                payload,
                warnings or ["OK"],
                debug_dir / label / f"{path.stem}_rot{payload['rotation']}_debug.png",
            )
            written += 1

        if warnings:
            records.append((label, path.name, *warnings))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "problem_files.txt").write_text(
        "\n".join("\t".join(map(str, record)) for record in records),
        encoding="utf-8",
    )
    print(f"checked={total} problem_records={len(records)} debug_images={written}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
