#!/usr/bin/env python3
"""Run PaddleOCR eval and save error-focused debug artifacts.

This script is intentionally noisy on disk and quiet in terminal. It clears the
output directory, evaluates every image, then groups logs and intermediate
images by error category:

- ERR_BLACK_PANEL_NOT_FOUND: all four rotated wide ROIs in one debug image.
- ERR_LABEL_NOT_FOUND / ERR_LABEL_INCOMPLETE / ERR_LABEL_GEOMETRY / ERR_BLUR:
  original image plus available label/preprocess diagnostics.
- OCR_MISMATCH: final OCR input panel with expected/pred/raw/score overlay.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from evaluate_paddle_text_recognition import (
    best_panel,
    black_rectangle_metrics,
    iter_images,
    prepare_paddle_panel,
    prepare_paddle_panel_stages,
    recognize_panel,
)
from extract_digit_crops import (
    WIDE_DIGIT_ROI_RATIOS,
    black_panel_edge_continuity_score,
    digit_stroke_layout_score,
    find_black_rect_edge_quad,
    find_black_digit_panel,
    local_equalized_gray,
    panel_selection_score,
    qr_like_score,
    tighten_black_rect_bounds,
    _slot_split_score,
)
from recognize_bin_labels import (
    DEFAULT_WARP_SIZE,
    blur_score,
    crop_bin_roi_from_warp,
    find_label_quad,
    find_inner_white_label_quad,
    has_incomplete_label_candidate,
    label_geometry_quality,
    preprocess_label_image,
    rotate_warp,
    warp_label,
)


FIELDNAMES = [
    "expected",
    "file",
    "pred",
    "raw_text",
    "score",
    "ok",
    "code",
    "debug_code",
    "quality_code",
    "rotation",
    "panel_path",
    "debug_image",
]


def safe_name(path: Path) -> str:
    return f"{path.parent.name}_{path.stem}"


def put_lines(image, lines, origin=(8, 24), scale=0.55, color=(0, 0, 255), thickness=2):
    out = image.copy()
    x, y = origin
    for line in lines:
        cv2.putText(out, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += int(26 * scale / 0.55)
    return out


def resize_to_width(image, width):
    if image.shape[1] == width:
        return image
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_caption(image, lines, scale=0.42, color=(0, 0, 255), thickness=1, value=255):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    line_h = max(14, int(round(22 * scale / 0.42)))
    header_h = line_h * max(1, len(lines)) + 6
    text_width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(str(line), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        text_width = max(text_width, tw)
    width = max(image.shape[1], text_width + 12)
    if width > image.shape[1]:
        image = pad_to_size(image, width, image.shape[0], value=value)
    header = np.full((header_h, width, 3), value, dtype=np.uint8)
    y = line_h
    for line in lines:
        cv2.putText(header, str(line), (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_h
    return np.vstack([header, image])


def pad_to_size(image, width, height, value=255):
    canvas = np.full((height, width, 3), value, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h = min(height, image.shape[0])
    w = min(width, image.shape[1])
    canvas[:h, :w] = image[:h, :w]
    return canvas


def make_grid(images, cols=2, pad=8, value=255):
    if not images:
        return np.full((80, 240, 3), value, dtype=np.uint8)
    max_w = max(image.shape[1] for image in images)
    max_h = max(image.shape[0] for image in images)
    cells = [pad_to_size(image, max_w, max_h, value=value) for image in images]
    rows = []
    for start in range(0, len(cells), cols):
        row_cells = cells[start : start + cols]
        while len(row_cells) < cols:
            row_cells.append(np.full((max_h, max_w, 3), value, dtype=np.uint8))
        rows.append(np.hstack(row_cells))
    grid = np.vstack(rows)
    if pad <= 0:
        return grid
    return cv2.copyMakeBorder(grid, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(value, value, value))


def compact_vstack(images, pad=8, value=255):
    """Stack images without forcing every row to the same height."""
    if not images:
        return np.full((80, 240, 3), value, dtype=np.uint8)
    width = max(image.shape[1] for image in images)
    rows = [pad_to_size(image, width, image.shape[0], value=value) for image in images]
    grid = np.vstack(rows)
    if pad <= 0:
        return grid
    return cv2.copyMakeBorder(grid, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(value, value, value))


def make_black_missing_grid(original, roi_panels, pad=8, value=255):
    roi_row = make_grid(roi_panels, cols=4, pad=0, value=value)
    width = max(original.shape[1], roi_row.shape[1])
    resized_original = resize_to_width(original, width)
    original_row = pad_to_size(resized_original, width, resized_original.shape[0], value=value)
    roi_row = pad_to_size(roi_row, width, roi_row.shape[0], value=value)
    grid = np.vstack([original_row, roi_row])
    return cv2.copyMakeBorder(grid, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(value, value, value))


def draw_quad(image, quad, color=(0, 255, 0)):
    out = image.copy()
    if quad is not None:
        pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)
    return out


def draw_detected_label_boxes(image):
    out = image.copy()
    outer_quad = find_label_quad(image)
    if outer_quad is None:
        return out, False, False

    out = draw_quad(out, outer_quad, color=(0, 255, 0))

    width, height = DEFAULT_WARP_SIZE
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    outer_matrix = cv2.getPerspectiveTransform(outer_quad, dst)
    outer_warped = cv2.warpPerspective(image, outer_matrix, (width, height))
    inner_quad = find_inner_white_label_quad(outer_warped)
    if inner_quad is None:
        return out, True, False

    inverse_matrix = cv2.getPerspectiveTransform(dst, outer_quad.astype("float32"))
    inner_original = cv2.perspectiveTransform(inner_quad.reshape(1, -1, 2).astype("float32"), inverse_matrix)[0]
    out = draw_quad(out, inner_original, color=(255, 0, 0))
    return out, True, True


def draw_debug_black_panel_edge(image):
    """Draw the black rectangle edge that the debug detector sees."""
    vis = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    if height < 8 or width < 16:
        return vis, False

    dark = (gray < 115).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((13, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.24 or h < height * 0.26:
            continue
        ratio = w / float(max(1, h))
        if not (1.25 <= ratio <= 7.2):
            continue
        roi = gray[y : y + h, x : x + w]
        roi_dark = roi < 115
        dark_ratio = float(roi_dark.mean())
        if dark_ratio < 0.45:
            continue
        row_dark = roi_dark.mean(axis=1)
        col_dark = roi_dark.mean(axis=0)
        continuous_rows = float((row_dark > 0.40).mean())
        continuous_cols = float((col_dark > 0.40).mean())
        if continuous_rows < 0.35 or continuous_cols < 0.42:
            continue
        area = cv2.contourArea(contour)
        score = area * min(1.0, ratio / 2.2) * (0.45 * dark_ratio + 0.30 * continuous_cols + 0.25 * continuous_rows)
        candidates.append((score, contour))

    if not candidates:
        return vis, False

    _, contour = max(candidates, key=lambda item: item[0])
    x, y, w, h = cv2.boundingRect(contour)
    x, y, w, h = tighten_black_rect_bounds(gray, x, y, w, h, threshold=115)
    local_image = image[y:y + h, x:x + w]
    edge_box = find_black_rect_edge_quad(local_image)
    if edge_box is not None:
        box = edge_box.copy()
        box[:, 0] += x
        box[:, 1] += y
        box = box.astype("int32")
        cv2.polylines(vis, [box.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)
        return vis, True

    local = gray[y:y + h, x:x + w]
    local_dark = (local < 115).astype("uint8")
    local_dark = cv2.morphologyEx(local_dark, cv2.MORPH_CLOSE, np.ones((13, 5), np.uint8))
    local_contours, _ = cv2.findContours(local_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if local_contours:
        local_contour = max(local_contours, key=cv2.contourArea)
        box = cv2.boxPoints(cv2.minAreaRect(local_contour))
        box[:, 0] += x
        box[:, 1] += y
        box = box.astype("int32")
        cv2.polylines(vis, [box.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)
    else:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis, True


def make_equalized_panel(image, title):
    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    eq = np.maximum(gray_raw, local_equalized_gray(image))
    eq_bgr = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return add_caption(eq_bgr, [title], scale=0.5, thickness=1, color=(255, 0, 0))


def make_original_panel(path, image, expected, code, extra_lines=()):
    original, outer_found, inner_found = draw_detected_label_boxes(image)
    original = resize_to_width(original, 640)
    lines = [
        f"ORIGINAL: {expected}/{path.name}",
        f"code={code}",
        f"outer_label=green:{outer_found}",
        f"inner_white=blue:{inner_found}",
        *extra_lines,
    ]
    return put_lines(original, lines, scale=0.55, thickness=2)


def make_original_with_equalized_panel(path, image, expected, code, extra_lines=()):
    original = make_original_panel(path, image, expected, code, extra_lines=extra_lines)
    equalized = resize_to_width(make_equalized_panel(image, "label detection CLAHE helper"), original.shape[1])
    return np.vstack([original, equalized])


def best_raw_black_candidate(search_roi):
    gray_raw = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY) if search_roi.ndim == 3 else search_roi
    gray = np.maximum(gray_raw, local_equalized_gray(search_roi))
    height, width = gray.shape
    dark = (gray < 105).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w <= 0 or h <= 0:
            continue
        roi = gray[y : y + h, x : x + w]
        ratio = w / float(max(1, h))
        area_ratio = area / float(max(1, width * height))
        dark_ratio = float((roi < 105).mean())
        bright_ratio = float((roi > 135).mean())
        row_dark_ratio = (roi < 105).mean(axis=1)
        col_dark_ratio = (roi < 105).mean(axis=0)
        continuous_dark_cols = float((col_dark_ratio > 0.45).mean())
        continuous_dark_rows = float((row_dark_ratio > 0.45).mean())
        center_x = (x + w * 0.5) / float(max(1, width))
        center_y = (y + h * 0.5) / float(max(1, height))

        reasons = []
        if w < width * 0.28:
            reasons.append("w_small")
        if h < height * 0.18:
            reasons.append("h_small")
        if not (1.35 <= ratio <= 4.8):
            reasons.append("bad_aspect")
        if not (0.055 <= area_ratio <= 0.42):
            reasons.append("bad_area")
        if center_x > 0.76:
            reasons.append("too_right")
        if center_y < 0.20:
            reasons.append("too_high")
        if dark_ratio < 0.46:
            reasons.append("low_dark")
        strong_black_panel = ratio >= 1.55 and dark_ratio > 0.62 and continuous_dark_rows > 0.50
        min_bright_ratio = 0.0 if strong_black_panel else 0.01
        if bright_ratio < min_bright_ratio:
            reasons.append("low_stroke")
        if ratio >= 1.70 and dark_ratio > 0.45:
            min_continuous_cols = 0.30
            min_continuous_rows = 0.25
        else:
            min_continuous_cols = 0.50 if ratio >= 2.35 else 0.62
            min_continuous_rows = 0.42
        if continuous_dark_cols < min_continuous_cols:
            reasons.append("broken_cols")
        if continuous_dark_rows < min_continuous_rows:
            reasons.append("broken_rows")

        score = (
            area
            * max(0.05, min(1.0, ratio / 2.2))
            * max(0.05, min(1.0, dark_ratio))
            * max(0.05, continuous_dark_cols)
            * max(0.05, continuous_dark_rows)
            * max(0.05, 1.0 - max(0.0, center_x - 0.55))
            * max(0.05, min(1.0, center_y + 0.25))
        )
        item = {
            "score": float(score),
            "box": (x, y, w, h),
            "ratio": ratio,
            "area_ratio": area_ratio,
            "dark_ratio": dark_ratio,
            "bright_ratio": bright_ratio,
            "continuous_dark_cols": continuous_dark_cols,
            "continuous_dark_rows": continuous_dark_rows,
            "reasons": reasons,
        }
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def make_rotation_roi_panels(preprocessed, path, expected):
    panels = []
    for rotation in (0, 90, 180, 270):
        oriented = rotate_warp(preprocessed["warped"], rotation)
        search_roi, _ = crop_bin_roi_from_warp(oriented, WIDE_DIGIT_ROI_RATIOS)
        panel, box, found = find_black_digit_panel(search_roi, digit_count=len(expected))

        vis = search_roi.copy()
        x, y, w, h = box
        color = (0, 180, 0) if found else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        score = panel_selection_score(panel, len(expected)) if found and panel is not None else 0.0
        slot = _slot_split_score(panel, len(expected)) if found and panel is not None else 0.0
        layout = digit_stroke_layout_score(panel, len(expected)) if found and panel is not None else 0.0
        edge = black_panel_edge_continuity_score(panel) if found and panel is not None else 0.0
        qr = qr_like_score(panel, len(expected)) if found and panel is not None else 0.0
        raw_candidate = None if found else best_raw_black_candidate(search_roi)
        raw_lines = []
        if raw_candidate is not None:
            rx, ry, rw, rh = raw_candidate["box"]
            edge_region = search_roi[
                max(0, ry - 2) : min(search_roi.shape[0], ry + rh + 2),
                max(0, rx - 2) : min(search_roi.shape[1], rx + rw + 2),
            ]
            edge_quad = find_black_rect_edge_quad(edge_region)
            if edge_quad is not None:
                edge_quad = edge_quad.copy()
                edge_quad[:, 0] += max(0, rx - 2)
                edge_quad[:, 1] += max(0, ry - 2)
                cv2.polylines(vis, [edge_quad.astype("int32").reshape(-1, 1, 2)], True, (0, 165, 255), 2)
            else:
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 165, 255), 2)
            raw_lines = [
                f"edge_quad={'yes' if edge_quad is not None else 'no'}",
                f"raw_score={raw_candidate['score']:.1f}",
                f"raw_ratio={raw_candidate['ratio']:.2f}",
                f"reject={','.join(raw_candidate['reasons'][:3]) or 'none'}",
            ]
        lines = [
            f"{expected}/{path.name}",
            f"rot={rotation} found={found}",
            f"score={score:.4f}",
            f"slot={slot:.3f} layout={layout:.3f}",
            f"edge={edge:.3f} qr={qr:.3f}",
            *raw_lines,
        ]
        panels.append(put_lines(vis, lines, color=color, scale=0.45, thickness=1))
        eq_vis = resize_to_width(make_equalized_panel(search_roi, f"rot={rotation} ROI CLAHE helper"), vis.shape[1])
        panels.append(eq_vis)
    return panels


def save_black_panel_missing_debug(path, image, expected, output_dir, min_blur, min_label_quality):
    preprocessed = preprocess_label_image(path, image, min_blur=min_blur, min_label_quality=min_label_quality)
    if preprocessed["error"] is not None:
        return ""

    original_panel = make_original_with_equalized_panel(path, image, expected, "ERR_BLACK_PANEL_NOT_FOUND")
    panels = make_rotation_roi_panels(preprocessed, path, expected)

    debug = make_black_missing_grid(original_panel, panels)
    out_path = output_dir / "debug_images" / "ERR_BLACK_PANEL_NOT_FOUND" / f"{safe_name(path)}_wide_rois.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug)
    return str(out_path)


def save_label_preprocess_debug(path, image, expected, code, output_dir):
    blur = blur_score(image) if image is not None else 0.0
    incomplete_candidate = has_incomplete_label_candidate(image) if image is not None else False
    quad = find_label_quad(image) if image is not None else None
    quality = label_geometry_quality(quad) if quad is not None else 0.0
    warped, _ = warp_label(image) if image is not None else (None, None)

    original = make_original_with_equalized_panel(
        path,
        image,
        expected,
        code,
        extra_lines=[
            f"blur={blur:.1f}",
            f"quad_found={quad is not None}",
            f"label_quality={quality:.3f}",
            f"incomplete_candidate={incomplete_candidate}",
        ],
    )

    panels = [original]
    if warped is not None:
        warped_vis = resize_to_width(warped, 420)
        warped_vis = put_lines(warped_vis, ["warped label"], scale=0.5, thickness=1)
        panels.append(warped_vis)

    debug = make_grid(panels, cols=1)
    out_path = output_dir / "debug_images" / code / f"{safe_name(path)}_label_preprocess.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug)
    return str(out_path)


def save_ocr_mismatch_debug(
    path,
    image,
    payload,
    raw_panel,
    panel,
    expected,
    pred,
    raw_text,
    score,
    output_dir,
    category="OCR_MISMATCH",
):
    original = make_original_with_equalized_panel(
        path,
        image,
        expected,
        category,
        extra_lines=[f"expected={expected}", f"pred={pred or '-'} raw={raw_text or '-'} score={score:.4f}"],
    )

    preprocessed = preprocess_label_image(path, image)
    roi_vis = None
    rotation_overview = None
    if preprocessed["error"] is None:
        rotation_overview = make_grid(make_rotation_roi_panels(preprocessed, path, expected), cols=2, pad=4)
        oriented = rotate_warp(preprocessed["warped"], payload["rotation"])
        search_roi, _ = crop_bin_roi_from_warp(oriented, WIDE_DIGIT_ROI_RATIOS)
        roi_vis = search_roi.copy()
        x, y, w, h = payload["box"]
        cv2.rectangle(roi_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_vis = add_caption(
            roi_vis,
            [
                f"selected ROI rot={payload['rotation']}",
                f"box={x},{y},{w},{h}",
                f"panel_score={payload['score']:.4f}",
            ],
            scale=0.45,
            thickness=1,
            color=(0, 180, 0),
        )
        roi_eq = resize_to_width(make_equalized_panel(search_roi, f"selected ROI CLAHE helper rot={payload['rotation']}"), roi_vis.shape[1])
        roi_vis = np.vstack([roi_vis, roi_eq])

    stages = prepare_paddle_panel_stages(raw_panel, digit_count=len(expected))

    def make_stage_panel(stage_image, title):
        metrics = black_rectangle_metrics(stage_image)
        edge_vis, has_edge = draw_debug_black_panel_edge(stage_image)
        return add_caption(
            edge_vis,
            [
                title,
                f"debug_edge={'yes' if has_edge else 'no'}",
                f"shape={stage_image.shape[1]}x{stage_image.shape[0]}",
                "black_rect ratio={ratio:.2f} dark={dark_ratio:.2f} cols={continuous_cols:.2f} rows={continuous_rows:.2f}".format(
                    **metrics
                ),
            ],
            scale=0.5,
            thickness=1,
        )

    first_vis = make_stage_panel(stages["first_black_panel"], "1st black panel from ROI detector")
    trim_vis = make_stage_panel(stages["after_first_trim"], "after white-margin trim")
    refine_vis = make_stage_panel(stages["after_second_refine"], "2nd black-panel refine result")

    ocr_edge_vis, has_ocr_edge = draw_debug_black_panel_edge(panel)
    ocr_vis = add_caption(
        ocr_edge_vis,
        [
            "final PaddleOCR input",
            f"debug_edge={'yes' if has_ocr_edge else 'no'}",
            f"shape={panel.shape[1]}x{panel.shape[0]}",
            f"expected={expected}",
            f"pred={pred or '-'} raw={raw_text or '-'} score={score:.4f}",
        ],
        scale=0.4,
        thickness=1,
    )

    panels = [original]
    if rotation_overview is not None:
        panels.append(rotation_overview)
    if roi_vis is not None:
        panels.append(roi_vis)
    panels.extend([first_vis, trim_vis, refine_vis, ocr_vis])
    debug = compact_vstack(panels)
    out_path = output_dir / "debug_images" / category / f"{safe_name(path)}_ocr_input.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug)
    return str(out_path)


def write_grouped_error_logs(rows, output_dir):
    by_debug_code = defaultdict(list)
    for row in rows:
        if not row["ok"]:
            by_debug_code[row["debug_code"]].append(row)

    error_root = output_dir / "errors_by_code"
    error_root.mkdir(parents=True, exist_ok=True)
    for debug_code, items in sorted(by_debug_code.items()):
        path = error_root / f"{debug_code}.tsv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
            writer.writeheader()
            writer.writerows(items)

    summary_path = output_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Bin Number Paddle Debug Summary\n\n")
        f.write(f"total: {len(rows)}\n\n")
        f.write("## Error Counts\n\n")
        for debug_code, items in sorted(by_debug_code.items(), key=lambda item: (-len(item[1]), item[0])):
            f.write(f"- {debug_code}: {len(items)}\n")
        f.write("\n## Files By Error\n")
        for debug_code, items in sorted(by_debug_code.items(), key=lambda item: item[0]):
            f.write(f"\n### {debug_code}\n\n")
            for row in items:
                rel = f"{Path(row['file']).parent.name}/{Path(row['file']).name}"
                pred = row["pred"] or "-"
                f.write(f"- {rel}: expected={row['expected']}, pred={pred}, score={row['score']}, code={row['code']}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate bin-number OCR and save grouped debug artifacts.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/workspace/huangjie/pure_vision_detection/datasets/bin_number_test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/huangjie/pure_vision_detection/datasets/bin_number_debug_test"),
    )
    parser.add_argument("--min-blur", type=float, default=35.0)
    parser.add_argument("--min-label-quality", type=float, default=0.55)
    parser.add_argument(
        "--save-correct-debug",
        action="store_true",
        help="Also save the full intermediate debug composite for correctly recognized samples.",
    )
    args = parser.parse_args()

    from paddleocr import TextRecognition

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_root = args.output_dir / "paddle_panels"
    panel_root.mkdir(parents=True, exist_ok=True)

    recognizer = TextRecognition()
    rows = []
    total = 0
    correct = 0

    for expected, path in iter_images(args.dataset_dir):
        total += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        pred = ""
        raw_text = ""
        score = 0.0
        panel_path = ""
        debug_image = ""
        rotation = ""
        quality_code = ""

        if image is None:
            code = "ERR_IMAGE_READ"
            debug_code = code
        else:
            payload, code, quality_code_value = best_panel(path, image, expected, args.min_blur, args.min_label_quality)
            quality_code = quality_code_value or ""
            debug_code = code
            if payload is None:
                if code == "ERR_BLACK_PANEL_NOT_FOUND":
                    debug_image = save_black_panel_missing_debug(
                        path, image, expected, args.output_dir, args.min_blur, args.min_label_quality
                    )
                elif code.startswith("ERR_LABEL") or code == "ERR_BLUR":
                    debug_image = save_label_preprocess_debug(path, image, expected, code, args.output_dir)
            else:
                rotation = str(payload["rotation"])
                panel = prepare_paddle_panel(payload["panel"], digit_count=len(expected))
                panel_dir = panel_root / expected
                panel_dir.mkdir(parents=True, exist_ok=True)
                panel_path = str(panel_dir / f"{path.stem}_rot{payload['rotation']}.png")
                cv2.imwrite(panel_path, panel)
                pred, score, raw_text = recognize_panel(recognizer, panel)
                if pred != expected:
                    debug_code = "OCR_MISMATCH"
                    debug_image = save_ocr_mismatch_debug(
                        path,
                        image,
                        payload,
                        payload["panel"],
                        panel,
                        expected,
                        pred,
                        raw_text,
                        score,
                        args.output_dir,
                    )
                elif args.save_correct_debug:
                    debug_image = save_ocr_mismatch_debug(
                        path,
                        image,
                        payload,
                        payload["panel"],
                        panel,
                        expected,
                        pred,
                        raw_text,
                        score,
                        args.output_dir,
                        category="OCR_CORRECT",
                    )

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
                "debug_code": debug_code,
                "quality_code": quality_code,
                "rotation": rotation,
                "panel_path": panel_path,
                "debug_image": debug_image,
            }
        )

    eval_csv = args.output_dir / "bin_number_paddle_eval.csv"
    with eval_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    error_tsv = args.output_dir / "bin_number_paddle_errors.tsv"
    with error_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(row for row in rows if not row["ok"])

    write_grouped_error_logs(rows, args.output_dir)

    print(f"total={total}")
    print(f"correct={correct}")
    print(f"accuracy={correct / total:.4f}" if total else "accuracy=nan")
    print(f"output_dir={args.output_dir}")
    print(f"eval_csv={eval_csv}")
    print(f"error_tsv={error_tsv}")
    print(f"summary={args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
