#!/usr/bin/env python3
"""Extract labeled single-digit crops from full bin-label images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from recognize_bin_labels import (
    assess_digit_panel_quality,
    crop_bin_roi_from_warp,
    preprocess_label_image,
    rotate_warp,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
# After warping the white label to a square, the black number block is measured
# at roughly x=3.5..20.5 mm and y=27.1..34.7 mm on a 43.5x43.5 mm label. Use a
# padded search region around that physical prior; the actual black rectangle
# is found again inside this wider region.
WIDE_DIGIT_ROI_RATIOS = (0.00, 0.52, 0.58, 0.88)
CANONICAL_PANEL_SIZE = (200, 90)
# Learned from current labels after square label warp and canonical panel
# normalization. Digits start from the left side, while the right side of the
# black panel is padding. These are boundary priors between digit slots for a
# 200 px wide panel, not equal divisions of the full black rectangle.
CANONICAL_SLOT_CUTS = (48, 84, 120)
CANONICAL_TEXT_RIGHT = {
    1: 58,
    2: 96,
    3: 132,
    4: 168,
}


def order_points(points):
    pts = np.asarray(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def collect_images(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            paths.append(path)
    return paths


def trim_panel_whitespace(panel):
    """Trim white margins after the black panel has been rectified."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    # Use a stricter black test here than the broad detector. Tan paper and
    # blurred white label borders can be darker than expected, but they are
    # still not the physical black number panel.
    dark = (gray < 92).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((17, 5), np.uint8)).astype(bool)

    row_ratio = dark.mean(axis=1)
    rows = np.where(row_ratio > 0.50)[0]
    if not rows.size:
        dark = (gray < 115).astype("uint8")
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((17, 5), np.uint8)).astype(bool)
        row_ratio = dark.mean(axis=1)
        rows = np.where(row_ratio > 0.45)[0]
    if rows.size:
        y0 = int(rows.min())
        y1 = int(rows.max()) + 1
    else:
        y0, y1 = 0, panel.shape[0]

    col_ratio = dark[y0:y1].mean(axis=0)
    cols = np.where(col_ratio > 0.34)[0]
    if not cols.size:
        loose_dark = (gray < 115).astype("uint8")
        loose_dark = cv2.morphologyEx(loose_dark, cv2.MORPH_CLOSE, np.ones((17, 5), np.uint8)).astype(bool)
        col_ratio = loose_dark[y0:y1].mean(axis=0)
        cols = np.where(col_ratio > 0.28)[0]
    if cols.size:
        x0 = int(cols.min())
        x1 = int(cols.max()) + 1
    else:
        x0, x1 = 0, panel.shape[1]

    # Keep the black rectangle itself, not the surrounding white label border.
    # Digit strokes already have black context inside the rectangle, so adding
    # top/bottom padding here mostly reintroduces the white edge we just found.
    x0 = max(0, x0)
    x1 = min(panel.shape[1], x1)
    y0 = max(0, y0)
    y1 = min(panel.shape[0], y1)
    return panel[y0:y1, x0:x1]


def normalize_panel_size(panel):
    """Put every detected black panel into the same physical coordinate frame."""
    return cv2.resize(panel, CANONICAL_PANEL_SIZE, interpolation=cv2.INTER_LINEAR)


def _digit_stroke_score(panel) -> float:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    if gray.size == 0:
        return 0.0
    bright_ratio = float((gray > 135).mean())
    contrast = float(np.percentile(gray, 92) - np.percentile(gray, 12))
    return bright_ratio * 2.0 + contrast / 255.0


def _leading_digit_signal_score(panel, digit_count: int | None = None) -> float:
    """Measure whether the leftmost digit survives a panel rectification."""
    gray = balanced_gray(panel)
    if gray.size == 0:
        return 0.0
    height, width = gray.shape
    y0 = max(0, int(round(height * 0.12)))
    y1 = min(height, int(round(height * 0.90)))
    if digit_count in CANONICAL_TEXT_RIGHT:
        first_slot_right = int(round(48 * width / float(CANONICAL_PANEL_SIZE[0])))
        text_right = int(round(CANONICAL_TEXT_RIGHT[digit_count] * width / float(CANONICAL_PANEL_SIZE[0])))
        first_slot_right = min(width, max(first_slot_right, int(round(text_right / max(1, digit_count)))))
    else:
        first_slot_right = max(1, int(round(width * 0.25)))
    region = gray[y0:y1, :first_slot_right]
    if region.size == 0:
        return 0.0
    threshold = max(125.0, min(205.0, float(np.percentile(region, 90))))
    bright = region > threshold
    active_cols = float((bright.sum(axis=0) > max(1.0, region.shape[0] * 0.035)).mean())
    return float(bright.mean() * 3.0 + active_cols)


def digit_stroke_layout_score(panel, digit_count: int | None = None) -> float:
    """Score whether a black panel actually contains left-aligned digit strokes."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    if gray.size == 0:
        return 0.0
    height, width = gray.shape
    y0 = max(0, int(round(height * 0.12)))
    y1 = min(height, int(round(height * 0.90)))
    core = gray[y0:y1]
    if core.size == 0:
        return 0.0

    if digit_count in CANONICAL_TEXT_RIGHT:
        expected_right = int(round(CANONICAL_TEXT_RIGHT[digit_count] * width / float(CANONICAL_PANEL_SIZE[0])))
        x1 = min(width, max(12, expected_right + max(4, int(round(width * 0.04)))))
    else:
        x1 = min(width, int(round(width * 0.78)))
    digit_region = core[:, :x1]
    if digit_region.size == 0:
        return 0.0

    threshold = max(125.0, min(185.0, float(np.percentile(digit_region, 90))))
    bright = digit_region > threshold
    bright_ratio = float(bright.mean())
    col_counts = bright.sum(axis=0)
    row_counts = bright.sum(axis=1)
    active_cols = float((col_counts > max(1.0, digit_region.shape[0] * 0.045)).mean())
    active_rows = float((row_counts > max(1.0, digit_region.shape[1] * 0.012)).mean())

    # The actual printed number is left-aligned. A border/shadow strip may be a
    # perfect black rectangle, but it has no compact white strokes in this area.
    return bright_ratio * 3.0 + active_cols * 1.6 + active_rows * 0.35


def black_panel_edge_continuity_score(panel) -> float:
    """Score whether the rectified panel has continuous dark rectangle edges."""
    gray = balanced_gray(panel)
    if gray.size == 0:
        return 0.0
    height, width = gray.shape
    if height < 8 or width < 16:
        return 0.0
    dark = gray < 125

    top_h = max(2, int(round(height * 0.10)))
    bottom_h = max(2, int(round(height * 0.10)))
    left_w = max(2, int(round(width * 0.035)))
    right_w = max(2, int(round(width * 0.035)))

    top = dark[:top_h, :].mean(axis=0)
    bottom = dark[height - bottom_h :, :].mean(axis=0)
    left = dark[:, :left_w].mean(axis=1)
    right = dark[:, width - right_w :].mean(axis=1)

    top_score = float((top > 0.55).mean())
    bottom_score = float((bottom > 0.55).mean())
    left_score = float((left > 0.55).mean())
    right_score = float((right > 0.55).mean())
    horizontal = min(top_score, bottom_score)
    vertical = min(left_score, right_score)
    return float(0.70 * horizontal + 0.30 * vertical)


def qr_like_score(panel, digit_count: int | None = None) -> float:
    """Score how much a candidate looks like a 2-D QR/barcode patch.

    The number panel is a one-line, left-aligned text strip on a continuous
    black rectangle. QR-like distractors also have high contrast, but their
    bright structures spread in two dimensions and contain many square-ish
    connected components. This penalty is intentionally geometric, not tied to
    any single image.
    """
    gray = balanced_gray(panel)
    if gray.size == 0:
        return 1.0
    height, width = gray.shape
    y0 = max(0, int(round(height * 0.08)))
    y1 = min(height, int(round(height * 0.94)))
    core = gray[y0:y1]
    if core.size == 0:
        return 1.0

    threshold = max(125.0, min(185.0, float(np.percentile(core, 90))))
    bright = (core > threshold).astype("uint8")
    bright_ratio = float(bright.mean())
    if bright_ratio <= 0.003:
        return 0.0

    ys, xs = np.where(bright > 0)
    bright_width_ratio = (int(xs.max()) - int(xs.min()) + 1) / float(max(1, width))
    bright_height_ratio = (int(ys.max()) - int(ys.min()) + 1) / float(max(1, core.shape[0]))

    n_components, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    components = []
    square_like = 0
    medium_components = 0
    area = float(max(1, core.shape[0] * width))
    for index in range(1, n_components):
        comp_area = int(stats[index, cv2.CC_STAT_AREA])
        if comp_area < 4:
            continue
        comp_w = int(stats[index, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        aspect = comp_w / float(max(1, comp_h))
        area_ratio = comp_area / area
        components.append((comp_area, comp_w, comp_h, aspect, area_ratio))
        if area_ratio > 0.004 and 0.55 <= aspect <= 1.75 and comp_w >= width * 0.025 and comp_h >= core.shape[0] * 0.07:
            square_like += 1
        if area_ratio > 0.003:
            medium_components += 1

    # QR/barcode distractors are dense and two-dimensional. Real digits may be
    # tall, but the white strokes usually occupy only the left text run and do
    # not create many square modules across the panel.
    edge_continuity = black_panel_edge_continuity_score(panel)
    spread_2d = max(0.0, bright_height_ratio - 0.55) * max(0.0, bright_width_ratio - 0.34) * 3.2
    component_penalty = min(1.2, square_like * 0.22 + max(0, medium_components - max(3, (digit_count or 1) + 2)) * 0.08)
    density_penalty = max(0.0, bright_ratio - 0.18) * 2.0
    broken_edge_penalty = max(0.0, 0.78 - edge_continuity) * 1.25
    return float(min(2.8, spread_2d + component_penalty + density_penalty + broken_edge_penalty))


def panel_selection_score(panel, digit_count: int | None = None) -> float:
    """Shared score for choosing the correct rotation/black panel candidate."""
    slot_score = _slot_split_score(panel, digit_count) if digit_count is not None else 0.0
    layout_score = digit_stroke_layout_score(panel, digit_count)
    qr_penalty = qr_like_score(panel, digit_count)
    black_score = _black_rect_score(panel)
    edge_score = black_panel_edge_continuity_score(panel)
    return slot_score + layout_score * 0.70 + black_score * 0.30 + edge_score * 0.85 - qr_penalty * 1.45


def is_plausible_number_panel(panel, digit_count: int | None = None) -> bool:
    """Hard gate for the physical black number block.

    Scores are useful for ranking close candidates, but QR codes and random
    dark textures should not be allowed to compete with the number panel at
    all. The real target is a dark, horizontally long, edge-continuous panel.
    """
    if panel is None or panel.size == 0:
        return False
    analysis_panel = auto_balance_brightness(panel)
    black_score = _black_rect_score(analysis_panel)
    edge_score = black_panel_edge_continuity_score(analysis_panel)
    qr_score = qr_like_score(analysis_panel, digit_count)
    layout_score = digit_stroke_layout_score(analysis_panel, digit_count)
    if black_score < 0.50:
        return False
    # Digit layout is deliberately not a hard gate here. Perspective warp,
    # glare, blur, and white digits clipped by the candidate quadrilateral can
    # make the layout score unreliable even when the physical black panel and
    # its four edges are correct. Layout remains part of ranking and OCR
    # quality, but must not turn a valid black quadrilateral into
    # ERR_BLACK_PANEL_NOT_FOUND.
    strong_digit_panel = black_score >= 0.52 and qr_score <= 1.75
    # A visible Canny quadrilateral can have weak dark edge continuity after
    # glare/overexposure, while still being the correct physical panel. The
    # black-area and digit-layout gates remain strict; only the edge gate gets
    # this controlled relaxation.
    min_edge_score = 0.24 if strong_digit_panel else 0.66
    if edge_score < min_edge_score:
        return False
    if qr_score > 1.85 and edge_score < 0.82:
        return False
    return True


def _black_rect_score(panel) -> float:
    gray = balanced_gray(panel)
    if gray.size == 0:
        return 0.0
    height, width = gray.shape
    y0 = max(0, int(round(height * 0.08)))
    y1 = min(height, int(round(height * 0.94)))
    core = gray[y0:y1]
    dark = core < 120
    dark_ratio = float(dark.mean())
    col_dark = dark.mean(axis=0)
    row_dark = dark.mean(axis=1)
    continuous_cols = float((col_dark > 0.42).mean())
    continuous_rows = float((row_dark > 0.42).mean())
    ratio = width / float(max(1, height))
    ratio_score = min(1.0, max(0.1, ratio / 2.2))
    return (dark_ratio * 0.45 + continuous_cols * 0.35 + continuous_rows * 0.20) * ratio_score


def auto_balance_brightness(image):
    """Conservatively compress highlights when a crop is over-bright.

    Bright label crops can wash out the black-panel border and make threshold
    tests unstable. This only scales the luminance channel when the image is
    clearly bright; normal images are returned unchanged.
    """
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    p95 = float(np.percentile(gray, 95))
    p50 = float(np.percentile(gray, 50))
    scales = []
    if p95 > 220:
        scales.append(214.0 / max(1.0, p95))
    if p50 > 178:
        scales.append(174.0 / max(1.0, p50))
    if not scales:
        return image
    scale = max(0.72, min(scales))
    if image.ndim == 2:
        return np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def balanced_gray(image):
    """Return grayscale after conservative highlight compression."""
    balanced = auto_balance_brightness(image)
    return cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY) if balanced.ndim == 3 else balanced


def adaptive_dark_mask(image):
    """Create a visualization-friendly local dark mask after brightness balance."""
    gray = balanced_gray(image)
    height, width = gray.shape[:2]
    block = max(11, int(round(min(height, width) * 0.18)))
    if block % 2 == 0:
        block += 1
    block = min(block, 51 if min(height, width) >= 51 else block)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        5,
    )


def local_equalized_gray(image):
    """CLAHE gray image for local contrast decisions, not for final OCR crops."""
    image = auto_balance_brightness(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def tighten_black_rect_bounds(gray, x, y, w, h, threshold=115):
    """Tighten an approximate black panel box using row/column dark projection."""
    height, width = gray.shape
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x + w))
    y1 = min(height, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    roi = gray[y0:y1, x0:x1]
    dark = (roi < threshold).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((13, 5), np.uint8)).astype(bool)
    bright_strokes = roi > max(135, min(190, float(np.percentile(roi, 88))))
    signal = np.logical_or(dark, bright_strokes)

    row_ratio = dark.mean(axis=1)
    signal_row_ratio = signal.mean(axis=1)
    rows = np.where(np.logical_or(row_ratio > 0.50, signal_row_ratio > 0.58))[0]
    if not rows.size:
        rows = np.where(np.logical_or(row_ratio > 0.38, signal_row_ratio > 0.46))[0]
    if rows.size:
        y0 += int(rows.min())
        y1 = y0 + int(rows.max() - rows.min() + 1)

    roi = gray[y0:y1, x0:x1]
    dark = (roi < threshold).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((13, 5), np.uint8)).astype(bool)
    bright_strokes = roi > max(135, min(190, float(np.percentile(roi, 88))))
    signal = np.logical_or(dark, bright_strokes)
    col_ratio = dark.mean(axis=0)
    signal_col_ratio = signal.mean(axis=0)
    cols = np.where(np.logical_or(col_ratio > 0.46, signal_col_ratio > 0.34))[0]
    if not cols.size:
        cols = np.where(np.logical_or(col_ratio > 0.34, signal_col_ratio > 0.24))[0]
    if cols.size:
        left = int(cols.min())
        right = int(cols.max())
        pad = max(1, int(round((right - left + 1) * 0.035)))
        new_x0 = x0 + max(0, left - pad)
        new_x1 = min(width, x0 + right + 1 + pad)
        x0, x1 = new_x0, new_x1

    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def warp_quad_to_panel(image, quad):
    box = order_points(np.asarray(quad, dtype="float32"))
    tl, tr, br, bl = box
    out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if out_w < 8 or out_h < 8:
        return None
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    # Keep the complete quadrilateral here. Trimming at this stage can mistake
    # a bright leading digit (for example the 0 in 0123) for the outside white
    # margin and remove it before the later, dedicated panel cleanup stage.
    return normalize_panel_size(warped)


def find_black_rect_edge_quad(image, min_ratio=1.35, max_ratio=7.5, max_area_ratio=0.95):
    """Find a black-panel quadrilateral inside a roughly cropped ROI.

    Canny is useful for visible borders, but a black panel can be slightly
    blurred or partly low-contrast. Also generate quadrilateral candidates from
    the dark foreground mask, then score all candidates by geometry and whether
    they enclose a dark long panel with bright digit strokes.
    """
    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray_raw.size == 0:
        return None
    gray_eq = local_equalized_gray(image)
    gray = np.maximum(gray_raw, gray_eq)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 5), np.uint8))
    height, width = gray_raw.shape
    candidates = []

    contour_groups = []
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_groups.append(("edge", edge_contours))
    for threshold in (85, 105, 125):
        dark = (gray < threshold).astype("uint8")
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dark_contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_groups.append((f"dark{threshold}", dark_contours))

    for source, contours in contour_groups:
        for contour in contours:
            if cv2.contourArea(contour) < width * height * 0.012:
                continue
            rect = cv2.minAreaRect(contour)
            box = order_points(cv2.boxPoints(rect))
            tl, tr, br, bl = box
            rect_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
            rect_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            if rect_w < width * 0.30 or rect_h < height * 0.22:
                continue
            ratio = rect_w / float(max(1.0, rect_h))
            if not (min_ratio <= ratio <= max_ratio):
                continue

            mask = np.zeros_like(gray_raw, dtype=np.uint8)
            cv2.fillConvexPoly(mask, box.astype("int32"), 255)
            inside = gray_raw[mask > 0]
            if inside.size == 0:
                continue
            dark_ratio = float((inside < 120).mean())
            bright_ratio = float((inside > 135).mean())
            if dark_ratio < 0.38:
                continue
            area = cv2.contourArea(box.astype("float32"))
            area_ratio = area / float(max(1, width * height))
            if not (0.018 <= area_ratio <= max_area_ratio):
                continue
            # Strongly prefer a visible quadrilateral that encloses a dark, long
            # region but still has some bright digit strokes.
            source_bonus = 1.08 if source == "edge" else 1.0
            score = area * source_bonus * min(1.0, ratio / 2.3) * (0.75 * dark_ratio + 0.25 * bright_ratio)
            candidates.append((score, box))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def rectify_panel_from_region(search_roi, x, y, w, h):
    """Rectify the selected black panel region into a true rectangle."""
    gray_search = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY) if search_roi.ndim == 3 else search_roi
    x, y, w, h = tighten_black_rect_bounds(gray_search, x, y, w, h, threshold=115)
    height, width = search_roi.shape[:2]
    # Keep a real margin around the approximate dark contour. White digits can
    # split the dark mask, especially at the left edge for labels such as 67 or
    # 0123, so a tiny padding can make the perspective refinement cut through
    # the first digit.
    pad_x = max(2, int(round(w * 0.10)))
    pad_y = max(1, int(round(h * 0.05)))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)

    region = search_roi[y0:y1, x0:x1]
    raw_panel = normalize_panel_size(region)
    trimmed_panel = normalize_panel_size(trim_panel_whitespace(region))
    raw_leading = _leading_digit_signal_score(raw_panel)
    trimmed_leading = _leading_digit_signal_score(trimmed_panel)
    direct_panel = (
        raw_panel
        if raw_leading >= max(0.012, trimmed_leading * 0.85)
        else trimmed_panel
    )
    edge_box = find_black_rect_edge_quad(region)
    if edge_box is not None:
        edge_panel = warp_quad_to_panel(region, edge_box)
        if edge_panel is not None:
            edge_rect = _black_rect_score(edge_panel)
            direct_rect = _black_rect_score(direct_panel)
            edge_strokes = _digit_stroke_score(edge_panel)
            direct_strokes = _digit_stroke_score(direct_panel)
            edge_layout = digit_stroke_layout_score(edge_panel)
            direct_layout = digit_stroke_layout_score(direct_panel)
            edge_leading = _leading_digit_signal_score(edge_panel)
            direct_leading = _leading_digit_signal_score(direct_panel)
            if (
                edge_rect >= direct_rect * 0.82
                # Perspective rectification is allowed to improve geometry,
                # but it must not discard a substantial part of the text that
                # is already present in the direct ROI crop.
                and edge_strokes >= direct_strokes * 0.82
                and edge_layout >= direct_layout * 0.88
                and edge_leading >= max(0.012, direct_leading * 0.85)
            ):
                return edge_panel, (x0, y0, x1 - x0, y1 - y0), True

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    dark = (gray < 115).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 5), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return direct_panel, (x0, y0, x1 - x0, y1 - y0), False

    contour = max(contours, key=cv2.contourArea)
    box = order_points(cv2.boxPoints(cv2.minAreaRect(contour)))
    warped = warp_quad_to_panel(region, box)
    if warped is None:
        return direct_panel, (x0, y0, x1 - x0, y1 - y0), False

    # Prefer the contour-based perspective warp when it gives a cleaner black
    # rectangle. Only fall back to the axis-aligned crop if the warp clearly
    # loses digit strokes or no longer looks like a black long rectangle.
    direct_rect = _black_rect_score(direct_panel)
    warped_rect = _black_rect_score(warped)
    direct_strokes = _digit_stroke_score(direct_panel)
    warped_strokes = _digit_stroke_score(warped)
    if warped_rect < direct_rect * 0.88 or warped_strokes < direct_strokes * 0.58:
        return direct_panel, (x0, y0, x1 - x0, y1 - y0), True
    return warped, (x0, y0, x1 - x0, y1 - y0), True


def _bright_component_stats(roi):
    """Describe bright structures inside a black-panel candidate.

    Real digits form one horizontal text band. QR codes often contain several
    large near-square white holes, and background/shadow regions contain one
    giant irregular bright area.
    """
    bright = (roi > 135).astype("uint8")
    n_components, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    components = []
    for index in range(1, n_components):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        components.append((area, width, height, x, y))
    return components


def _looks_like_qr_or_background(roi) -> bool:
    height, width = roi.shape
    ratio = width / float(max(1, height))
    dark_ratio = float((roi < 105).mean())
    row_dark_ratio = (roi < 105).mean(axis=1)
    col_dark_ratio = (roi < 105).mean(axis=0)
    continuous_dark_cols = float((col_dark_ratio > 0.45).mean())
    continuous_dark_rows = float((row_dark_ratio > 0.45).mean())

    # A valid digit panel can contain many white digit strokes, especially for
    # labels such as 304/8088. If the black background is still a continuous
    # horizontal rectangle, do not reject it as QR/background just because the
    # white foreground is prominent.
    if ratio >= 1.45 and dark_ratio >= 0.50 and continuous_dark_cols >= 0.55 and continuous_dark_rows >= 0.55:
        return False

    # A true number block is a horizontal black strip. Even if it contains
    # digits like 0/8/9 with large white holes, the whole bright structure is
    # still arranged as one row. Do not let the QR-code guard reject these.
    if ratio >= 2.2:
        return False

    area = float(max(1, height * width))
    bright_ratio = float((roi > 135).mean())
    components = _bright_component_stats(roi)
    large_square_components = 0
    giant_components = 0
    for component_area, component_width, component_height, _, _ in components:
        aspect = component_width / float(max(1, component_height))
        area_ratio = component_area / area
        if (
            area_ratio > 0.035
            and 0.65 <= aspect <= 1.55
            and component_width > width * 0.12
            and component_height > height * 0.18
        ):
            large_square_components += 1
        if area_ratio > 0.25:
            giant_components += 1

    if bright_ratio > 0.22 and (large_square_components >= 2 or giant_components):
        return True

    bright = roi > 135
    ys, xs = np.where(bright)
    if xs.size == 0:
        return False
    bright_height_ratio = (int(ys.max()) - int(ys.min()) + 1) / float(max(1, height))
    bright_width_ratio = (int(xs.max()) - int(xs.min()) + 1) / float(max(1, width))
    # QR codes spread bright holes over most of the candidate height. Digits are
    # a horizontal text band on the left side of a long black rectangle.
    return bright_ratio > 0.24 and bright_height_ratio > 0.78 and bright_width_ratio > 0.55


def find_black_digit_panel(search_roi, digit_count: int | None = None):
    """Find the physical black number rectangle inside a wider label ROI.

    The wide ROI only says "the digit block should be somewhere here". The
    returned panel should be the actual black rectangle, constrained by shape,
    darkness, and the presence of bright digit strokes.
    """
    gray_raw = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY) if search_roi.ndim == 3 else search_roi
    gray_balanced = balanced_gray(search_roi)
    gray_eq = local_equalized_gray(search_roi)
    # Use the equalized image to make shadows less destructive for threshold
    # decisions, but crop the final panel from the original RGB search_roi.
    # Use the balanced gray image for the dark-mask geometry. Taking the
    # maximum with CLAHE makes overexposed black panels look artificially gray
    # and is exactly what caused otherwise valid panels to fail the dark-ratio
    # gate. CLAHE remains available for edge discovery and debug visualization.
    gray = np.minimum(gray_balanced, gray_eq)
    height, width = gray.shape

    dark = (gray < 105).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    panel_candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # The physical panel occupies a substantial horizontal part of this
        # ROI. This rejects small QR patches before they can win by contrast.
        if w < width * 0.38 or h < height * 0.18:
            continue
        ratio = w / float(max(1, h))
        if not (1.35 <= ratio <= 4.8):
            continue
        area_ratio = area / float(max(1, width * height))
        if not (0.055 <= area_ratio <= 0.42):
            continue
        center_x = (x + w * 0.5) / float(max(1, width))
        center_y = (y + h * 0.5) / float(max(1, height))
        # The number panel is the lower-left black rectangle. This rejects
        # right-side QR/barcode areas that otherwise look like dark rectangles.
        if center_x > 0.76 or center_y < 0.20:
            continue

        roi = gray[y : y + h, x : x + w]
        dark_ratio = float((roi < 105).mean())
        bright_ratio = float((roi > 135).mean())
        if dark_ratio < 0.46:
            continue
        # A valid digit panel is not just a black blob; it should contain
        # white/gray digit strokes. Single-digit panels such as "0" have a
        # very small white area compared with the black padding, so allow a
        # lower stroke ratio when the black background is a strong rectangle.
        row_dark_ratio = (roi < 105).mean(axis=1)
        col_dark_ratio = (roi < 105).mean(axis=0)
        continuous_dark_cols = float((col_dark_ratio > 0.45).mean())
        continuous_dark_rows = float((row_dark_ratio > 0.45).mean())
        strong_black_panel = ratio >= 1.55 and dark_ratio > 0.62 and continuous_dark_rows > 0.50
        # Under shadow or low exposure, the white digit can be below the fixed
        # bright threshold, especially for a single "0". A strong horizontal
        # black panel in the expected ROI is still a useful candidate; let OCR
        # and quality checks decide later instead of rejecting it here.
        min_bright_ratio = 0.0 if strong_black_panel else 0.01
        if bright_ratio < min_bright_ratio:
            continue
        if _looks_like_qr_or_background(roi):
            continue

        # Barcodes have many alternating bright gaps; the real number panel has
        # a mostly continuous black background with white strokes on top.
        # White digits can break many black-background columns. Keep this
        # check strict for ambiguous square-ish blobs, but relax it for
        # horizontal black strips in the expected lower-left area.
        if ratio >= 1.70 and dark_ratio > 0.45:
            min_continuous_cols = 0.30
            min_continuous_rows = 0.25
        else:
            min_continuous_cols = 0.50 if ratio >= 2.35 else 0.62
            min_continuous_rows = 0.42
        if continuous_dark_cols < min_continuous_cols or continuous_dark_rows < min_continuous_rows:
            continue

        rectangularity = area / float(max(1, w * h))
        if ratio >= 1.45 and rectangularity < 0.48:
            continue
        width_prior = min(1.0, w / max(1.0, width * 0.42))
        left_prior = max(0.15, 1.0 - max(0.0, center_x - 0.48) * 1.8)
        lower_prior = min(1.0, max(0.2, (center_y - 0.18) * 1.8))
        continuity = 0.55 * continuous_dark_cols + 0.45 * continuous_dark_rows
        aspect_prior = min(1.0, max(0.2, (ratio - 1.45) / 0.9))
        area_prior = min(1.0, max(0.2, area_ratio / 0.16))
        edge_bonus = 0.82
        edge_region = search_roi[max(0, y - 2) : min(height, y + h + 2), max(0, x - 2) : min(width, x + w + 2)]
        edge_box = find_black_rect_edge_quad(edge_region)
        if edge_box is not None:
            abs_edge_box = edge_box.copy()
            abs_edge_box[:, 0] += max(0, x - 2)
            abs_edge_box[:, 1] += max(0, y - 2)
            edge_area = cv2.contourArea(edge_box.astype("float32")) / float(max(1, edge_region.shape[0] * edge_region.shape[1]))
            edge_w = max(
                np.linalg.norm(edge_box[1] - edge_box[0]),
                np.linalg.norm(edge_box[2] - edge_box[3]),
            )
            edge_h = max(
                np.linalg.norm(edge_box[3] - edge_box[0]),
                np.linalg.norm(edge_box[2] - edge_box[1]),
            )
            edge_ratio = edge_w / float(max(1.0, edge_h))
            if 1.35 <= edge_ratio <= 7.5 and edge_area >= 0.18:
                edge_bonus = 1.28
                edge_panel = warp_quad_to_panel(search_roi, abs_edge_box)
                direct_compare = normalize_panel_size(search_roi[y : y + h, x : x + w])
                leading_ok = edge_panel is not None and _leading_digit_signal_score(edge_panel, digit_count) >= max(
                    0.012,
                    _leading_digit_signal_score(direct_compare, digit_count) * 0.85,
                )
                if edge_panel is not None and leading_ok and is_plausible_number_panel(edge_panel, digit_count):
                    qx, qy, qw, qh = cv2.boundingRect(abs_edge_box.astype("float32"))
                    qx = max(0, qx)
                    qy = max(0, qy)
                    qw = min(width - qx, qw)
                    qh = min(height - qy, qh)
                    edge_score = (
                        area
                        * 1.42
                        * (1.0 + _black_rect_score(edge_panel))
                        * max(0.15, panel_selection_score(edge_panel, digit_count) + 1.0)
                    )
                    panel_candidates.append((edge_score, (qx, qy, qw, qh), edge_panel))
        score = (
            area
            * width_prior
            * left_prior
            * lower_prior
            * continuity
            * aspect_prior
            * area_prior
            * edge_bonus
            * (0.75 * dark_ratio + 0.25 * bright_ratio)
            * min(1.0, max(0.15, rectangularity) ** 1.8 + 0.18)
        )
        candidates.append((score, x, y, w, h))

    edge_quad = find_black_rect_edge_quad(search_roi, max_area_ratio=0.45)
    if edge_quad is not None:
        ex, ey, ew, eh = cv2.boundingRect(edge_quad.astype("float32"))
        ex = max(0, ex)
        ey = max(0, ey)
        ew = min(width - ex, ew)
        eh = min(height - ey, eh)
        if ew > 0 and eh > 0 and ew >= width * 0.38:
            center_x = (ex + ew * 0.5) / float(max(1, width))
            center_y = (ey + eh * 0.5) / float(max(1, height))
            area = max(1.0, cv2.contourArea(edge_quad.astype("float32")))
            area_ratio = area / float(max(1, width * height))
            edge_panel = warp_quad_to_panel(search_roi, edge_quad)
            if (
                edge_panel is not None
                and center_x <= 0.78
                and center_y >= 0.16
                and 0.030 <= area_ratio <= 0.48
                and is_plausible_number_panel(edge_panel, digit_count)
            ):
                edge_score = area * (1.0 + _black_rect_score(edge_panel)) * (0.75 + 0.25 * _digit_stroke_score(edge_panel))
                panel_candidates.append((edge_score * 1.35, (ex, ey, ew, eh), edge_panel))

    if not candidates and not panel_candidates:
        return None, (0, 0, width, height), False

    best_panel = None
    best_box = None
    best_score = None
    for raw_score, box, panel in sorted(panel_candidates, key=lambda item: item[0], reverse=True):
        if panel is None:
            continue
        if not is_plausible_number_panel(panel, digit_count):
            continue
        panel_score = raw_score * max(0.03, panel_selection_score(panel, digit_count) + 1.0)
        if best_score is None or panel_score > best_score:
            best_score = panel_score
            best_panel = panel
            best_box = box

    for raw_score, x, y, w, h in sorted(candidates, key=lambda item: item[0], reverse=True):
        panel, box, _ = rectify_panel_from_region(search_roi, x, y, w, h)
        if panel is None:
            continue
        if not is_plausible_number_panel(panel, digit_count):
            continue
        panel_score = (
            raw_score
            * max(0.15, _black_rect_score(panel))
            * max(0.03, panel_selection_score(panel, digit_count) + 1.0)
        )
        if best_score is None or panel_score > best_score:
            best_score = panel_score
            best_panel = panel
            best_box = box

    if best_panel is not None and best_box is not None:
        return best_panel, best_box, True

    if not candidates:
        return None, (0, 0, width, height), False

    _, x, y, w, h = max(candidates, key=lambda item: item[0])

    # Refine inside the selected contour only. Keep the whole physical black
    # rectangle; do not choose the single darkest horizontal band, because
    # bright digit strokes can split the rectangle and make only the top/bottom
    # half look like the "largest black band".
    local_dark = gray[y : y + h, x : x + w] < 112
    local_dark = cv2.morphologyEx(local_dark.astype("uint8"), cv2.MORPH_CLOSE, np.ones((11, 5), np.uint8)).astype(bool)
    local_row_ratio = local_dark.mean(axis=1)
    local_rows = np.where(local_row_ratio > 0.18)[0]
    if local_rows.size:
        ry0 = int(local_rows.min())
        ry1 = int(local_rows.max())
        y += ry0
        h = ry1 - ry0 + 1

    local_dark = gray[y : y + h, x : x + w] < 105
    local_col_ratio = local_dark.mean(axis=0)
    # Use both black background and white digit strokes when refining columns.
    # In overexposed samples the left edge of the black panel can be washed out,
    # but the white digit strokes still indicate that the panel started earlier.
    local_bright = gray[y : y + h, x : x + w] > 135
    local_signal = np.logical_or(local_dark, local_bright)
    local_col_ratio = local_signal.mean(axis=0)
    local_cols = np.where(local_col_ratio > 0.14)[0]
    if local_cols.size:
        cx0 = int(local_cols.min())
        cx1 = int(local_cols.max()) + 1
        x += cx0
        w = cx1 - cx0

    panel, box, _ = rectify_panel_from_region(search_roi, x, y, w, h)
    return panel, box, True


def _panel_profile(panel):
    height, width = panel.shape[:2]
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    y0 = max(0, int(round(height * 0.16)))
    y1 = min(height, int(round(height * 0.88)))
    core = gray[y0:y1]

    threshold = max(95, min(145, float(np.percentile(core, 86))))
    bright = core > threshold
    col = bright.sum(axis=0).astype(np.float32)
    smooth_width = max(3, int(round(width * 0.018)))
    kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
    smooth = np.convolve(col, kernel, mode="same")
    active = smooth > max(1.2, core.shape[0] * 0.035)
    active_indices = np.where(active)[0]
    return gray, core, smooth, active_indices


def _estimate_text_bounds(panel, digit_count: int):
    height, width = panel.shape[:2]
    _, core, smooth, active_indices = _panel_profile(panel)
    if active_indices.size:
        x0 = max(0, int(active_indices.min()) - max(1, int(round(width * 0.015))))
        x1 = min(width, int(active_indices.max()) + 1 + max(2, int(round(width * 0.025))))
        if x0 <= width * 0.08:
            # If a little white label border leaked into the left side, the
            # projection rises at the very edge and then drops to black. In that
            # case the text itself still starts near the panel edge; keep a
            # small black margin but do not trust the physical crop x=0 blindly.
            x0 = max(0, int(active_indices.min()) - max(1, int(round(width * 0.03))))
    else:
        x0 = 0
        x1 = width

    return x0, x1, smooth, core.shape[0]


def _projection_runs(smooth, core_height: int, width: int, digit_count: int):
    """Find digit stroke runs from the white-pixel column projection."""
    for fraction in (0.16, 0.12, 0.09, 0.06):
        threshold = max(1.5, core_height * fraction)
        mask = smooth > threshold
        xs = np.where(mask)[0]
        if not xs.size:
            continue
        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[xs[0], xs[breaks + 1]]
        ends = np.r_[xs[breaks], xs[-1]]
        runs = []
        for start, end in zip(starts, ends):
            start = int(start)
            end = int(end)
            run_width = end - start + 1
            peak = float(smooth[start : end + 1].max())
            # Ignore tiny edge artifacts from panel borders or leaked label
            # text. Real digits have a visible width or a strong peak.
            touches_edge = start <= 2 or end >= width - 3
            if touches_edge and run_width <= max(3, int(width * 0.035)):
                continue
            if run_width < max(2, int(width * 0.025)) and peak < core_height * 0.18:
                continue
            runs.append((start, end, peak))
        if len(runs) >= digit_count:
            return runs[:digit_count]
    return []


def _scaled_slot_cuts(width: int, digit_count: int):
    scale = width / float(CANONICAL_PANEL_SIZE[0])
    return [int(round(cut * scale)) for cut in CANONICAL_SLOT_CUTS[: max(0, digit_count - 1)]]


def _right_edge_from_projection(width: int, runs, digit_count: int):
    if digit_count in CANONICAL_TEXT_RIGHT:
        scale = width / float(CANONICAL_PANEL_SIZE[0])
        return min(width, int(round(CANONICAL_TEXT_RIGHT[digit_count] * scale)))
    if len(runs) >= digit_count:
        first_start = runs[0][0]
        last_end = runs[digit_count - 1][1]
        nominal_width = max(1, last_end - first_start + 1)
        pad = max(3, int(round(nominal_width / max(1, digit_count) * 0.22)))
        return min(width, last_end + 1 + pad)
    if digit_count < 4:
        scale = width / float(CANONICAL_PANEL_SIZE[0])
        return min(width, int(round((CANONICAL_SLOT_CUTS[digit_count - 1] + 8) * scale)))
    return width


def _slot_split_score(panel, digit_count: int):
    width = panel.shape[1]
    x0, x1, smooth, core_height = _estimate_text_bounds(panel, digit_count)
    text_ratio = (x1 - x0) / max(1.0, float(width))
    slot_ratio = text_ratio / max(1, digit_count)

    expected_text = min(1.0, digit_count / 4.0)
    score = 1.0 - abs(text_ratio - expected_text)

    if digit_count < 4 and text_ratio > 0.9:
        score -= 0.8
    if not (0.20 <= slot_ratio <= 0.42):
        score -= 0.6

    slot_width = (x1 - x0) / max(1, digit_count)
    for i in range(digit_count):
        lo = int(round(x0 + slot_width * i))
        hi = int(round(x0 + slot_width * (i + 1)))
        if hi <= lo:
            score -= 0.5
            continue
        peak = float(smooth[lo:hi].max()) if hi > lo else 0.0
        if peak < max(2.0, core_height * 0.10):
            score -= 0.25
    return score


def compute_panel_split(panel, digit_count: int):
    """Compute digit crop boundaries for a rectified black digit panel.

    The projection is the same signal shown in the debug image. Use it to find
    where white digit strokes start/end and where the low valleys between
    digits are. Keep black padding around digits; do not crop tightly to strokes.
    """
    height, width = panel.shape[:2]
    text_x0, text_x1, smooth, core_height = _estimate_text_bounds(panel, digit_count)
    runs = _projection_runs(smooth, core_height, width, digit_count)
    valley_threshold = max(1.2, core_height * 0.04)

    slot_priors = _scaled_slot_cuts(width, digit_count)
    if slot_priors:
        cuts = [0]
        for index, prior in enumerate(slot_priors):
            lo = max(cuts[-1] + 3, prior - max(5, int(round(width * 0.045))))
            hi = min(width - 3, prior + max(5, int(round(width * 0.045))))
            if hi >= lo:
                candidates = np.arange(lo, hi + 1)
                valley_candidates = candidates[smooth[candidates] <= valley_threshold]
                if valley_candidates.size:
                    candidates = valley_candidates
                distance_penalty = np.abs(candidates - prior) * max(0.05, core_height * 0.006)
                cut = int(candidates[np.argmin(smooth[candidates] + distance_penalty)])
            else:
                cut = int(prior)
            cuts.append(cut)
        right_edge = _right_edge_from_projection(width, runs, digit_count)
        right_edge = max(right_edge, cuts[-1] + max(4, int(round(width * 0.04))))
        cuts.append(min(width, right_edge))
    else:
        text_width = max(1, text_x1 - text_x0)
        cuts = [text_x0]
        for i in range(1, digit_count):
            expected = text_x0 + text_width * i / float(digit_count)
            search_radius = max(3, int(round(text_width / digit_count * 0.18)))
            lo = max(cuts[-1] + 3, int(round(expected - search_radius)))
            hi = min(text_x1 - 3, int(round(expected + search_radius)))
            if hi <= lo:
                cut = int(round(expected))
            else:
                candidates = np.arange(lo, hi + 1)
                distance_penalty = np.abs(candidates - expected) * max(0.08, core_height * 0.01)
                cut = int(candidates[np.argmin(smooth[candidates] + distance_penalty)])
            cuts.append(cut)
        cuts.append(text_x1)

    crop_y0 = max(0, int(round(height * 0.03)))
    crop_y1 = min(height, int(round(height * 0.97)))
    return cuts, crop_y0, crop_y1, runs, slot_priors, smooth, core_height


def split_panel_evenly(panel, digit_count: int):
    """Split a rectified black digit panel using slot priors and projection."""
    height, width = panel.shape[:2]
    cuts, crop_y0, crop_y1, *_ = compute_panel_split(panel, digit_count)

    crops = []
    for i in range(digit_count):
        x0 = max(0, cuts[i])
        x1 = min(width, cuts[i + 1])
        crops.append(panel[crop_y0:crop_y1, x0:x1])
    return crops


def extract_digit_images(path: Path, image, label: str, min_blur: float, min_label_quality: float):
    preprocessed = preprocess_label_image(
        path,
        image,
        min_blur=min_blur,
        min_label_quality=min_label_quality,
    )
    if preprocessed["error"] is not None:
        return None, preprocessed["error"]["code"], None

    warped = preprocessed["warped"]
    best = None
    for rotation in (0, 90, 180, 270):
        oriented = rotate_warp(warped, rotation)
        search_roi, _ = crop_bin_roi_from_warp(oriented, WIDE_DIGIT_ROI_RATIOS)
        panel, _, found = find_black_digit_panel(search_roi, digit_count=len(label))
        if not found:
            continue
        panel_quality_code = assess_digit_panel_quality(panel)
        if panel_quality_code is not None:
            candidate = (_slot_split_score(panel, len(label)) - 0.05, rotation, None, panel_quality_code)
            if best is None or candidate[0] > best[0]:
                best = candidate
            continue
        digit_images = split_panel_evenly(panel, len(label))
        score = _slot_split_score(panel, len(label))
        candidate = (score, rotation, digit_images, "OK")
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None, "ERR_BLACK_PANEL_NOT_FOUND", None
    _, rotation, digit_images, code = best
    if code != "OK":
        return None, code, rotation
    return digit_images, "OK", rotation


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract single-digit CNN training crops from full label images.")
    parser.add_argument("inputs", nargs="+", help="Image files or folders.")
    parser.add_argument("--label", required=True, help="Ground-truth number printed on these labels, e.g. 89.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder with 0/..9/ subfolders.")
    parser.add_argument("--min-blur", type=float, default=35.0, help="Same conservative blur gate as recognizer.")
    parser.add_argument(
        "--min-label-quality",
        type=float,
        default=0.55,
        help="Same conservative label geometry gate as recognizer.",
    )
    parser.add_argument("--prefix", default="", help="Optional prefix for saved crop file names.")
    args = parser.parse_args()

    if not args.label.isdigit():
        raise SystemExit("--label must contain digits only.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for digit in "0123456789":
        (args.output_dir / digit).mkdir(exist_ok=True)

    saved = 0
    failed = 0
    for path in collect_images(args.inputs):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"{path}\tERR_IMAGE_READ")
            failed += 1
            continue

        digit_images, code, rotation = extract_digit_images(
            path,
            image,
            args.label,
            args.min_blur,
            args.min_label_quality,
        )
        if code != "OK" or digit_images is None:
            print(f"{path}\t{code}\trotation={rotation}")
            failed += 1
            continue

        for index, (digit, crop) in enumerate(zip(args.label, digit_images)):
            out_name = f"{args.prefix}{path.stem}_{index}_{digit}.png"
            cv2.imwrite(str(args.output_dir / digit / out_name), crop)
            saved += 1
        print(f"{path}\tOK\trotation={rotation}\tdigits={len(digit_images)}")

    print(f"saved_digit_crops={saved} failed_images={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
