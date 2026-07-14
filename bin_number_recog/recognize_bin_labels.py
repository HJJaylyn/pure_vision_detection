#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from cnn_digit_model import load_digit_cnn, predict_digit_sequence


DEFAULT_WARP_SIZE = (420, 420)
DEFAULT_ROI_RATIOS = (0.03, 0.61, 0.38, 0.83)
DEFAULT_MIN_DIGITS = 1
DEFAULT_MAX_DIGITS = 8


ERROR_MESSAGES = {
    "OK": "recognized",
    "ERR_IMAGE_READ": "cannot read image",
    "ERR_BLUR": "image is too blurry; retake photo",
    "ERR_LABEL_INCOMPLETE": "label is clipped by image boundary; realign and retake",
    "ERR_LABEL_NOT_FOUND": "label quadrilateral was not found; realign and retake",
    "ERR_LABEL_GEOMETRY": "label perspective/shape is too distorted; realign and retake",
    "ERR_BLACK_PANEL_NOT_FOUND": "expected black digit panel was not found inside ROI",
    "ERR_PANEL_GLARE": "black digit panel has glare or overexposure; retake photo",
    "ERR_PANEL_LOW_CONTRAST": "black digit panel contrast is too low; retake photo",
    "ERR_DIGITS_NOT_FOUND": "black panel was found but no usable digits were detected",
    "ERR_DIGIT_LENGTH": "recognized digit count is outside the expected range",
    "ERR_LOW_CONFIDENCE": "recognition confidence is too low",
    "ERR_OCR_MISSING": "requested OCR engine is not installed",
    "ERR_CNN_MISSING": "CNN checkpoint was requested but not provided or could not be loaded",
}


def build_result(
    path,
    code,
    value="",
    confidence=0.0,
    source="none",
    recognizer="none",
    box="",
    blur=0.0,
    label_quality=0.0,
    rotation="",
):
    return {
        "file": path.name,
        "ok": code == "OK",
        "code": code,
        "message": ERROR_MESSAGES[code],
        "value": value,
        "confidence": round(float(confidence), 3),
        "source": source,
        "recognizer": recognizer,
        "box": box,
        "blur": round(float(blur), 1),
        "label_quality": round(float(label_quality), 3),
        "rotation": rotation,
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


def local_equalized_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def label_border_and_inner_score(image, box, size=160):
    """Score whether a candidate looks like the printed label, not the table.

    The label is primarily a white square face, often with a pink or black
    border. The border can be incomplete, so this is a score instead of a hard
    requirement, but the inner white face is required.
    """
    ordered = order_points(box)
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    gray_raw = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = np.maximum(gray_raw, local_equalized_gray(warped))

    border = max(10, int(round(size * 0.10)))
    ring_mask = np.zeros((size, size), dtype=bool)
    ring_mask[:border, :] = True
    ring_mask[-border:, :] = True
    ring_mask[:, :border] = True
    ring_mask[:, -border:] = True

    inner_margin = max(border, int(round(size * 0.16)))
    inner = np.zeros((size, size), dtype=bool)
    inner[inner_margin : size - inner_margin, inner_margin : size - inner_margin] = True

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    inner_white = ((gray > 128) & (sat < 170))[inner].mean()
    inner_green = ((hue > 35) & (hue < 95) & (sat > 45) & (val > 70))[inner].mean()

    pink_border = (((hue < 12) | (hue > 160)) & (sat > 35) & (val > 90))[ring_mask].mean()
    dark_border = (gray_raw < 95)[ring_mask].mean()
    edges = cv2.Canny(gray_raw, 60, 150)
    edge_border = (edges > 0)[ring_mask].mean()

    # The true label face is mostly white after rectification. A green tabletop
    # patch may pass the coarse bright/low-saturation mask under glare, but it
    # will not have enough white face pixels and should be rejected.
    if inner_white < 0.24 or inner_green > 0.20:
        return 0.0, inner_white, pink_border + dark_border + edge_border

    border_score = min(1.0, pink_border * 6.0 + dark_border * 4.0 + edge_border * 9.0)
    inner_score = min(1.0, inner_white / 0.62)
    return 0.55 * inner_score + 0.45 * border_score, inner_white, border_score


def _label_candidate_score(image, box, base_score, source_boost=1.0):
    height, width = image.shape[:2]
    ordered = order_points(box)
    xs = ordered[:, 0]
    ys = ordered[:, 1]
    touches_edge = (
        xs.min() <= max(4, width * 0.015)
        or ys.min() <= max(4, height * 0.015)
        or xs.max() >= width - max(4, width * 0.015)
        or ys.max() >= height - max(4, height * 0.015)
    )
    label_score, inner_white, border_score = label_border_and_inner_score(image, ordered)
    if label_score <= 0.0:
        return None
    # A larger contour can wrap the real label plus surrounding table/objects.
    # It may still have border edges, but after warping its center is no longer
    # mostly white label face. Penalize that strongly so the tighter label
    # quadrilateral beats the outer wrapper.
    if inner_white < 0.34:
        return None
    inner_white_prior = min(1.0, max(0.05, inner_white / 0.78))
    inner_white_prior = inner_white_prior * inner_white_prior * inner_white_prior

    edge_penalty = 0.08 if touches_edge else 1.0
    geometry = label_geometry_quality(ordered)
    if geometry < 0.25:
        return None

    return (
        float(base_score)
        * edge_penalty
        * source_boost
        * (0.30 + label_score)
        * (0.70 + border_score * 0.55)
        * (0.55 + geometry * 0.45)
        * (0.20 + 0.80 * inner_white_prior)
    )


def expand_quad_about_center(quad, scale):
    quad = order_points(quad)
    center = quad.mean(axis=0, keepdims=True)
    return (center + (quad - center) * float(scale)).astype("float32")


def recover_outer_label_frame(image, quad):
    """Expand an inner white-face quad to include the outer black/pink frame.

    Edge detection can lock onto the white printable face instead of the true
    outer label frame. Try small outward expansions and keep one that still has
    a usable white face but gains stronger border evidence.
    """
    if quad is None:
        return quad
    current = order_points(quad)
    current_score, current_inner, current_border = label_border_and_inner_score(image, current)
    current_geom = label_geometry_quality(current)
    best = (current_score * (0.75 + current_border * 0.25) * (0.65 + current_geom * 0.35), current, 1.0)

    height, width = image.shape[:2]
    for scale in (1.06, 1.10, 1.14, 1.18, 1.22):
        expanded = expand_quad_about_center(current, scale)
        # Do not invent a frame far outside the image. A small overshoot is
        # acceptable because perspective warp can sample border pixels.
        if (
            expanded[:, 0].min() < -width * 0.03
            or expanded[:, 1].min() < -height * 0.03
            or expanded[:, 0].max() > width * 1.03
            or expanded[:, 1].max() > height * 1.03
        ):
            continue
        score, inner_white, border_score = label_border_and_inner_score(image, expanded)
        geom = label_geometry_quality(expanded)
        if score <= 0.0 or geom < 0.45 or inner_white < max(0.38, current_inner * 0.55):
            continue
        # Prefer expansions that gain border evidence, but avoid expanding so
        # much that the white face collapses into a small part of the warp.
        expansion_gain = 1.0 + max(0.0, border_score - current_border) * 0.45 + (scale - 1.0) * 0.18
        candidate_score = score * (0.75 + border_score * 0.35) * (0.65 + geom * 0.35) * expansion_gain
        # If the current contour is the inner white face, a modest expansion is
        # usually the correct outer frame even when the score is similar. Keep
        # expanding while the white face remains usable and border evidence is
        # not worse.
        similar_or_better = candidate_score > best[0] * 0.90 and border_score >= current_border * 0.65
        clearly_better = candidate_score > best[0] * 1.03
        if clearly_better or (similar_or_better and scale > best[2]):
            best = (candidate_score, expanded, scale)
    return order_points(best[1])


def edge_label_candidate_boxes(gray, image_shape):
    """Find quadrilateral candidates from visible label border/edge lines."""
    height, width = image_shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 115)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    # Keep the close kernel modest. A large kernel can merge the label border
    # with the image boundary/table edges, hiding the real inner quadrilateral.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Use RETR_LIST, not RETR_EXTERNAL: when the outer scene edge becomes one
    # big contour, the actual label border can be an internal contour.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.arcLength(contour, True) < min(width, height) * 0.08:
            continue

        area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        ratio = long_side / short_side
        if not (0.75 <= ratio <= 2.10):
            continue
        if long_side > max(width, height) * 0.86 or short_side < 35:
            continue
        box_area = rw * rh
        if box_area < 1800:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            box = approx.reshape(4, 2).astype("float32")
        else:
            box = cv2.boxPoints(rect)

        box_pts = np.asarray(box, dtype="float32")
        touches_edge = (
            box_pts[:, 0].min() <= max(4, width * 0.01)
            or box_pts[:, 1].min() <= max(4, height * 0.01)
            or box_pts[:, 0].max() >= width - max(4, width * 0.01)
            or box_pts[:, 1].max() >= height - max(4, height * 0.01)
        )
        if touches_edge and box_area > width * height * 0.20:
            continue

        # Edge contours are often just the border line, so contour fill is not
        # expected to be high. Use edge coverage around the fitted rectangle as
        # the base score instead.
        x, y, w, h = cv2.boundingRect(box_pts.astype(np.int32))
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        edge_density = float((edges[y0:y1, x0:x1] > 0).mean())
        base_score = box_area * (0.35 + min(1.0, edge_density * 8.0))
        boxes.append((base_score, box))
    return boxes


def find_label_quad(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq = local_equalized_gray(image)
    gray = np.maximum(gray_raw, gray_eq)

    # Use local contrast only as detection assistance. The final warp still
    # samples from the original RGB image.
    bright = ((gray > 125) & (hsv[:, :, 1] < 175)).astype("uint8") * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    candidates = []

    for base_score, box in edge_label_candidate_boxes(gray_raw, image.shape):
        score = _label_candidate_score(image, box, base_score=base_score, source_boost=0.85)
        if score is not None:
            candidates.append((score, box))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2500:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        ratio = long_side / short_side
        if not (1.0 <= ratio <= 1.75):
            continue
        if long_side > max(width, height) * 0.8 or short_side < 45:
            continue

        box_area = rw * rh
        fill = area / box_area
        if fill < 0.45:
            continue

        box = cv2.boxPoints(rect)
        score = _label_candidate_score(image, box, base_score=area * fill, source_boost=1.20)
        if score is not None:
            candidates.append((score, box))

    if not candidates:
        return None

    _, box = max(candidates, key=lambda item: item[0])
    return recover_outer_label_frame(image, order_points(box))


def has_incomplete_label_candidate(image, edge_margin_ratio=0.02):
    """Detect bright label-like regions clipped by the image boundary."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright = ((gray > 130) & (hsv[:, :, 1] < 165)).astype("uint8") * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    margin = max(3, int(min(width, height) * edge_margin_ratio))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        touches_edge = x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin
        if not touches_edge:
            continue
        ratio = max(w, h) / float(max(1, min(w, h)))
        if not (0.85 <= ratio <= 2.2):
            continue

        # A clipped label should contain part of the pink/red border. Plain
        # white cables, the robot board, or specular highlights near the image
        # edge are bright and label-sized, but they should not become
        # ERR_LABEL_INCOMPLETE.
        patch_hsv = hsv[y : y + h, x : x + w]
        patch_gray = gray[y : y + h, x : x + w]
        hue = patch_hsv[:, :, 0]
        sat = patch_hsv[:, :, 1]
        val = patch_hsv[:, :, 2]
        pink_border = (((hue < 12) | (hue > 160)) & (sat > 35) & (val > 95)).mean()
        label_white = ((patch_gray > 135) & (sat < 165)).mean()
        if pink_border > 0.025 and label_white > 0.18:
            return True
    return False


def label_quad_touches_image_edge(quad, image_shape, edge_margin_ratio=0.025):
    if quad is None:
        return False
    height, width = image_shape[:2]
    margin = max(4, int(min(width, height) * edge_margin_ratio))
    xs = quad[:, 0]
    ys = quad[:, 1]
    return bool(xs.min() <= margin or ys.min() <= margin or xs.max() >= width - margin or ys.max() >= height - margin)


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def label_geometry_quality(quad):
    if quad is None:
        return 0.0

    tl, tr, br, bl = quad
    top = np.linalg.norm(tr - tl)
    right = np.linalg.norm(br - tr)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    if min(top, right, bottom, left) <= 1:
        return 0.0

    opposite_balance = min(top, bottom) / max(top, bottom)
    side_balance = min(left, right) / max(left, right)

    def corner_score(a, b, c):
        v1 = a - b
        v2 = c - b
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom <= 1e-6:
            return 0.0
        cos_angle = abs(float(np.dot(v1, v2) / denom))
        return max(0.0, 1.0 - cos_angle)

    angles = [
        corner_score(bl, tl, tr),
        corner_score(tl, tr, br),
        corner_score(tr, br, bl),
        corner_score(br, bl, tl),
    ]
    return float(min(opposite_balance, side_balance, min(angles)))


def find_inner_white_label_quad(warped):
    """Find the inner white label face after the outer label has been warped.

    The outer detector uses the colored/black border as useful geometry, but
    downstream ROI ratios should be measured on the white printable label area.
    This second pass removes the border and corrects any residual skew.
    """
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    gray_raw = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_eq = local_equalized_gray(warped)
    gray = np.maximum(gray_raw, gray_eq)
    height, width = gray.shape

    white = ((gray > 118) & (hsv[:, :, 1] < 155)).astype("uint8") * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * 0.25:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect
        if rw <= 1 or rh <= 1:
            continue

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        ratio = long_side / short_side
        if not (1.0 <= ratio <= 1.70):
            continue
        if long_side > max(width, height) * 0.99 or short_side < min(width, height) * 0.42:
            continue

        box_area = rw * rh
        fill = area / max(1.0, box_area)
        if fill < 0.42:
            continue

        box = cv2.boxPoints(rect)
        candidates.append((area * fill, box))

    if not candidates:
        return None

    _, box = max(candidates, key=lambda item: item[0])
    return order_points(box)


def inner_white_quad_touches_warp_edge(inner_quad, warp_shape, edge_margin_ratio=0.015):
    if inner_quad is None:
        return True
    height, width = warp_shape[:2]
    margin = max(3, int(min(width, height) * edge_margin_ratio))
    xs = inner_quad[:, 0]
    ys = inner_quad[:, 1]
    return bool(xs.min() <= margin or ys.min() <= margin or xs.max() >= width - margin or ys.max() >= height - margin)


def has_complete_inner_white_label(image, outer_quad, size=DEFAULT_WARP_SIZE):
    """Return whether the white label face is complete even if the outer border is not."""
    if outer_quad is None:
        return False
    width, height = size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(outer_quad, dst)
    outer_warped = cv2.warpPerspective(image, matrix, (width, height))
    inner_quad = find_inner_white_label_quad(outer_warped)
    return inner_quad is not None and not inner_white_quad_touches_warp_edge(inner_quad, outer_warped.shape)


def has_usable_inner_white_label(image, outer_quad, size=DEFAULT_WARP_SIZE):
    """Return whether the white printable label face is usable.

    This is looser than has_complete_inner_white_label: if the outer colored
    frame touches the image edge but the actual white printable face is still
    visible and square-like, the sample should remain usable.
    """
    if outer_quad is None:
        return False
    width, height = size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(outer_quad.astype("float32"), dst)
    outer_warped = cv2.warpPerspective(image, matrix, (width, height))
    inner_quad = find_inner_white_label_quad(outer_warped)
    if inner_quad is None:
        return False

    x, y, w, h = cv2.boundingRect(inner_quad.astype(np.int32))
    area_ratio = cv2.contourArea(inner_quad.astype(np.float32)) / float(width * height)
    ratio = w / float(max(1, h))
    if area_ratio < 0.24 or not (0.68 <= ratio <= 1.55):
        return False

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, inner_quad.astype(np.int32), 255)
    hsv = cv2.cvtColor(outer_warped, cv2.COLOR_BGR2HSV)
    gray = np.maximum(cv2.cvtColor(outer_warped, cv2.COLOR_BGR2GRAY), local_equalized_gray(outer_warped))
    inside = mask > 0
    white_ratio = ((gray > 118) & (hsv[:, :, 1] < 175))[inside].mean() if inside.any() else 0.0
    return bool(white_ratio > 0.42)


def outer_label_face_is_usable(image, outer_quad):
    """Return whether the selected outer quad already contains a usable label face.

    This catches cases where the colored/black frame is very close to the image
    edge, but the white printable label face is still clearly present.
    """
    if outer_quad is None:
        return False
    label_score, inner_white, border_score = label_border_and_inner_score(image, outer_quad)
    geometry = label_geometry_quality(outer_quad)
    return bool(label_score > 0.72 and inner_white > 0.62 and border_score > 0.45 and geometry > 0.70)


def warp_inner_white_label(warped, size=DEFAULT_WARP_SIZE):
    inner_quad = find_inner_white_label_quad(warped)
    if inner_quad is None:
        return warped

    width, height = size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(inner_quad, dst)
    return cv2.warpPerspective(warped, matrix, (width, height))


def warp_label(image, size=DEFAULT_WARP_SIZE):
    quad = find_label_quad(image)
    if quad is None:
        return None, None

    width, height = size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    warped = warp_inner_white_label(warped, size=size)
    return warped, quad


def crop_bin_roi_from_warp(warped, ratios=DEFAULT_ROI_RATIOS):
    height, width = warped.shape[:2]
    x0r, y0r, x1r, y1r = ratios
    x0 = int(width * x0r)
    y0 = int(height * y0r)
    x1 = int(width * x1r)
    y1 = int(height * y1r)
    return warped[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def rotate_warp(warped, rotation):
    if rotation == 0:
        return warped
    if rotation == 90:
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(warped, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation: {rotation}")


def refine_black_digit_panel(panel):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    dark = (gray < 95).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)

    candidates = []
    height, width = gray.shape
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        ratio = w / float(h)
        if area < width * height * 0.06:
            continue
        if not (1.2 <= ratio <= 4.8):
            continue
        if h < height * 0.25 or w < width * 0.18:
            continue
        candidates.append((area, x, y, w, h))

    if not candidates:
        return None, (0, 0, width, height), False

    _, x, y, w, h = max(candidates, key=lambda item: item[0])
    pad_x = max(2, int(w * 0.06))
    pad_y = max(1, int(h * 0.12))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return panel[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0), True


def find_digit_panel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    label_box = find_label_box(image)
    if label_box is not None:
        lx, ly, lw, lh = label_box
        # Current labels have the bin number in the lower-left part of the
        # printed label. This is more stable than searching all black blobs.
        x0 = lx + int(lw * 0.03)
        y0 = ly + int(lh * 0.58)
        x1 = lx + int(lw * 0.38)
        y1 = ly + int(lh * 0.82)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    dark = (gray < 75).astype("uint8")
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    candidates = []
    height, width = gray.shape

    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not (18 <= w <= width * 0.28 and 12 <= h <= height * 0.16):
            continue
        ratio = w / float(h)
        if not (1.25 <= ratio <= 3.8):
            continue
        if area < 180:
            continue

        roi = gray[y : y + h, x : x + w]
        white_ratio = (roi > 135).mean()
        # The target is a black rectangle with clearly visible white digits.
        if white_ratio < 0.08:
            continue
        score = area * white_ratio
        candidates.append((score, x, y, w, h))

    if not candidates:
        return None

    _, x, y, w, h = max(candidates)
    pad_x = max(2, int(w * 0.04))
    pad_y = max(2, int(h * 0.08))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def find_label_box(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # White label paper and yellow title strip are both bright; the green mat
    # is saturated, so requiring brightness is enough after contour cleanup.
    bright = ((gray > 135) & (hsv[:, :, 1] < 140)).astype("uint8")
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 2500:
            continue
        if w < 50 or h < 45 or w > width * 0.75 or h > height * 0.75:
            continue
        ratio = w / float(h)
        if not (0.75 <= ratio <= 1.65):
            continue
        candidates.append((area, x, y, w, h))

    if not candidates:
        return None

    _, x, y, w, h = max(candidates)
    return x, y, w, h


def make_templates(size=(32, 48)):
    templates = {}
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    for digit in "0123456789":
        items = []
        for font in fonts:
            for scale in (1.15, 1.3, 1.45):
                canvas = np.zeros((70, 54), np.uint8)
                (tw, th), baseline = cv2.getTextSize(digit, font, scale, 2)
                x = (canvas.shape[1] - tw) // 2
                y = (canvas.shape[0] + th) // 2
                cv2.putText(canvas, digit, (x, y), font, scale, 255, 2, cv2.LINE_AA)
                ys, xs = np.where(canvas > 0)
                crop = canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                items.append(normalize_digit(crop, size))
        templates[digit] = items
    return templates


def normalize_digit(binary, size=(32, 48)):
    binary = (binary > 0).astype("uint8") * 255
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((size[1], size[0]), np.uint8)

    crop = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target_w, target_h = size
    scale = min((target_w - 6) / crop.shape[1], (target_h - 6) / crop.shape[0])
    new_w = max(1, int(crop.shape[1] * scale))
    new_h = max(1, int(crop.shape[0] * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((target_h, target_w), np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    out[y : y + new_h, x : x + new_w] = resized
    return out


def prepare_digit_mask(panel):
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    white = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -3,
    )
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return white


def assess_digit_panel_quality(panel):
    """Reject panel crops that are visibly unsafe before OCR.

    The target panel should be mostly dark with a small amount of bright digit
    strokes. Strong reflection often turns large regions white/gray and can
    create false strokes, so it is better to ask for a retake than to guess.
    """
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    if gray.size == 0:
        return "ERR_DIGITS_NOT_FOUND"

    p05, p50, p95 = np.percentile(gray, [5, 50, 95])
    contrast = float(p95 - p05)
    dark_ratio = float((gray < 115).mean())
    very_bright_ratio = float((gray > 235).mean())

    # A valid black panel should keep most of its background dark. If too much
    # of the panel is very bright, reflection is likely covering the digits.
    if very_bright_ratio > 0.34 or dark_ratio < 0.38:
        return "ERR_PANEL_GLARE"

    # Low contrast panels usually come from blur, defocus, bad exposure, or
    # glossy reflection washing out the white digits against the black panel.
    if contrast < 45 or p50 > 150:
        return "ERR_PANEL_LOW_CONTRAST"

    return None


def split_digits(panel, max_digits=DEFAULT_MAX_DIGITS):
    white = prepare_digit_mask(panel)

    count, _, stats, _ = cv2.connectedComponentsWithStats((white > 0).astype("uint8"), 8)
    h, w = white.shape
    components = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < 28 or bh < h * 0.12 or bw < 3:
            continue
        if bw > w * 0.5 or bh > h * 0.92:
            continue
        components.append([int(x), int(y), int(x + bw), int(y + bh), int(area)])

    components = sorted(components, key=lambda item: item[0])
    groups = []
    merge_gap = max(4, int(w * 0.02))
    for comp in components:
        if not groups:
            groups.append(comp)
            continue
        prev = groups[-1]
        overlaps_x = comp[0] <= prev[2]
        close_x = comp[0] - prev[2] <= merge_gap
        similar_vertical_band = not (comp[1] > prev[3] or comp[3] < prev[1])
        if overlaps_x or (close_x and similar_vertical_band):
            prev[0] = min(prev[0], comp[0])
            prev[1] = min(prev[1], comp[1])
            prev[2] = max(prev[2], comp[2])
            prev[3] = max(prev[3], comp[3])
            prev[4] += comp[4]
        else:
            groups.append(comp)

    boxes = []
    for x0, y0, x1, y1, area in groups:
        bw = x1 - x0
        bh = y1 - y0
        if area < 50 or bh < h * 0.24 or bw < 6:
            continue
        if bw > w * 0.46 or bh > h * 0.92:
            continue
        pad_x = max(1, int(bw * 0.08))
        pad_y = max(1, int(bh * 0.08))
        nx0 = max(0, x0 - pad_x)
        ny0 = max(0, y0 - pad_y)
        nx1 = min(w, x1 + pad_x)
        ny1 = min(h, y1 + pad_y)
        boxes.append((nx0, ny0, nx1 - nx0, ny1 - ny0))

    boxes = sorted(boxes, key=lambda b: b[0])
    if len(boxes) > max_digits:
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_digits]
        boxes = sorted(boxes, key=lambda b: b[0])
    return [white[y : y + h, x : x + w] for x, y, w, h in boxes], boxes, white


def recognize_panel(panel, templates, max_digits=DEFAULT_MAX_DIGITS):
    digit_images, boxes, mask = split_digits(panel, max_digits=max_digits)
    result = []
    scores = []
    for digit_img in digit_images:
        norm = normalize_digit(digit_img)
        best_digit = "?"
        best_score = -1.0
        for digit, digit_templates in templates.items():
            for templ in digit_templates:
                score = cv2.matchTemplate(norm, templ, cv2.TM_CCOEFF_NORMED)[0, 0]
                if score > best_score:
                    best_score = float(score)
                    best_digit = digit
        result.append(best_digit)
        scores.append(best_score)
    return "".join(result), scores, boxes, mask


def recognize_panel_tesseract(panel, max_digits=DEFAULT_MAX_DIGITS):
    if not shutil.which("tesseract"):
        return None

    mask = prepare_digit_mask(panel)
    _, digit_boxes, _ = split_digits(panel, max_digits=max_digits)
    detected_digit_count = len(digit_boxes)
    ocr_image = cv2.bitwise_not(mask)
    padded = cv2.copyMakeBorder(ocr_image, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        cv2.imwrite(tmp.name, padded)
        cmd = [
            "tesseract",
            tmp.name,
            "stdout",
            "--oem",
            "1",
            "--psm",
            "8",
            "-c",
            "tessedit_char_whitelist=0123456789",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    text = re.sub(r"\D", "", result.stdout)
    if not text:
        return "", 0.0, mask

    if len(text) > max_digits:
        text = text[:max_digits]
    confidence = 0.9 if result.returncode == 0 and text else 0.45
    if detected_digit_count and len(text) != detected_digit_count:
        confidence = min(confidence, 0.35)
    return text, confidence, mask


def recognize_panel_cnn(panel, cnn_bundle, max_digits=DEFAULT_MAX_DIGITS):
    if cnn_bundle is None:
        return "", 0.0, prepare_digit_mask(panel)
    digit_images, _, mask = split_digits(panel, max_digits=max_digits)
    value, confidence, _ = predict_digit_sequence(digit_images, cnn_bundle)
    return value, confidence, mask


def recognize_panel_with_engine(panel, templates, engine="template", max_digits=DEFAULT_MAX_DIGITS, cnn_bundle=None):
    if engine == "cnn":
        value, confidence, mask = recognize_panel_cnn(panel, cnn_bundle, max_digits=max_digits)
        return value, confidence, "cnn" if cnn_bundle is not None else "cnn-missing", mask

    if engine in ("auto", "tesseract"):
        ocr_result = recognize_panel_tesseract(panel, max_digits=max_digits)
        if ocr_result is not None:
            value, confidence, mask = ocr_result
            return value, confidence, "tesseract", mask
        return "", 0.0, "tesseract-missing", prepare_digit_mask(panel)

    value, scores, _, mask = recognize_panel(panel, templates, max_digits=max_digits)
    confidence = min(scores) if scores else 0.0
    return value, confidence, "template", mask


def code_rank(code):
    ranks = {
        "OK": 100,
        "ERR_LOW_CONFIDENCE": 80,
        "ERR_DIGIT_LENGTH": 70,
        "ERR_DIGITS_NOT_FOUND": 60,
        "ERR_PANEL_LOW_CONTRAST": 58,
        "ERR_PANEL_GLARE": 58,
        "ERR_OCR_MISSING": 55,
        "ERR_CNN_MISSING": 55,
        "ERR_BLACK_PANEL_NOT_FOUND": 40,
        "ERR_LABEL_INCOMPLETE": 35,
    }
    return ranks.get(code, 0)


def evaluate_warp_orientation(
    warped,
    rotation,
    templates,
    engine,
    min_digits,
    max_digits,
    min_confidence,
    cnn_bundle=None,
):
    oriented = rotate_warp(warped, rotation)
    panel, panel_box = crop_bin_roi_from_warp(oriented)
    x, y, w, h = panel_box

    panel, inner_box, found_black_panel = refine_black_digit_panel(panel)
    if not found_black_panel:
        return {
            "code": "ERR_BLACK_PANEL_NOT_FOUND",
            "value": "",
            "confidence": 0.0,
            "recognizer": "none",
            "panel": None,
            "mask": np.zeros((1, 1), np.uint8),
            "oriented": oriented,
            "panel_box": panel_box,
            "inner_box": inner_box,
            "rotation": rotation,
            "box": f"{x},{y},{w},{h}",
        }

    panel_quality_code = assess_digit_panel_quality(panel)
    if panel_quality_code is not None:
        return {
            "code": panel_quality_code,
            "value": "",
            "confidence": 0.0,
            "recognizer": "none",
            "panel": panel,
            "mask": prepare_digit_mask(panel),
            "oriented": oriented,
            "panel_box": panel_box,
            "inner_box": inner_box,
            "rotation": rotation,
            "box": f"{x},{y},{w},{h}",
        }

    value, confidence, recognizer, mask = recognize_panel_with_engine(
        panel,
        templates,
        engine,
        max_digits=max_digits,
        cnn_bundle=cnn_bundle,
    )

    if recognizer == "cnn-missing":
        code = "ERR_CNN_MISSING"
    elif recognizer == "tesseract-missing":
        code = "ERR_OCR_MISSING"
    elif not value:
        code = "ERR_DIGITS_NOT_FOUND"
    elif len(value) < min_digits or len(value) > max_digits:
        code = "ERR_DIGIT_LENGTH"
    elif confidence < min_confidence:
        code = "ERR_LOW_CONFIDENCE"
    else:
        code = "OK"

    return {
        "code": code,
        "value": value,
        "confidence": confidence,
        "recognizer": recognizer,
        "panel": panel,
        "mask": mask,
        "oriented": oriented,
        "panel_box": panel_box,
        "inner_box": inner_box,
        "rotation": rotation,
        "box": f"{x},{y},{w},{h}",
    }


def select_best_candidate(candidates):
    return max(
        candidates,
        key=lambda item: (
            code_rank(item["code"]),
            item["confidence"],
            len(item["value"]),
            -item["rotation"],
        ),
    )


def preprocess_label_image(path, image, min_blur=35.0, min_label_quality=0.55):
    """Run the shared conservative label checks before panel recognition.

    This is intentionally shared by the production recognizer and by dataset
    preparation, so extracted CNN training crops are produced only from images
    that pass the same front-end quality gates used at inference time.
    """
    if image is None:
        return {"error": build_result(path, "ERR_IMAGE_READ")}

    blur = blur_score(image)
    if blur < min_blur:
        return {"error": build_result(path, "ERR_BLUR", blur=blur)}

    warped, quad = warp_label(image)
    if warped is None:
        if has_incomplete_label_candidate(image):
            return {"error": build_result(path, "ERR_LABEL_INCOMPLETE", blur=blur)}
        return {"error": build_result(path, "ERR_LABEL_NOT_FOUND", blur=blur)}

    if (
        label_quad_touches_image_edge(quad, image.shape)
        and not has_complete_inner_white_label(image, quad)
        and not has_usable_inner_white_label(image, quad)
        and not outer_label_face_is_usable(image, quad)
    ):
        return {
            "error": build_result(
                path,
                "ERR_LABEL_INCOMPLETE",
                blur=blur,
                label_quality=label_geometry_quality(quad),
            )
        }

    label_quality = label_geometry_quality(quad)
    if label_quality < min_label_quality:
        return {
            "error": build_result(
                path,
                "ERR_LABEL_GEOMETRY",
                blur=blur,
                label_quality=label_quality,
            )
        }

    return {
        "error": None,
        "warped": warped,
        "quad": quad,
        "blur": blur,
        "label_quality": label_quality,
    }


def process_image(
    path,
    templates,
    debug_dir=None,
    engine="template",
    min_digits=DEFAULT_MIN_DIGITS,
    max_digits=DEFAULT_MAX_DIGITS,
    min_confidence=0.55,
    min_blur=35.0,
    min_label_quality=0.55,
    cnn_bundle=None,
):
    image = cv2.imread(str(path))
    preprocessed = preprocess_label_image(path, image, min_blur=min_blur, min_label_quality=min_label_quality)
    if preprocessed["error"] is not None:
        return preprocessed["error"]

    warped = preprocessed["warped"]
    quad = preprocessed["quad"]
    blur = preprocessed["blur"]
    label_quality = preprocessed["label_quality"]

    source = "warp"
    candidates = [
        evaluate_warp_orientation(
            warped,
            rotation,
            templates,
            engine,
            min_digits,
            max_digits,
            min_confidence,
            cnn_bundle,
        )
        for rotation in (0, 90, 180, 270)
    ]
    best = select_best_candidate(candidates)

    if debug_dir:
        write_debug_images(path, debug_dir, image, quad, best, candidates)

    return build_result(
        path,
        best["code"],
        value=best["value"],
        confidence=best["confidence"],
        source=source,
        recognizer=best["recognizer"],
        box=best["box"],
        blur=blur,
        label_quality=label_quality,
        rotation=best["rotation"],
    )


def write_debug_images(
    path,
    debug_dir,
    image,
    quad,
    best,
    candidates,
):
    debug_dir.mkdir(parents=True, exist_ok=True)
    vis = image.copy()
    cv2.polylines(vis, [quad.astype("int32")], True, (0, 255, 0), 2)

    for candidate in candidates:
        suffix = f"rot{candidate['rotation']}"
        candidate_vis = draw_candidate_debug(candidate)
        cv2.imwrite(str(debug_dir / f"{path.stem}_{suffix}_warp.png"), candidate_vis)

    warped = best["oriented"]
    panel_box = best["panel_box"]
    inner_box = best["inner_box"]
    value = best["value"]
    mask = best["mask"]
    panel = best["panel"]

    x, y, w, h = panel_box
    warp_vis = warped.copy()
    cv2.rectangle(warp_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    ix, iy, iw, ih = inner_box
    cv2.rectangle(warp_vis, (x + ix, y + iy), (x + ix + iw, y + iy + ih), (255, 0, 0), 1)
    cv2.putText(warp_vis, value, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(str(debug_dir / f"{path.stem}_debug.png"), vis)
    cv2.imwrite(str(debug_dir / f"{path.stem}_warp.png"), warp_vis)
    if panel is not None:
        cv2.imwrite(str(debug_dir / f"{path.stem}_panel.png"), panel)
    cv2.imwrite(str(debug_dir / f"{path.stem}_mask.png"), mask)


def draw_candidate_debug(candidate):
    warped = candidate["oriented"].copy()
    x, y, w, h = candidate["panel_box"]
    ix, iy, iw, ih = candidate["inner_box"]
    cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(warped, (x + ix, y + iy), (x + ix + iw, y + iy + ih), (255, 0, 0), 1)
    label = f"{candidate['rotation']} {candidate['code']} {candidate['value']}"
    cv2.putText(warped, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return warped


def main():
    parser = argparse.ArgumentParser(description="Recognize white bin numbers on black label panels.")
    parser.add_argument("inputs", nargs="+", help="Image files or directories.")
    parser.add_argument("--debug-dir", type=Path, help="Write panel crops and debug overlays.")
    parser.add_argument("--csv", type=Path, help="Write results as CSV.")
    parser.add_argument("--min-digits", type=int, default=DEFAULT_MIN_DIGITS, help="Minimum accepted digit count.")
    parser.add_argument("--max-digits", type=int, default=DEFAULT_MAX_DIGITS, help="Maximum accepted digit count.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum accepted recognizer confidence.")
    parser.add_argument("--min-blur", type=float, default=35.0, help="Minimum Laplacian blur score.")
    parser.add_argument(
        "--min-label-quality",
        type=float,
        default=0.55,
        help="Minimum accepted quadrilateral geometry score.",
    )
    parser.add_argument(
        "--engine",
        choices=("template", "tesseract", "auto", "cnn"),
        default="auto",
        help="Digit recognizer. cnn uses --cnn-checkpoint; auto uses Tesseract; template is for development only.",
    )
    parser.add_argument("--cnn-checkpoint", type=Path, help="TinyDigitCNN checkpoint for --engine cnn.")
    parser.add_argument("--cnn-device", default="cpu", help="Torch device for --engine cnn, usually cpu or cuda.")
    args = parser.parse_args()

    paths = []
    for item in args.inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("*.png")))
            paths.extend(sorted(p.glob("*.jpg")))
            paths.extend(sorted(p.glob("*.jpeg")))
        else:
            paths.append(p)

    templates = make_templates()
    cnn_bundle = None
    if args.engine == "cnn":
        if args.cnn_checkpoint is None:
            print("WARN: --engine cnn was selected without --cnn-checkpoint; results will be ERR_CNN_MISSING.")
        else:
            try:
                cnn_bundle = load_digit_cnn(args.cnn_checkpoint, device=args.cnn_device)
            except Exception as exc:
                print(f"WARN: failed to load CNN checkpoint {args.cnn_checkpoint}: {exc}")

    rows = [
        process_image(
            path,
            templates,
            args.debug_dir,
            args.engine,
            min_digits=args.min_digits,
            max_digits=args.max_digits,
            min_confidence=args.min_confidence,
            min_blur=args.min_blur,
            min_label_quality=args.min_label_quality,
            cnn_bundle=cnn_bundle,
        )
        for path in paths
    ]

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "ok",
                    "code",
                    "message",
                    "value",
                    "confidence",
                    "source",
                    "recognizer",
                    "box",
                    "blur",
                    "label_quality",
                    "rotation",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(
            f"{row['file']}\tok={row['ok']}\tcode={row['code']}\t"
            f"value={row['value'] or 'N/A'}\tconf={row['confidence']}\t"
            f"blur={row['blur']}\tlabel_quality={row['label_quality']}\t"
            f"rotation={row['rotation']}\tsource={row['source']}\t"
            f"recognizer={row['recognizer']}\tbox={row['box']}"
        )


if __name__ == "__main__":
    main()
