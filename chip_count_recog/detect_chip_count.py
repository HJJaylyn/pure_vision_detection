#!/usr/bin/env python3
"""Count silver chips in a 2 x 3 waffle box with conservative CV geometry."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


WARP_SIZE = (360, 360)
WAFFLE_CROP_MARGIN = 0.06


@dataclass
class QuadCandidate:
    quad: np.ndarray
    score: float
    metrics: dict[str, float]


def order_points(points: np.ndarray) -> np.ndarray:
    """Return four distinct points in a stable cyclic image order.

    The common min/max ``x + y`` and ``x - y`` shortcut breaks for a rotated
    diamond: the top and left corners can have the same sum, causing one
    point to be assigned twice and another one to disappear.  Sort around the
    centroid instead, then rotate the cyclic order to its most top-left point.
    """
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    cyclic = points[np.argsort(angles)]
    start = int(np.argmin(cyclic.sum(axis=1)))
    return np.roll(cyclic, -start, axis=0).astype(np.float32)


def clahe_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def edge_masks(image: np.ndarray, *, strict_black: bool = False) -> tuple[np.ndarray, np.ndarray]:
    gray = clahe_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 110)
    edge_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    # The black tray border is a color cue, but it is used only to join its
    # Canny-derived edge fragments into a physical quadrilateral candidate.
    if strict_black:
        # Do not use CLAHE for waffle material colour: it makes table shadows
        # look much darker than they physically are.
        raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(raw_gray, 0, 50)
    else:
        # Chip candidates retain the former local-contrast helper unchanged.
        dark = cv2.inRange(gray, 0, 95)
    dark_closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    return edges, cv2.bitwise_or(edge_closed, dark_closed)


def dark_mask(image: np.ndarray) -> np.ndarray:
    """A coarse black-material mask, used only to restrict edge searches."""
    # This is only for locating the black waffle box.  Use raw luminance so a
    # broad grey table shadow never becomes a fake black-material component.
    raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(raw_gray, 0, 50)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)


def black_component_coverage(
    labels: np.ndarray,
    component_areas: np.ndarray,
    quad: np.ndarray,
) -> float:
    """Return coverage of a tray-sized connected black-material component.

    A chip divider can make an excellent straight line, but a quadrilateral
    built around it contains only a small fraction of the tray's continuous
    black plastic.  The tray side wall may be included, so this intentionally
    measures component coverage rather than demanding a four-corner contour.
    """
    region = np.zeros(labels.shape, dtype=np.uint8)
    cv2.fillConvexPoly(region, quad.astype(np.int32), 255)
    quad_area = max(1, cv2.countNonZero(region))
    values, counts = np.unique(labels[region > 0], return_counts=True)
    minimum_component_area = max(labels.size * 0.020, quad_area * 0.60)
    coverage = 0.0
    for label, count in zip(values, counts):
        if label == 0 or component_areas[label] < minimum_component_area:
            continue
        coverage = max(coverage, float(count / component_areas[label]))
    return coverage


def waffle_edge_envelope(edges: np.ndarray) -> np.ndarray:
    """Bridge short Canny gaps in the tray outline without filling shadows."""
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    return cv2.dilate(joined, np.ones((3, 3), np.uint8), iterations=1)


def contour_quads(mask: np.ndarray, min_area: float, max_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 20:
            continue
        for epsilon_ratio in (0.012, 0.018, 0.026, 0.036):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(order_points(approx.reshape(4, 2)))
                break
    return quads


def quad_geometry(quad: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = quad
    width = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    height = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    ratio = max(width, height) / max(1.0, min(width, height))
    return float(width * height), float(ratio)


def warp_quad(image: np.ndarray, quad: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, output_size)


def expand_quad(quad: np.ndarray, image_shape: tuple[int, ...], margin: float) -> np.ndarray:
    """Retain a small safe border around the tray, so no slot is clipped."""
    center = quad.mean(axis=0, keepdims=True)
    expanded = center + (quad - center) * (1.0 + margin)
    height, width = image_shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    # ``quad`` already has the clockwise TL/TR/BR/BL correspondence required
    # by warp_quad, and uniform expansion preserves it.  Re-running the
    # sum/difference-based order heuristic here can collapse a diamond-shaped
    # perspective quad when two corners have near-identical coordinate sums.
    return expanded.astype(np.float32)


def border_edge_score(warped: np.ndarray) -> float:
    gray = clahe_gray(warped)
    edges = cv2.Canny(gray, 35, 110)
    band = max(3, int(min(gray.shape) * 0.045))
    border = np.zeros_like(edges, dtype=np.uint8)
    border[:band] = 1
    border[-band:] = 1
    border[:, :band] = 1
    border[:, -band:] = 1
    return float((edges[border > 0] > 0).mean())


def quad_edge_continuity(edges: np.ndarray, quad: np.ndarray) -> float:
    """Measure whether all four proposed sides have nearby Canny evidence."""
    support = np.zeros_like(edges, dtype=np.uint8)
    cv2.polylines(support, [quad.astype(np.int32).reshape(-1, 1, 2)], True, 255, 5, cv2.LINE_AA)
    expanded_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    side_values = []
    points = quad.astype(np.int32)
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        side = np.zeros_like(edges, dtype=np.uint8)
        cv2.line(side, tuple(start), tuple(end), 255, 5, cv2.LINE_AA)
        values = expanded_edges[side > 0]
        side_values.append(float((values > 0).mean()) if values.size else 0.0)
    # A single very good edge must not hide a missing side.
    return float(0.65 * min(side_values) + 0.35 * np.mean(side_values))


def straight_line_segments(edges: np.ndarray) -> np.ndarray | None:
    """Extract the reusable Canny line set used to support tray sides."""
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=18,
        maxLineGap=12,
    )
    return None if segments is None else segments.reshape(-1, 4)


def quad_straight_line_support(
    edges: np.ndarray,
    quad: np.ndarray,
    segments: np.ndarray | None = None,
) -> tuple[float, float, int]:
    """Return side support without requiring Canny corners to be connected."""
    if segments is None:
        segments = straight_line_segments(edges)
    if segments is None:
        return 0.0, 0.0, 0
    side_support = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 20:
            return 0.0, 0.0, 0
        unit = direction / length
        intervals: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in segments:
            first, second = np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)
            segment = second - first
            segment_length = float(np.linalg.norm(segment))
            if segment_length < max(14.0, length * 0.12):
                continue
            alignment = abs(float(np.dot(segment / segment_length, unit)))
            midpoint = 0.5 * (first + second)
            midpoint_distance = abs(float(np.cross(unit, midpoint - start)))
            if alignment < 0.90 or midpoint_distance > max(7.0, length * 0.07):
                continue
            projected = np.array([np.dot(first - start, unit), np.dot(second - start, unit)])
            left, right = max(0.0, float(projected.min())), min(length, float(projected.max()))
            if right > left:
                intervals.append((left, right))
        if not intervals:
            side_support.append(0.0)
            continue
        intervals.sort()
        covered, current_left, current_right = 0.0, intervals[0][0], intervals[0][1]
        for left, right in intervals[1:]:
            if left <= current_right:
                current_right = max(current_right, right)
            else:
                covered += current_right - current_left
                current_left, current_right = left, right
        covered += current_right - current_left
        side_support.append(min(1.0, covered / length))
    supported_sides = sum(value >= 0.12 for value in side_support)
    return float(min(side_support)), float(np.mean(side_support)), supported_sides


def min_area_quads(mask: np.ndarray, min_area: float, max_area: float) -> list[np.ndarray]:
    """Fallback coarse quadrilaterals from connected black material.

    They are deliberately not accepted by themselves: their purpose is to
    recover a search area when the top/side corner is broken in Canny.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        quads.append(order_points(box))
    return quads


def hull_quads(mask: np.ndarray, min_area: float, max_area: float) -> list[np.ndarray]:
    """Fit perspective quadrilaterals to a thick tray's irregular silhouette.

    Unlike ``minAreaRect``, a four-point convex-hull approximation can retain
    the visible trapezoid/parallelogram caused by camera perspective.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        if perimeter < 20:
            continue
        for epsilon_ratio in (0.010, 0.018, 0.028, 0.042, 0.060, 0.085):
            approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quads.append(order_points(approx.reshape(4, 2)))
                break
    return quads


def select_waffle_component(
    black: np.ndarray,
    edges: np.ndarray,
    min_area: float,
    max_area: float,
    line_segments: np.ndarray | None = None,
) -> np.ndarray | None:
    """Choose one strict-black connected component before fitting any geometry."""
    contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray]] = []
    image_area = black.shape[0] * black.shape[1]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if min(width, height) < 12:
            continue
        ratio = max(width, height) / max(1.0, min(width, height))
        if not 0.50 <= ratio <= 2.00:
            continue
        component = np.zeros_like(black)
        cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
        envelope = order_points(cv2.boxPoints(rect))
        _, line_mean_support, line_side_count = quad_straight_line_support(edges, envelope, line_segments)
        # Curved cable edges can be dense, but they cannot support several
        # long, mutually consistent sides of this rotated tray envelope.
        if line_side_count < 2:
            continue
        area_score = min(1.0, area / (image_area * 0.035))
        line_score = min(1.0, line_mean_support / 0.35)
        side_score = line_side_count / 4.0
        candidates.append((0.64 * line_score + 0.22 * side_score + 0.14 * area_score, component))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    vx1, vy1, x1, y1 = first.reshape(4)
    vx2, vy2, x2, y2 = second.reshape(4)
    matrix = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    rhs = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    if abs(float(np.linalg.det(matrix))) < 1e-4:
        return None
    scale = np.linalg.solve(matrix, rhs)[0]
    return np.array([x1 + scale * vx1, y1 + scale * vy1], dtype=np.float32)


def _segment_to_line(segment: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = segment.astype(np.float32)
    direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    direction /= max(1e-6, float(np.linalg.norm(direction)))
    return np.array([direction[0], direction[1], x1, y1], dtype=np.float32)


def hough_perspective_quads(
    edges: np.ndarray,
    support_segments: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Build quadrilaterals from two pairs of roughly parallel long lines.

    Canny may show four tray sides as disconnected segments.  Rather than
    demanding a closed contour, this joins two parallel line pairs through
    their intersections.  The result is a general perspective quadrilateral,
    not a forced square or rotated rectangle.
    """
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=40, maxLineGap=15)
    if raw is None:
        return []
    raw_segments = raw.reshape(-1, 4).astype(np.float32)
    raw_segments = sorted(raw_segments, key=lambda item: -float(np.linalg.norm(item[2:] - item[:2])))[:28]
    # Canny commonly produces two or more nearly coincident Hough segments
    # for one thick physical tray edge.  Keep the longest representative of
    # each such edge before pairing lines; otherwise their combinations grow
    # quadratically and dominate batch runtime.
    segments: list[np.ndarray] = []
    lines: list[np.ndarray] = []
    for segment in raw_segments:
        line = _segment_to_line(segment)
        is_duplicate = False
        for existing in lines:
            if abs(float(np.dot(line[:2], existing[:2]))) < 0.99:
                continue
            normal = np.array([-existing[1], existing[0]], dtype=np.float32)
            offset = abs(float(np.dot(normal, line[2:] - existing[2:])))
            if offset < 12.0:
                is_duplicate = True
                break
        if not is_duplicate:
            segments.append(segment)
            lines.append(line)
    pairs: list[tuple[int, int]] = []
    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            direction_similarity = abs(float(np.dot(lines[first][:2], lines[second][:2])))
            if direction_similarity < 0.94:
                continue
            # Do not pair duplicate Hough fragments from the same physical edge.
            normal = np.array([-lines[first][1], lines[first][0]], dtype=np.float32)
            offset = abs(float(np.dot(normal, lines[second][2:] - lines[first][2:])))
            if offset >= 18:
                pairs.append((first, second))

    height, width = edges.shape[:2]
    # Keep the source-line extent alongside each candidate.  It is a cheap
    # first-pass proxy for an external tray boundary and lets us avoid running
    # the full per-side support calculation on every Hough combination.
    quads_by_signature: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    for first_index, first_pair in enumerate(pairs):
        first_lines = (lines[first_pair[0]], lines[first_pair[1]])
        for second_pair in pairs[first_index + 1:]:
            second_lines = (lines[second_pair[0]], lines[second_pair[1]])
            # A waffle box has two distinct side directions.  Perspective can
            # skew the angle, hence the intentionally broad range.
            direction_difference = abs(float(np.dot(first_lines[0][:2], second_lines[0][:2])))
            if direction_difference > 0.70:
                continue
            corners = [
                _line_intersection(first_lines[0], second_lines[0]),
                _line_intersection(first_lines[0], second_lines[1]),
                _line_intersection(first_lines[1], second_lines[1]),
                _line_intersection(first_lines[1], second_lines[0]),
            ]
            if any(corner is None for corner in corners):
                continue
            quad = order_points(np.asarray(corners, dtype=np.float32))
            if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
                continue
            if np.any(quad[:, 0] < -0.08 * width) or np.any(quad[:, 0] > 1.08 * width):
                continue
            if np.any(quad[:, 1] < -0.08 * height) or np.any(quad[:, 1] > 1.08 * height):
                continue
            area, ratio = quad_geometry(quad)
            # Match the later broad waffle geometry sanity range before the
            # expensive line-support ranking.  This removes long table/cable
            # quadrilaterals, not perspective-valid tray quadrilaterals.
            if area < height * width * 0.008 or not 0.55 <= ratio <= 1.85:
                continue
            # Several Hough fragments frequently describe exactly the same
            # physical side.  Keep one representative per near-identical
            # quadrilateral so batch evaluation does not rescore hundreds of
            # pixel-level duplicates.
            signature = tuple(np.round(quad.reshape(-1) / 4.0).astype(int))
            source_lengths = [
                float(np.linalg.norm(segments[index][2:] - segments[index][:2]))
                for index in (*first_pair, *second_pair)
            ]
            source_extent = min(source_lengths)
            previous = quads_by_signature.get(signature)
            if previous is None or source_extent > previous[1]:
                quads_by_signature[signature] = (quad, source_extent)
    # This is a safety cap for cluttered scenes.  Rank by the shortest of the
    # four source lines: a tray's outer boundary needs four substantial edges,
    # whereas an internal chip divider or cable combination usually has at
    # least one short fragment.  Full Canny support is still evaluated later
    # for the retained candidates.
    ranked = sorted(
        quads_by_signature.values(),
        key=lambda item: (item[1], quad_geometry(item[0])[0]),
        reverse=True,
    )
    return [quad for quad, _ in ranked[:96]]


def fit_canny_quad_from_envelope(edges: np.ndarray, envelope: np.ndarray) -> np.ndarray | None:
    """Turn disconnected Canny tray sides into a perspective quadrilateral.

    The strict-black component supplies only a coarse rotated envelope.  Canny
    points near each of its four sides independently fit the actual lines, so
    opposite sides may converge instead of being forced parallel.
    """
    points = np.column_stack(np.nonzero(edges > 0))[:, ::-1].astype(np.float32)
    if points.size == 0:
        return None
    fitted_lines = []
    for start, end in zip(envelope, np.roll(envelope, -1, axis=0)):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 20:
            return None
        unit = direction / length
        relative = points - start
        projection = relative @ unit
        closest = start + np.clip(projection, -0.15 * length, 1.15 * length)[:, None] * unit
        distance = np.linalg.norm(points - closest, axis=1)
        nearby = points[(projection >= -0.15 * length) & (projection <= 1.15 * length) & (distance <= max(7.0, length * 0.08))]
        if len(nearby) < 12:
            return None
        line = cv2.fitLine(nearby.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
        fitted_lines.append(line.astype(np.float32))

    top, right, bottom, left = fitted_lines
    corners = [
        _line_intersection(left, top),
        _line_intersection(top, right),
        _line_intersection(right, bottom),
        _line_intersection(bottom, left),
    ]
    if any(corner is None for corner in corners):
        return None
    quad = np.asarray(corners, dtype=np.float32)
    height, width = edges.shape[:2]
    if np.any(quad[:, 0] < -0.10 * width) or np.any(quad[:, 0] > 1.10 * width):
        return None
    if np.any(quad[:, 1] < -0.10 * height) or np.any(quad[:, 1] > 1.10 * height):
        return None
    quad = order_points(quad)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    return quad


def score_waffle_envelope(image: np.ndarray, edges: np.ndarray, envelope: np.ndarray) -> QuadCandidate | None:
    """Score a rotated safety envelope from an irregular, thick tray silhouette.

    The black external silhouette is not required to have four corners.  Its
    minimum-area rectangle is only a safe crop envelope and may include a side
    wall; it is deliberately scored without the four-side Canny requirement.
    """
    height, width = image.shape[:2]
    area, ratio = quad_geometry(envelope)
    if not 0.65 <= ratio <= 1.55:
        return None
    warped = warp_quad(image, envelope, (220, 220))
    raw_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    strict_black_coverage = float((raw_gray < 50).mean())
    dark_total = float((raw_gray < 58).mean())
    if strict_black_coverage < 0.16:
        return None
    structure_score = waffle_structure_score(warped)
    edge_score = border_edge_score(warped)
    area_score = min(1.0, area / (height * width * 0.075))
    square_score = max(0.0, 1.0 - (ratio - 1.0) / 0.55)
    score = (
        0.26 * strict_black_coverage
        + 0.24 * dark_total
        + 0.22 * structure_score
        + 0.15 * square_score
        + 0.08 * min(1.0, edge_score * 5.0)
        + 0.05 * area_score
    )
    return QuadCandidate(envelope, float(score), {
        "area": area,
        "ratio": ratio,
        "dark_border": 0.0,
        "dark_total": dark_total,
        "strict_black_coverage": strict_black_coverage,
        "edge_score": edge_score,
        "edge_continuity": 0.0,
        "structure_score": structure_score,
        "is_thick_box_envelope": 1.0,
    })


def waffle_structure_score(warped: np.ndarray) -> float:
    """Score the repeated tray-divider edges, independent of 2x3 orientation."""
    gray = clahe_gray(warped)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 110).astype(np.float32) / 255.0
    height, width = edges.shape
    interior = edges[int(height * 0.12):int(height * 0.88), int(width * 0.12):int(width * 0.88)]
    if interior.size == 0:
        return 0.0
    density = float(np.clip(interior.mean() / 0.08, 0.0, 1.0))

    def separated_peaks(projection: np.ndarray) -> float:
        values = projection.copy()
        peaks = []
        for _ in range(2):
            index = int(np.argmax(values))
            peaks.append(values[index])
            radius = max(3, values.size // 12)
            values[max(0, index - radius):min(values.size, index + radius + 1)] = 0
        return float(np.mean(peaks) / max(1e-6, projection.max()))

    vertical = cv2.GaussianBlur(interior.mean(axis=0).reshape(1, -1), (1, 13), 0).reshape(-1)
    horizontal = cv2.GaussianBlur(interior.mean(axis=1).reshape(-1, 1), (13, 1), 0).reshape(-1)
    return float(0.38 * density + 0.31 * separated_peaks(vertical) + 0.31 * separated_peaks(horizontal))


def score_waffle_quad(image: np.ndarray, edges: np.ndarray, quad: np.ndarray) -> QuadCandidate | None:
    """Score a possible top face, not merely a dark external component."""
    height, width = image.shape[:2]
    area, ratio = quad_geometry(quad)
    if not 0.72 <= ratio <= 1.38:
        return None
    warped = warp_quad(image, quad, (220, 220))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    border = max(5, int(gray.shape[0] * 0.07))
    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:border] = True
    border_mask[-border:] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    raw_warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    dark_border = float((raw_warped_gray[border_mask] < 68).mean())
    dark_total = float((raw_warped_gray < 72).mean())
    warped_edge_score = border_edge_score(warped)
    edge_continuity = quad_edge_continuity(edges, quad)
    structure_score = waffle_structure_score(warped)
    area_score = min(1.0, area / (height * width * 0.075))
    square_score = max(0.0, 1.0 - (ratio - 1.0) / 0.38)
    score = (
        0.25 * edge_continuity
        + 0.20 * structure_score
        + 0.18 * dark_border
        + 0.12 * dark_total
        + 0.15 * square_score
        + 0.05 * min(1.0, warped_edge_score * 5.0)
        + 0.05 * area_score
    )
    return QuadCandidate(quad, float(score), {
        "area": area,
        "ratio": ratio,
        "dark_border": dark_border,
        "dark_total": dark_total,
        "edge_score": warped_edge_score,
        "edge_continuity": edge_continuity,
        "structure_score": structure_score,
    })


def map_quad_from_warp(quad: np.ndarray, source_quad: np.ndarray, warp_size: tuple[int, int]) -> np.ndarray:
    """Map a local refinement quadrilateral back into the original image."""
    width, height = warp_size
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    inverse = cv2.getPerspectiveTransform(dst, source_quad.astype(np.float32))
    return cv2.perspectiveTransform(quad.astype(np.float32).reshape(1, 4, 2), inverse).reshape(4, 2)


def refine_top_face(image: np.ndarray, coarse_quad: np.ndarray) -> QuadCandidate | None:
    """Re-detect the top plane inside a rough thick-box quadrilateral."""
    local_size = (420, 420)
    rough_warp = warp_quad(image, coarse_quad, local_size)
    local_edges, local_mask = edge_masks(rough_warp, strict_black=True)
    local_black = dark_mask(rough_warp)
    area_total = local_size[0] * local_size[1]
    raw_quads = contour_quads(local_mask, area_total * 0.30, area_total * 0.98)
    raw_quads.extend(min_area_quads(local_black, area_total * 0.30, area_total * 0.98))
    candidates = []
    for local_quad in raw_quads:
        original_quad = order_points(map_quad_from_warp(local_quad, coarse_quad, local_size))
        candidate = score_waffle_quad(image, cv2.Canny(cv2.GaussianBlur(clahe_gray(image), (5, 5), 0), 35, 110), original_quad)
        if candidate is not None:
            candidates.append(candidate)
    return max(candidates, key=lambda item: item.score) if candidates else None


def find_waffle_box(image: np.ndarray) -> tuple[QuadCandidate | None, dict[str, np.ndarray]]:
    """Find the waffle top surface from Canny quadrilaterals plus black material.

    The thick black side wall is allowed to generate a *coarse* candidate, but
    a candidate only wins when its four sides are also supported by Canny.
    This avoids treating a generic dark blob as the tray's top plane.
    """
    # This is the established crop used by the working six-chip result.  Do
    # not silently replace the chip input with a tighter experimental top-face
    # refinement: it can clip a slot before chip detection starts.
    raw_edges, _ = edge_masks(image, strict_black=True)
    black = dark_mask(image)
    _, black_labels, black_stats, _ = cv2.connectedComponentsWithStats(black)
    black_component_areas = black_stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    line_segments = straight_line_segments(raw_edges)
    height, width = image.shape[:2]
    min_area = height * width * 0.006
    max_area = height * width * 0.42
    edge_envelope = waffle_edge_envelope(raw_edges)
    # Keep this visualisation and contour source edge-only.  The strict-black
    # mask is a score helper, never a filled candidate region: otherwise a
    # dark table shadow can fuse into a fake waffle-box contour.
    candidate_mask = edge_envelope
    # Primary geometry source: directly extracted Canny quadrilaterals.  This
    # is where the real perspective tray in 000022 already appears as four
    # long straight sides, before any black-component fallback is needed.
    edge_quads = contour_quads(edge_envelope, min_area, max_area)
    hough_quads = hough_perspective_quads(raw_edges, line_segments)
    waffle_component = select_waffle_component(black, raw_edges, min_area, max_area, line_segments)
    component_hull_quads = hull_quads(waffle_component, min_area, max_area) if waffle_component is not None else []
    component_envelopes = min_area_quads(waffle_component, min_area, max_area) if waffle_component is not None else []
    fitted_canny_quads = [
        quad
        for envelope in component_envelopes
        if (quad := fit_canny_quad_from_envelope(raw_edges, envelope)) is not None
    ]
    candidates: list[QuadCandidate] = []
    seen: set[tuple[int, ...]] = set()
    # Canny four-line quadrilaterals are primary.  Component-derived geometry
    # is only a fallback for real broken-edge cases.
    for quad in [*hough_quads, *edge_quads, *fitted_canny_quads, *component_hull_quads]:
        signature = tuple(np.round(quad.reshape(-1) / 4.0).astype(int))
        if signature in seen:
            continue
        seen.add(signature)
        area, ratio = quad_geometry(quad)
        # Perspective may make the physical near-square waffle appear as a
        # clear trapezoid.  This is only a broad sanity bound, not a square fit.
        if not 0.55 <= ratio <= 1.85:
            continue
        quad_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(quad_mask, quad.astype(np.int32), 255)
        strict_black_coverage = float((black[quad_mask > 0] > 0).mean())
        # A tray contains a large connected mass of truly black plastic.  A
        # broad table/edge candidate may contain a few dark lines, but cannot
        # satisfy this material-coverage requirement.
        if strict_black_coverage < 0.16:
            continue
        component_coverage = black_component_coverage(
            black_labels,
            black_component_areas,
            quad,
        )
        warped = warp_quad(image, quad, (220, 220))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        border = max(5, int(gray.shape[0] * 0.07))
        border_mask = np.zeros_like(gray, dtype=bool)
        border_mask[:border] = True
        border_mask[-border:] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True
        raw_warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        dark_border = float((raw_warped_gray[border_mask] < 50).mean())
        dark_total = float((raw_warped_gray < 58).mean())
        warped_edge_score = border_edge_score(warped)
        edge_continuity = quad_edge_continuity(raw_edges, quad)
        line_min_support, line_mean_support, line_side_count = quad_straight_line_support(
            raw_edges,
            quad,
            line_segments,
        )
        # Canny often breaks at the physical corners.  Two or three supported
        # side directions plus the strict-black envelope are sufficient to
        # infer the missing portions; no end-to-end closed contour is needed.
        if line_side_count < 3 or line_mean_support < 0.35:
            continue
        area_score = min(1.0, area / (height * width * 0.075))
        # ``strict_black_coverage`` is deliberately a gate above, not a
        # reward here.  A real tray with several reflective chips is less
        # uniformly black than one of its internal black dividers.  Rewarding
        # raw darkness made the divider win over the actual outer tray in
        # 000022.  Once material plausibility is established, prefer the
        # inferred four-side geometry and its supported physical extent.
        score = (
            0.38 * line_mean_support
            + 0.20 * edge_continuity
            + 0.12 * min(1.0, warped_edge_score * 5.0)
            + 0.15 * area_score
            + 0.15 * component_coverage
        )
        candidates.append(QuadCandidate(quad, float(score), {
            "area": area,
            "ratio": ratio,
            "dark_border": dark_border,
            "dark_total": dark_total,
            "strict_black_coverage": strict_black_coverage,
            "component_coverage": component_coverage,
            "edge_score": warped_edge_score,
            "edge_continuity": edge_continuity,
            "line_min_support": line_min_support,
            "line_mean_support": line_mean_support,
            "line_side_count": float(line_side_count),
            # Kept for CSV/debug compatibility.  It is not a selection term,
            # so computing a separate internal Canny map for every candidate
            # only made dataset evaluation needlessly slow.
            "structure_score": 0.0,
            "is_line_quad": 1.0,
        }))
    for envelope in component_envelopes:
        signature = tuple(np.round(envelope.reshape(-1) / 4.0).astype(int))
        if signature in seen:
            continue
        seen.add(signature)
        candidate = score_waffle_envelope(image, raw_edges, envelope)
        if candidate is not None:
            candidates.append(candidate)
    line_candidates = [candidate for candidate in candidates if candidate.metrics.get("is_line_quad", 0.0) > 0.0]
    best = max(line_candidates or candidates, key=lambda item: item.score) if candidates else None
    return best, {"canny": raw_edges, "candidate_mask": candidate_mask, "black_mask": black}


def silver_score(image: np.ndarray, quad: np.ndarray) -> tuple[float, dict[str, float]]:
    """Weak material evidence only; reflected chips are not reliably white."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[mask > 0]
    if values.size < 20:
        return 0.0, {"median": 0.0, "bright_ratio": 0.0, "edge_score": 0.0}
    balanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    bright_ratio = float((values > 90).mean())
    balanced_ratio = float((balanced[mask > 0] > 120).mean())
    median = float(np.median(values))
    edges = cv2.Canny(gray, 35, 110)
    edge_score = float((edges[mask > 0] > 0).mean())
    score = 0.42 * bright_ratio + 0.26 * balanced_ratio + 0.22 * np.clip((median - 35.0) / 115.0, 0.0, 1.0) + 0.10 * np.clip(edge_score * 7.0, 0.0, 1.0)
    return float(score), {"median": median, "bright_ratio": bright_ratio, "balanced_ratio": balanced_ratio, "edge_score": edge_score}


def black_surround_score(image: np.ndarray, quad: np.ndarray) -> tuple[float, dict[str, float]]:
    """Measure the physical cue 'a rectangular surface enclosed by black tray'."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inside = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
    outer = cv2.dilate(inside, np.ones((13, 13), np.uint8), iterations=1)
    ring = cv2.bitwise_and(outer, cv2.bitwise_not(inside))
    inner_values = gray[inside > 0]
    ring_values = gray[ring > 0]
    if inner_values.size < 20 or ring_values.size < 20:
        return 0.0, {"surround_dark_ratio": 0.0, "local_contrast": 0.0}
    surround_dark_ratio = float((ring_values < 105).mean())
    local_contrast = float(np.median(inner_values) - np.median(ring_values))
    # The chip must be brighter than its immediately surrounding black slot.
    # This is deliberately a *local* difference, not an absolute whiteness
    # requirement, so exposure changes and reflections remain tolerable.
    score = 0.42 * surround_dark_ratio + 0.58 * np.clip((local_contrast - 3.0) / 60.0, 0.0, 1.0)
    return float(score), {"surround_dark_ratio": surround_dark_ratio, "local_contrast": local_contrast}


def chip_surface_mask(image: np.ndarray) -> np.ndarray:
    """Generate chip candidates from surfaces brighter than the black tray.

    This is intentionally not a fixed six-cell layout.  Otsu is applied only
    after the waffle has been cropped, so it separates each locally brighter
    chip surface from its immediately surrounding black grid.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # A reflected chip can be materially brighter than the black tray while
    # still falling below the global Otsu split.  Relax the split a little,
    # then let geometry and the black surround reject unrelated regions.
    # The tray itself is genuinely close to black.  Let locally darker silver
    # surfaces enter as weak evidence, then use their black enclosure and
    # rectangular Canny boundary to reject tray/background regions.
    _, bright = cv2.threshold(blurred, max(1, int(otsu_threshold * 0.72)), 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)


def quad_bbox(quad: np.ndarray) -> tuple[int, int, int, int]:
    x, y, width, height = cv2.boundingRect(quad.astype(np.float32))
    return x, y, x + width, y + height


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return float(intersection / union) if union else 0.0


def quad_overlap_of_smaller(first: np.ndarray, second: np.ndarray) -> float:
    """Return how much of the smaller convex quadrilateral is overlapped."""
    first_area = abs(float(cv2.contourArea(first.astype(np.float32))))
    second_area = abs(float(cv2.contourArea(second.astype(np.float32))))
    if min(first_area, second_area) < 1e-6:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(
        first.astype(np.float32),
        second.astype(np.float32),
    )
    return float(intersection / min(first_area, second_area))


def quad_mask_coverage(mask: np.ndarray, quad: np.ndarray) -> float:
    """Measure how much of a proposed chip surface is supported by a mask."""
    region = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(region, quad.astype(np.int32), 255)
    pixels = region > 0
    return float((mask[pixels] > 0).mean()) if np.any(pixels) else 0.0


def find_chip_quads(rectified_waffle: np.ndarray) -> tuple[list[QuadCandidate], dict[str, np.ndarray]]:
    """Find all silver chip quadrilaterals directly, without fixed grid cells.

    The waffle layout can be used by a future consistency check, but it must
    never define a chip's final bounds: tray crop errors and arbitrary rotation
    make that assumption unsafe.
    """
    raw_edges, candidate_mask = edge_masks(rectified_waffle)
    bright_surfaces = chip_surface_mask(rectified_waffle)
    # Keep Canny independent from the dark-material helper.  A small close
    # reconnects broken chip borders, while preserving the individual
    # rectangles that get merged when dark tray regions are ORed into a mask.
    chip_canny_candidates = cv2.morphologyEx(
        raw_edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    height, width = rectified_waffle.shape[:2]
    min_area = height * width * 0.010
    # This is a conservative physical sanity bound, not a fixed-slot model.
    # Perspective correction and the retained tray margin can enlarge a valid
    # chip somewhat, but no one chip can plausibly fill over one quarter of
    # the rectified waffle image.
    max_area = height * width * 0.25
    raw_quads = contour_quads(chip_canny_candidates, min_area, max_area)
    raw_quads.extend(contour_quads(candidate_mask, min_area, max_area))
    raw_quads.extend(contour_quads(bright_surfaces, min_area, max_area))
    raw_quads.extend(min_area_quads(bright_surfaces, min_area, max_area))
    candidates: list[QuadCandidate] = []
    seen: set[tuple[int, ...]] = set()
    for quad in raw_quads:
        signature = tuple(np.round(quad.reshape(-1) / 3.0).astype(int))
        if signature in seen:
            continue
        seen.add(signature)
        area, ratio = quad_geometry(quad)
        # Contour filtering above constrains the source component.  A coarse
        # four-point approximation can still expand well beyond that contour,
        # so enforce the physical bound again on the final quadrilateral.
        if not min_area <= area <= max_area:
            continue
        if not 0.75 <= ratio <= 3.20:
            continue
        silver, color_metrics = silver_score(rectified_waffle, quad)
        surround, surround_metrics = black_surround_score(rectified_waffle, quad)
        bright_support = quad_mask_coverage(bright_surfaces, quad)
        edge = border_edge_score(warp_quad(rectified_waffle, quad, (120, 80)))
        continuity = quad_edge_continuity(raw_edges, quad)
        rectangularity = min(1.0, area / max(1.0, cv2.contourArea(quad.astype(np.float32))))
        ratio_score = max(0.0, 1.0 - abs(ratio - 1.55) / 0.85)
        edge_score = min(1.0, edge * 5.0)
        geometry_score = 0.48 * continuity + 0.28 * ratio_score + 0.14 * edge_score + 0.10 * rectangularity
        # Geometry and black enclosure decide acceptance.  Absolute surface
        # brightness is deliberately weak because a silver chip may reflect
        # the black tray, the camera, or a dim part of the room.
        # Canny proposes the shape; the independently thresholded brighter
        # surface is corroborating evidence.  It remains soft because a chip
        # can reflect a dark object and therefore look almost black.
        score = 0.44 * geometry_score + 0.37 * surround + 0.10 * silver + 0.09 * bright_support
        candidates.append(QuadCandidate(quad, float(score), {
            "ratio": ratio,
            "area": area,
            "geometry_score": geometry_score,
            "edge_continuity": continuity,
            "surround_score": surround,
            "silver_score": silver,
            "edge_score": edge,
            "bright_support": bright_support,
            **surround_metrics,
            **color_metrics,
        }))

    # Physical chips cannot overlap.  Canny can nevertheless produce a small
    # inner rectangle plus a larger surface rectangle for the same chip.  Use
    # the actual quadrilateral intersection (not an axis-aligned bbox) and
    # prefer the size that agrees with the other chip candidates.
    plausible_areas = [
        candidate.metrics["area"]
        for candidate in candidates
        if candidate.metrics["area"] <= height * width * 0.15
    ]
    typical_area = float(np.median(plausible_areas)) if plausible_areas else 0.0

    def selection_score(candidate: QuadCandidate) -> float:
        if typical_area <= 0.0:
            return candidate.score
        size_distance = abs(float(np.log(max(1.0, candidate.metrics["area"]) / typical_area)))
        size_consistency = float(np.exp(-size_distance))
        return float(0.75 * candidate.score + 0.25 * size_consistency)

    selected: list[QuadCandidate] = []
    for candidate in sorted(candidates, key=selection_score, reverse=True):
        if any(quad_overlap_of_smaller(candidate.quad, kept.quad) >= 0.60 for kept in selected):
            continue
        selected.append(candidate)
    return selected, {
        "canny": raw_edges,
        "candidate_mask": candidate_mask,
        "chip_canny_candidates": chip_canny_candidates,
        "bright_surfaces": bright_surfaces,
    }


def analyze_image(path: Path, occupancy_threshold: float) -> tuple[dict, dict[str, np.ndarray]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "code": "ERR_IMAGE_READ", "file": str(path)}, {}
    waffle, waffle_stages = find_waffle_box(image)
    if waffle is None:
        return {"ok": False, "code": "ERR_WAFFLE_BOX_NOT_FOUND", "file": str(path)}, {
            "original": image,
            "waffle_canny": waffle_stages["canny"],
            "waffle_candidates": waffle_stages["candidate_mask"],
            "waffle_black": waffle_stages["black_mask"],
        }

    # The detected outer contour may land on the top/side corner.  Retain a
    # small outward margin so a valid slot is never clipped before the chip
    # detector sees it.
    crop_quad = expand_quad(waffle.quad, image.shape, WAFFLE_CROP_MARGIN)
    rectified = warp_quad(image, crop_quad, WARP_SIZE)
    original_annotated = image.copy()
    cv2.polylines(original_annotated, [waffle.quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    cv2.polylines(original_annotated, [crop_quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 220, 255), 2)
    cv2.putText(
        original_annotated,
        f"green=Canny tray quad {waffle.score:.2f}; yellow=safe crop margin",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    chips_debug = rectified.copy()
    detections = []
    candidates, chip_stages = find_chip_quads(rectified)
    for index, chip in enumerate(candidates):
        occupied = chip.score >= occupancy_threshold
        color = (40, 210, 40) if occupied else (0, 80, 235)
        cv2.polylines(chips_debug, [chip.quad.astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
        anchor = chip.quad.astype(np.int32).min(axis=0)
        cv2.putText(
            chips_debug,
            f"{index}: {'chip' if occupied else 'reject'} {chip.score:.2f} g={chip.metrics['geometry_score']:.2f} e={chip.metrics['surround_score']:.2f}",
            tuple(anchor + np.array([2, 16])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        detections.append({
            "candidate": index,
            "occupied": occupied,
            "score": round(chip.score, 4),
            **{key: round(value, 4) for key, value in chip.metrics.items()},
        })

    count = sum(int(item["occupied"]) for item in detections)
    result = {
        "ok": True,
        "code": "OK",
        "file": str(path),
        "chip_count": count,
        "waffle_box_score": round(waffle.score, 4),
        "occupancy_threshold": occupancy_threshold,
        "slots": detections,
    }
    return result, {
        "original": original_annotated,
        "waffle_canny": waffle_stages["canny"],
        "waffle_candidates": waffle_stages["candidate_mask"],
        "waffle_black": waffle_stages["black_mask"],
        "rectified": rectified,
        "chip_canny_candidates": chip_stages["chip_canny_candidates"],
        "chip_bright_surfaces": chip_stages["bright_surfaces"],
        "chips": chips_debug,
    }


def caption(image: np.ndarray, text: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = image.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def make_debug_image(stages: dict[str, np.ndarray]) -> np.ndarray:
    items = [
        ("original + selected waffle quadrilateral", stages.get("original")),
        ("waffle Canny edges", stages.get("waffle_canny")),
        ("joined Canny outline + strict-black helper mask", stages.get("waffle_candidates")),
        ("strict raw-black helper mask (not the final quadrilateral)", stages.get("waffle_black")),
        ("rectified waffle top face", stages.get("rectified")),
        ("rectified chip Canny candidates (small gaps reconnected)", stages.get("chip_canny_candidates")),
        ("chip bright-surface candidates (not fixed grid cells)", stages.get("chip_bright_surfaces")),
        (
            "green: accepted silver chip quadrilateral; orange: rejected candidate",
            stages.get("chips"),
        ),
    ]
    panels = [caption(image, label) for label, image in items if image is not None]
    width = max(panel.shape[1] for panel in panels)
    resized = [cv2.resize(panel, (width, max(1, int(panel.shape[0] * width / panel.shape[1])))) for panel in panels]
    return np.vstack(resized)


def iter_images(dataset: Path) -> Iterable[tuple[str, Path]]:
    for label_dir in sorted(path for path in dataset.iterdir() if path.is_dir() and path.name.isdigit()):
        for image_path in sorted(label_dir.glob("*.jpg")):
            yield label_dir.name, image_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--occupancy-threshold", type=float, default=0.38)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    if (args.image is None) == (args.dataset is None):
        parser.error("provide exactly one IMAGE or --dataset DATASET")

    if args.image is not None:
        result, stages = analyze_image(args.image, args.occupancy_threshold)
        if args.debug_dir is not None and stages:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.debug_dir / f"{args.image.stem}_debug.jpg"), make_debug_image(stages))
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("ok") else 1

    # A dataset evaluation is one self-contained run: never leave stale
    # problem images or rows from a previous run beside the new results.
    if args.debug_dir is not None and args.debug_dir.exists():
        shutil.rmtree(args.debug_dir)
    if args.csv is not None and args.csv.exists():
        args.csv.unlink()

    image_items = list(iter_images(args.dataset))
    print(f"Evaluating {len(image_items)} images...", flush=True)
    rows = []
    for index, (expected, image_path) in enumerate(image_items, start=1):
        result, stages = analyze_image(image_path, args.occupancy_threshold)
        result["expected_count"] = int(expected)
        result["correct"] = bool(result.get("ok") and result.get("chip_count") == int(expected))
        rows.append(result)
        if args.debug_dir is not None and stages and not result["correct"]:
            output_dir = args.debug_dir / "problems" / expected
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_debug.jpg"), make_debug_image(stages))
        if index % 10 == 0 or index == len(image_items):
            print(f"Processed {index}/{len(image_items)} images", flush=True)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            fields = ["expected_count", "file", "ok", "code", "chip_count", "correct", "waffle_box_score"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    print(json.dumps({"total": len(rows), "ok": sum(row.get("ok", False) for row in rows), "correct": sum(row.get("correct", False) for row in rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
