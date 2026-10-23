"""Automatische Erkennung der äußeren Brettecken in einem Screenshot.

Die Erkennung ist bewusst zurückhaltend: Sie liefert nur dann ein Viereck, wenn
es deutlich nach einem quadratischen Brett aussieht und die Kanten des Vierecks
tatsächlich auf Bildkanten liegen. Schlägt sie fehl, bleibt die manuelle
Kalibrierung (vier Klicks oder ein aufgezogenes Rechteck) der Weg.

Bewertet werden mehrere Kandidaten je Kontur (Hüllkurve mit verschiedenen
Toleranzen sowie das minimale Rechteck). Das beste Viereck gewinnt über
Kantenunterstützung, Fläche und Quadratähnlichkeit.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_AREA_RATIO = 0.012
MAX_AREA_RATIO = 0.96
MIN_SQUARENESS = 0.40
MIN_DIAGONAL_RATIO = 0.50
MIN_EDGE_SUPPORT = 0.72
MAX_CONTOURS = 40
APPROX_EPSILONS = (0.01, 0.02, 0.03, 0.045, 0.06)


def _order_corners(points: np.ndarray | list[list[float]]) -> np.ndarray:
    """Sortiert vier Punkte als oben links, oben rechts, unten rechts, unten links."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if len(np.unique(pts.round(2), axis=0)) != 4:
        raise ValueError("Die vier Brettecken sind nicht eindeutig.")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    first = ordered[1] - ordered[0]
    second = ordered[2] - ordered[1]
    if first[0] * second[1] - first[1] * second[0] < 0:
        ordered = ordered[[0, 3, 2, 1]]
    ordered = ordered.astype(np.float32)
    if not _is_convex(ordered):
        raise ValueError("Die vier Brettecken bilden kein konvexes Viereck.")
    sides = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    if sides.min() <= 2.0:
        raise ValueError("Das Viereck ist zu klein.")
    return ordered


def _is_convex(points: np.ndarray) -> bool:
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if np.any(np.isclose(cross, 0.0, atol=1e-6)):
        return False
    return bool(np.all(cross > 0) or np.all(cross < 0))


def _edge_support(
    quad: np.ndarray, edge_distance: np.ndarray, tolerance: float, samples: int = 48
) -> float:
    """Anteil der Viereckkanten, der nahe an erkennbaren Bildkanten liegt.

    ``edge_distance`` ist die Distanz zum nächsten Kantenpixel; ein kleiner
    Toleranzradius fängt Rahmenbreiten und Antialiasing ab.
    """
    height, width = edge_distance.shape[:2]
    hits = 0
    total = 0
    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        for t in np.linspace(0.03, 0.97, samples):
            point = start + (end - start) * t
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < width and 0 <= y < height:
                total += 1
                if edge_distance[y, x] <= tolerance:
                    hits += 1
    return hits / total if total else 0.0


def _quad_quality(points: np.ndarray, image_area: float) -> tuple[float, float]:
    """Liefert (Flächenanteil, Quadratähnlichkeit) oder (0, 0) bei Ausschluss."""
    area = abs(float(cv2.contourArea(points)))
    if image_area <= 0:
        return 0.0, 0.0
    ratio = area / image_area
    if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
        return 0.0, 0.0
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    squareness = float(sides.min() / sides.max()) if sides.max() > 0 else 0.0
    diagonal_a = float(np.linalg.norm(points[0] - points[2]))
    diagonal_b = float(np.linalg.norm(points[1] - points[3]))
    diagonal_ratio = min(diagonal_a, diagonal_b) / max(diagonal_a, diagonal_b) if max(
        diagonal_a, diagonal_b
    ) > 0 else 0.0
    if squareness < MIN_SQUARENESS or diagonal_ratio < MIN_DIAGONAL_RATIO:
        return 0.0, 0.0
    return ratio, squareness


def _quad_candidates(contour: np.ndarray) -> list[np.ndarray]:
    """Erzeugt plausible Brettvierecke für eine Kontur."""
    result: list[np.ndarray] = []
    hull = cv2.convexHull(contour)
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter > 8.0:
        for epsilon in APPROX_EPSILONS:
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(approx) == 4:
                result.append(approx.reshape(4, 2).astype(np.float32))
    rect = cv2.minAreaRect(contour)
    if min(rect[1]) > 8.0:
        result.append(cv2.boxPoints(rect).astype(np.float32))
    return result


def _refine_quad(contour: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Schiebt die Ecken auf die Schnittpunkte der gefitteten Brettkanten."""
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 24:
        return None
    lines: list[tuple[float, float, float, float]] = []
    for index in range(4):
        corner_a = quad[index]
        corner_b = quad[(index + 1) % 4]
        edge = corner_b - corner_a
        length = float(np.linalg.norm(edge))
        if length < 16.0:
            return None
        direction = edge / length
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        distance = np.abs((points - corner_a) @ normal)
        selected = points[distance < max(3.0, length * 0.05)]
        if len(selected) < 12:
            return None
        projection = (selected - corner_a) @ direction
        selected = selected[(projection > -length * 0.1) & (projection < length * 1.1)]
        if len(selected) < 12:
            return None
        vx, vy, x0, y0 = cv2.fitLine(
            selected.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        lines.append((float(x0), float(y0), float(vx), float(vy)))
    refined: list[tuple[float, float] | None] = []
    for index in range(4):
        refined.append(_intersect(lines[(index - 1) % 4], lines[index]))
    if any(point is None for point in refined):
        return None
    candidate = np.asarray(refined, dtype=np.float32)
    if np.max(np.linalg.norm(candidate - quad, axis=1)) > 0.12 * float(
        np.linalg.norm(quad[0] - quad[2])
    ):
        return None
    try:
        return _order_corners(candidate)
    except ValueError:
        return None


def _intersect(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    x1, y1, vx1, vy1 = first
    x2, y2, vx2, vy2 = second
    denominator = vx1 * vy2 - vy1 * vx2
    if abs(denominator) < 1e-6:
        return None
    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denominator
    return float(x1 + t * vx1), float(y1 + t * vy1)


def _candidate_maps(gray: np.ndarray) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    maps = [
        cv2.Canny(blurred, 30, 90),
        cv2.Canny(blurred, 60, 170),
    ]
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maps.append(otsu)
    maps.append(cv2.bitwise_not(otsu))
    return maps


def detect_board_corners(
    image: np.ndarray,
    min_area_ratio: float = MIN_AREA_RATIO,
) -> list[list[float]] | None:
    """Sucht das äußere Schachbrett im Bild.

    Rückgabe ist eine Liste ``[[x, y], …]`` in der Reihenfolge oben links,
    oben rechts, unten rechts, unten links – oder ``None``, wenn kein
    ausreichend sicheres Brett gefunden wurde.
    """

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    height, width = gray.shape[:2]
    if min(height, width) < 64:
        return None
    image_area = float(height * width)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_map = cv2.Canny(blurred, 40, 130)
    edge_distance = cv2.distanceTransform(
        255 - edge_map, cv2.DIST_L2, 3
    )
    tolerance = max(1.5, 0.006 * float(np.hypot(height, width)))

    accepted: list[tuple[float, float, np.ndarray]] = []
    for edges in _candidate_maps(gray):
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS]:
            area = float(cv2.contourArea(contour))
            if area / image_area < min_area_ratio or area / image_area > MAX_AREA_RATIO:
                continue
            quads: list[np.ndarray] = []
            for raw in _quad_candidates(contour):
                try:
                    quads.append(_order_corners(raw))
                except ValueError:
                    continue
            if area / image_area >= 0.05:
                for quad in list(quads):
                    refined = _refine_quad(contour, quad)
                    if refined is not None:
                        quads.append(refined)
            for quad in quads:
                ratio, _squareness = _quad_quality(quad, image_area)
                if ratio <= 0.0:
                    continue
                support = _edge_support(quad, edge_distance, tolerance)
                if support >= MIN_EDGE_SUPPORT:
                    accepted.append((abs(float(cv2.contourArea(quad))), support, quad))

    if not accepted:
        return None
    # Bewertung: gut gestützte Kanten und große Fläche. Ein einzelnes Feld hat
    # nur 1/64 der Brettfläche und verliert dadurch deutlich.
    best = max(accepted, key=lambda item: item[1] ** 2 * item[0] ** 0.5)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in best[2]]


def rectangle_to_corners(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Wandelt ein aufgezogenes Rechteck in die vier Brett-Ecken um."""
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    return [[left, top], [right, top], [right, bottom], [left, bottom]]
