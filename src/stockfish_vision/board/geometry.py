from __future__ import annotations

import chess
import cv2
import numpy as np

ORIENTATIONS = ("white_bottom", "white_top", "white_left", "white_right")


def _as_corners(corners: np.ndarray | list[list[float]]) -> np.ndarray:
    try:
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    except (ValueError, TypeError) as exc:
        raise ValueError("Es werden genau vier Brettecken mit x/y-Koordinaten benötigt.") from exc
    if not np.isfinite(points).all():
        raise ValueError("Brettecken müssen endliche Koordinaten haben.")
    return points


def validate_corners(corners: np.ndarray | list[list[float]]) -> np.ndarray:
    """Validate an ordered, convex quadrilateral before warping."""
    points = _as_corners(corners)
    if len(np.unique(points, axis=0)) != 4:
        raise ValueError("Die vier Brettecken müssen voneinander verschieden sein.")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if np.any(np.isclose(cross, 0.0, atol=1e-4)) or not (
        np.all(cross > 0) or np.all(cross < 0)
    ):
        raise ValueError(
            "Die Ecken müssen ein konvexes Viereck bilden und der Reihe nach "
            "(oben links, oben rechts, unten rechts, unten links) gesetzt werden."
        )
    area = abs(float(cv2.contourArea(points)))
    if area < 64.0:
        raise ValueError("Der kalibrierte Brettbereich ist zu klein.")
    return points


def perspective_matrix(
    corners: np.ndarray | list[list[float]], size: int = 400
) -> np.ndarray:
    if not 64 <= int(size) <= 4096:
        raise ValueError("Die Warp-Größe muss zwischen 64 und 4096 Pixeln liegen.")
    src = validate_corners(corners)
    dst = np.array(
        [[0.0, 0.0], [size - 1.0, 0.0], [size - 1.0, size - 1.0], [0.0, size - 1.0]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Die Kalibrierungsmatrix ist nicht umkehrbar.") from exc


def warp_board(
    image: np.ndarray, corners: np.ndarray | list[list[float]], size: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Für die Bretttransformation wird ein Bild benötigt.")
    matrix = perspective_matrix(corners, size)
    warped = cv2.warpPerspective(image, matrix, (size, size))
    if warped.size == 0:
        raise RuntimeError("Die Bretttransformation ist fehlgeschlagen.")
    return warped, matrix


def square_to_image_rc(square: chess.Square, orientation: str) -> tuple[int, int]:
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unbekannte Orientierung: {orientation}")
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    if orientation == "white_bottom":
        return file_index, 7 - rank_index
    if orientation == "white_top":
        return 7 - file_index, rank_index
    if orientation == "white_left":
        return rank_index, file_index
    return 7 - rank_index, 7 - file_index


def image_rc_to_square(col: int, row: int, orientation: str) -> chess.Square:
    if not (0 <= col <= 7 and 0 <= row <= 7):
        raise ValueError("Feldkoordinaten müssen im Bereich 0..7 liegen.")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unbekannte Orientierung: {orientation}")
    if orientation == "white_bottom":
        file_index, rank_index = col, 7 - row
    elif orientation == "white_top":
        file_index, rank_index = 7 - col, row
    elif orientation == "white_left":
        file_index, rank_index = row, col
    else:
        file_index, rank_index = 7 - row, 7 - col
    return chess.square(file_index, rank_index)


def extract_square(warped: np.ndarray, col: int, row: int) -> np.ndarray:
    if not (0 <= col <= 7 and 0 <= row <= 7):
        raise ValueError("Feldkoordinaten müssen im Bereich 0..7 liegen.")
    height, width = warped.shape[:2]
    if height < 8 or width < 8:
        raise ValueError("Das transformierte Brett ist zu klein.")
    tile_w = width / 8.0
    tile_h = height / 8.0
    x0 = int(round(col * tile_w))
    y0 = int(round(row * tile_h))
    x1 = int(round((col + 1) * tile_w))
    y1 = int(round((row + 1) * tile_h))
    return warped[y0:y1, x0:x1]


def square_center_image(
    square: chess.Square, orientation: str, size: int
) -> tuple[float, float]:
    col, row = square_to_image_rc(square, orientation)
    tile = size / 8.0
    return (col + 0.5) * tile, (row + 0.5) * tile


def square_center_screen(
    square: chess.Square,
    orientation: str,
    size: int,
    matrix_inv: np.ndarray,
) -> tuple[float, float]:
    x, y = square_center_image(square, orientation, size)
    point = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, matrix_inv)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def square_polygon_screen(
    square: chess.Square,
    orientation: str,
    size: int,
    matrix_inv: np.ndarray,
) -> list[tuple[float, float]]:
    col, row = square_to_image_rc(square, orientation)
    tile = size / 8.0
    margin = tile * 0.06
    x0, y0 = col * tile + margin, row * tile + margin
    x1, y1 = (col + 1) * tile - margin, (row + 1) * tile - margin
    points = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points, matrix_inv)
    return [(float(x), float(y)) for x, y in mapped[0]]


def board_mean_abs_diff(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None or previous.shape != current.shape:
        return 255.0
    delta = cv2.absdiff(previous, current)
    return float(np.mean(delta))
