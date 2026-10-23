from __future__ import annotations

import cv2
import numpy as np

from stockfish_vision.board.detect import detect_board_corners, rectangle_to_corners
from stockfish_vision.board.geometry import validate_corners


def _board_image(
    size: int = 480,
    background: int = 30,
    border: bool = True,
    canvas: tuple[int, int] = (700, 900),
) -> np.ndarray:
    tile = size // 8
    layer = np.full((size, size, 3), background, dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            color = (235, 205, 160) if (row + col) % 2 == 0 else (120, 80, 55)
            cv2.rectangle(
                layer,
                (col * tile, row * tile),
                ((col + 1) * tile - 1, (row + 1) * tile - 1),
                color,
                -1,
            )
    if border:
        cv2.rectangle(layer, (0, 0), (size - 1, size - 1), (10, 10, 10), 4)
    height, width = canvas
    image = np.full((height, width, 3), background, dtype=np.uint8)
    return image, layer


def _place(size: int = 480, offset: tuple[int, int] = (120, 60)) -> np.ndarray:
    image, layer = _board_image(size=size)
    x0, y0 = offset
    image[y0 : y0 + size, x0 : x0 + size] = layer
    return image


def test_detect_board_corners_finds_axis_aligned_board() -> None:
    image = _place()
    corners = detect_board_corners(image)
    assert corners is not None
    expected = np.array([[120, 60], [600, 60], [600, 540], [120, 540]], dtype=float)
    error = np.max(np.abs(np.array(corners) - expected))
    assert error < 12.0, corners
    validate_corners(corners)


def test_detect_board_corners_returns_ordered_corners() -> None:
    corners = detect_board_corners(_place(size=320, offset=(60, 40)))
    assert corners is not None
    top_left, top_right, bottom_right, bottom_left = corners
    assert top_left[0] < top_right[0]
    assert top_left[1] < bottom_left[1]
    assert bottom_right[1] > top_right[1]


def test_detect_board_corners_with_noise() -> None:
    image = _place().astype(np.int16)
    rng = np.random.default_rng(7)
    noisy = np.clip(image + rng.normal(0, 9, image.shape), 0, 255).astype(np.uint8)
    corners = detect_board_corners(noisy)
    assert corners is not None
    validate_corners(corners)


def test_detect_board_corners_returns_none_for_empty_image() -> None:
    assert detect_board_corners(np.full((400, 500, 3), 32, dtype=np.uint8)) is None


def test_detect_board_corners_rejects_invalid_input() -> None:
    assert detect_board_corners(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert detect_board_corners(np.zeros((40, 40, 3), dtype=np.uint8)) is None


def test_rectangle_to_corners_orders_points() -> None:
    corners = rectangle_to_corners(300, 250, 100, 50)
    assert corners == [[100.0, 50.0], [300.0, 50.0], [300.0, 250.0], [100.0, 250.0]]
    validate_corners(corners)
