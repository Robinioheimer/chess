from __future__ import annotations

import chess
import numpy as np

from stockfish_vision.board.geometry import (
    extract_square,
    image_rc_to_square,
    inverse_matrix,
    perspective_matrix,
    square_center_image,
    square_center_screen,
    square_to_image_rc,
    warp_board,
)


def test_square_mapping_white_bottom() -> None:
    assert image_rc_to_square(0, 0, "white_bottom") == chess.A8
    assert image_rc_to_square(7, 7, "white_bottom") == chess.H1
    assert square_to_image_rc(chess.A1, "white_bottom") == (0, 7)
    assert square_to_image_rc(chess.H8, "white_bottom") == (7, 0)


def test_square_mapping_roundtrip() -> None:
    for orientation in ("white_bottom", "white_top", "white_left", "white_right"):
        for square in chess.SQUARES:
            col, row = square_to_image_rc(square, orientation)
            assert image_rc_to_square(col, row, orientation) == square


def test_perspective_and_square_centers() -> None:
    size = 400
    corners = [[0, 0], [size, 0], [size, size], [0, size]]
    matrix = perspective_matrix(corners, size)
    matrix_inv = inverse_matrix(matrix)
    x, y = square_center_image(chess.A8, "white_bottom", size)
    assert 20 < x < 30
    assert 20 < y < 30
    sx, sy = square_center_screen(chess.H1, "white_bottom", size, matrix_inv)
    assert 370 < sx < 380
    assert 370 < sy < 380


def test_warp_splits_64_squares() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :] = (40, 80, 120)
    corners = [[0, 0], [199, 0], [199, 199], [0, 199]]
    warped, _ = warp_board(image, corners, 160)
    assert warped.shape == (160, 160, 3)
    tile = extract_square(warped, 3, 4)
    assert tile.shape[0] > 10
    assert tile.shape[1] > 10
