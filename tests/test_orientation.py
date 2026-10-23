from __future__ import annotations

import chess

from stockfish_vision.board.geometry import image_rc_to_square, square_to_image_rc


def test_white_top_is_black_perspective() -> None:
    assert image_rc_to_square(0, 0, "white_top") == chess.H1
    assert image_rc_to_square(7, 7, "white_top") == chess.A8
    assert square_to_image_rc(chess.A8, "white_top") == (7, 7)


def test_white_left_and_right() -> None:
    assert square_to_image_rc(chess.A1, "white_left") == (0, 0)
    assert square_to_image_rc(chess.A1, "white_right") == (7, 7)
    assert image_rc_to_square(0, 0, "white_left") == chess.A1
    assert image_rc_to_square(0, 0, "white_right") == chess.H8
