from __future__ import annotations

import random

import chess
import cv2
import numpy as np

from stockfish_vision.board.geometry import square_to_image_rc
from stockfish_vision.recognition.glyphs import render_piece_glyph
from stockfish_vision.recognition.pieces import PieceClassifier


def test_reconstructs_valid_board_from_synthetic_tiles() -> None:
    tile_size = 48
    warped = np.zeros((tile_size * 8, tile_size * 8, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            light = (row + col) % 2 == 0
            rgb = np.array((225, 205, 165) if light else (110, 80, 50), dtype=np.uint8)
            tile = np.full((tile_size, tile_size, 3), rgb, dtype=np.uint8)
            warped[
                row * tile_size : (row + 1) * tile_size,
                col * tile_size : (col + 1) * tile_size,
            ] = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    expected = {chess.E1: "K", chess.E8: "k", chess.D4: "Q"}
    for square, symbol in expected.items():
        col, row = square_to_image_rc(square, "white_bottom")
        tile = render_piece_glyph(
            symbol,
            size=tile_size,
            light_square=(row + col) % 2 == 0,
            jitter=False,
            rng=random.Random(square),
        )
        warped[
            row * tile_size : (row + 1) * tile_size,
            col * tile_size : (col + 1) * tile_size,
        ] = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

    board, detections, validity = PieceClassifier().detect_board(warped, "white_bottom")
    assert validity == (True, [])
    assert len(board.piece_map()) == 3
    for square, symbol in expected.items():
        assert board.piece_at(square).symbol() == symbol
        detection = next(item for item in detections if item.square == square)
        assert detection.occupied
        assert detection.confidence > 0.45
        assert detection.alternatives
    assert not [item for item in detections if item.occupied and item.piece is None]
