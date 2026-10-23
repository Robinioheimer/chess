from __future__ import annotations

from datetime import datetime, timedelta

import chess
import numpy as np
import pytest

from stockfish_vision.board.geometry import perspective_matrix, square_polygon_screen
from stockfish_vision.capture.screen import validate_region
from stockfish_vision.overlay.hotkey import parse_hotkey
from stockfish_vision.policy import LiveGameGuard, PositionChange


def test_invalid_calibration_is_rejected() -> None:
    with pytest.raises(ValueError, match="verschieden"):
        perspective_matrix([[0, 0], [0, 0], [100, 100], [0, 100]])
    with pytest.raises(ValueError, match="konvexes Viereck"):
        perspective_matrix([[0, 0], [100, 100], [100, 0], [0, 100]])


def test_square_polygon_perspective_roundtrip() -> None:
    matrix = perspective_matrix([[0, 0], [399, 0], [399, 399], [0, 399]], 400)
    polygon = square_polygon_screen(chess.A8, "white_bottom", 400, np.linalg.inv(matrix))
    assert len(polygon) == 4
    assert all(0 <= x < 50 and 0 <= y < 50 for x, y in polygon)


def test_capture_region_validation_allows_negative_monitor_origin() -> None:
    region = validate_region({"left": -1920, "top": 0, "width": 1920, "height": 1080})
    assert region["left"] == -1920
    with pytest.raises(ValueError, match="mindestens"):
        validate_region({"left": 0, "top": 0, "width": 2, "height": 100})


def test_hotkey_rejects_invalid_combinations() -> None:
    with pytest.raises(ValueError, match="Modifikator"):
        parse_hotkey("f8")
    with pytest.raises(ValueError, match="Unbekannter"):
        parse_hotkey("ctrl+f25")
    with pytest.raises(ValueError, match="nur eine"):
        parse_hotkey("ctrl+a+b")


def test_live_guard_expires_old_changes() -> None:
    guard = LiveGameGuard()
    old = datetime.now() - timedelta(seconds=30)
    guard.history = [PositionChange(f"pos{index}", old) for index in range(3)]
    assert not guard.looks_like_live_game()
    assert guard.history == []
