from __future__ import annotations

import os

import chess
import pytest

from stockfish_vision.config import EngineSettings
from stockfish_vision.engine.client import EngineClient


@pytest.mark.skipif(
    not os.environ.get("STOCKFISH_PATH"),
    reason="Optionale echte Engine: STOCKFISH_PATH ist nicht gesetzt",
)
def test_official_stockfish_uci_multipv() -> None:
    client = EngineClient()
    settings = EngineSettings(depth=5, move_time_ms=100, threads=1, hash_mb=32, multipv=2)
    try:
        name = client.start(settings)
        assert "stockfish" in name.lower()
        board = chess.Board()
        result = client.analyse(board, settings)
        assert len(result.lines) == 2
        for line in result.lines:
            assert line.pv_uci
            assert chess.Move.from_uci(line.pv_uci[0]) in board.legal_moves
    finally:
        client.stop()
