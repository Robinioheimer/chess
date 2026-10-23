from __future__ import annotations

import sys
from pathlib import Path

import chess

from stockfish_vision.config import EngineSettings
from stockfish_vision.engine.client import EngineClient

FAKE = Path(__file__).resolve().parent / "fake_uci_engine.py"


def test_fake_uci_bestmove_and_multipv() -> None:
    client = EngineClient()
    settings = EngineSettings(depth=8, move_time_ms=0, threads=1, hash_mb=16, multipv=2)
    try:
        name = client.start(settings, command=[sys.executable, "-u", str(FAKE)])
        assert "FakeVisionEngine" in name
        board = chess.Board()
        result = client.analyse(board, settings)
        assert result.lines
        assert result.lines[0].best_move_uci == "e2e4"
        assert "e4" in result.lines[0].best_move_san
        assert result.lines[0].pv_san.startswith("e4")
        assert len(result.lines) >= 1
        assert result.check == "Kein Schach."
    finally:
        client.stop()


def test_missing_binary_message() -> None:
    client = EngineClient()
    try:
        client.resolve_executable("C:/definitely-missing-stockfish-xyz.exe")
    except FileNotFoundError as exc:
        assert "Stockfish nicht gefunden" in str(exc)
    else:
        raise AssertionError("FileNotFoundError erwartet")
