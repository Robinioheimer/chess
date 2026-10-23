from __future__ import annotations

from pathlib import Path

from stockfish_vision.chesslogic.pgnutil import game_positions, load_pgn_games

PGN = Path(__file__).resolve().parent / "data" / "mini.pgn"


def test_pgn_positions_include_start_and_moves() -> None:
    games = load_pgn_games(PGN, limit=1)
    positions = game_positions(games[0])
    assert positions[0][0] == "Start"
    sans = [item[0] for item in positions[1:]]
    assert sans[0] == "e4"
    assert "O-O" in sans
    assert games[0].headers["White"] == "Alpha"
