from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn
import pytest

from stockfish_vision.chesslogic.fenutil import load_fen, save_fen, update_board_state
from stockfish_vision.chesslogic.pgnutil import (
    game_from_board,
    game_is_completed,
    load_pgn_games,
    save_pgn_game,
)


def test_fen_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "position.fen"
    board = chess.Board()
    board.push_san("e4")
    save_fen(path, board)
    loaded = load_fen(path)
    assert loaded.fen() == board.fen()
    assert loaded.turn == chess.BLACK


def test_fen_file_rejects_multiple_positions(tmp_path: Path) -> None:
    path = tmp_path / "invalid.fen"
    path.write_text(f"{chess.STARTING_FEN}\n{chess.STARTING_FEN}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="genau eine"):
        load_fen(path)


def test_update_unseen_fen_state() -> None:
    board = update_board_state(
        chess.Board(), turn=chess.BLACK, castling_fen="Kq", ep_fen="e3"
    )
    assert board.turn == chess.BLACK
    assert board.castling_xfen() == "Kq"
    assert board.ep_square == chess.E3


def test_pgn_export_import_and_custom_position(tmp_path: Path) -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    game = game_from_board(board)
    assert game.headers["SetUp"] == "1"
    assert game.headers["FEN"] == board.fen()
    path = tmp_path / "training.pgn"
    save_pgn_game(path, game)
    loaded = load_pgn_games(path)
    assert loaded[0].board().fen() == board.fen()
    assert not game_is_completed(loaded[0])


def test_completed_pgn_result() -> None:
    game = chess.pgn.Game()
    game.headers["Result"] = "1/2-1/2"
    assert game_is_completed(game)
    game.headers["Result"] = "*"
    assert not game_is_completed(game)
