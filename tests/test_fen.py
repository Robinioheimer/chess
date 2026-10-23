from __future__ import annotations

import chess

from stockfish_vision.chesslogic.fenutil import (
    apply_manual_piece,
    board_from_piece_map,
    describe_validity,
    fen_from_board,
    parse_fen,
)


def test_parse_start_fen() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = parse_fen(fen)
    assert fen_from_board(board).startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    assert board.piece_at(chess.E1) == chess.Piece.from_symbol("K")


def test_invalid_fen_raises() -> None:
    try:
        parse_fen("not-a-fen")
    except ValueError as exc:
        assert "Ungültige FEN" in str(exc)
    else:
        raise AssertionError("ValueError erwartet")


def test_manual_correction_roundtrip() -> None:
    board = chess.Board()
    board = apply_manual_piece(board, chess.E2, None)
    board = apply_manual_piece(board, chess.E4, "P")
    assert board.piece_at(chess.E2) is None
    assert board.piece_at(chess.E4).symbol() == "P"


def test_validity_missing_kings() -> None:
    board = board_from_piece_map({chess.A1: chess.Piece.from_symbol("R")})
    ok, issues = describe_validity(board)
    assert not ok
    assert any("König" in item for item in issues)


def test_valid_start_position() -> None:
    ok, issues = describe_validity(chess.Board())
    assert ok
    assert issues == []
