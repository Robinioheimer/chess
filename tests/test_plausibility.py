from __future__ import annotations

import chess

from stockfish_vision.recognition.plausibility import MAX_PAWNS, repair_symbols


def _squares_with(symbols: dict[int, str | None], symbol: str) -> list[int]:
    return sorted(square for square, value in symbols.items() if value == symbol)


def test_two_white_kings_are_reduced_to_one() -> None:
    symbols = {chess.E1: "K", chess.G1: "K", chess.E8: "k"}
    scores = {
        chess.E1: {"K": 0.82},
        chess.G1: {"K": 0.71, "R": 0.2},
        chess.E8: {"k": 0.9},
    }
    repaired, corrections = repair_symbols(symbols, scores)

    assert _squares_with(repaired, "K") == [chess.E1]
    assert repaired[chess.G1] == "R"
    assert len(corrections) == 1
    correction = corrections[0]
    assert correction.square == chess.G1
    assert correction.previous == "K"
    assert correction.current == "R"
    assert "König" in correction.description
    assert "g1" in correction.description


def test_home_square_king_wins_even_with_lower_score() -> None:
    symbols = {chess.E1: "K", chess.D1: "K"}
    scores = {chess.E1: {"K": 0.60}, chess.D1: {"K": 0.78}}
    repaired, corrections = repair_symbols(symbols, scores)

    assert _squares_with(repaired, "K") == [chess.E1]
    assert repaired[chess.D1] in {"Q", "R", "B", "N"}
    assert len(corrections) == 1


def test_home_rank_king_wins_on_equal_score() -> None:
    symbols = {chess.E1: "K", chess.D4: "K"}
    scores = {chess.E1: {"K": 0.5}, chess.D4: {"K": 0.5}}
    repaired, corrections = repair_symbols(symbols, scores)

    assert _squares_with(repaired, "K") == [chess.E1]
    assert repaired[chess.D4] == "Q"  # ohne Alternative bleibt eine Dame
    assert corrections[0].square == chess.D4


def test_extra_kings_without_alternative_become_queen_not_invalid() -> None:
    symbols = {chess.E8: "k", chess.D8: "k", chess.A8: "k"}
    scores = {square: {"k": 0.5} for square in symbols}
    repaired, corrections = repair_symbols(symbols, scores)

    assert _squares_with(repaired, "k") == [chess.E8]
    assert repaired[chess.D8] == "q"
    assert repaired[chess.A8] == "q"
    assert len(corrections) == 2


def test_pawn_on_back_rank_is_replaced() -> None:
    symbols = {chess.A1: "P", chess.E1: "K", chess.E8: "k"}
    scores = {chess.A1: {"P": 0.6, "R": 0.3}}
    repaired, corrections = repair_symbols(symbols, scores)

    assert repaired[chess.A1] == "R"
    assert corrections[0].reason == "Bauer auf der Grundreihe"


def test_more_than_eight_pawns_loses_the_weakest() -> None:
    files = [0, 1, 2, 3, 4, 5, 6, 7, 0]
    ranks = [1] * 8 + [2]
    squares = [chess.square(file, rank) for file, rank in zip(files, ranks, strict=True)]
    symbols = {square: "P" for square in squares}
    symbols[chess.E1] = "K"
    symbols[chess.E8] = "k"
    scores = {square: {"P": 0.5 + index * 0.02} for index, square in enumerate(squares)}

    repaired, corrections = repair_symbols(symbols, scores)

    pawns = _squares_with(repaired, "P")
    assert len(pawns) == MAX_PAWNS
    # Der unsicherste Bauer (A2, niedrigster Score) verschwindet.
    assert chess.A2 not in pawns
    assert len(corrections) == 1
    assert corrections[0].square == chess.A2


def test_valid_position_is_unchanged() -> None:
    board = chess.Board()
    symbols = {
        square: (board.piece_at(square).symbol() if board.piece_at(square) else None)
        for square in chess.SQUARES
    }
    repaired, corrections = repair_symbols(symbols, {})

    assert corrections == []
    assert repaired == symbols


def test_black_and_white_piece_counts_are_independent() -> None:
    squares = [chess.square(file, 1) for file in range(8)] + [
        chess.square(file, 6) for file in range(8)
    ]
    symbols = {square: ("P" if chess.square_rank(square) == 1 else "p") for square in squares}
    symbols[chess.E1] = "K"
    symbols[chess.E8] = "k"
    symbols[chess.A3] = "P"  # neunter weißer Bauer
    scores = {square: {symbol: 0.4 + index * 0.01} for index, (square, symbol) in enumerate(symbols.items())}

    repaired, corrections = repair_symbols(symbols, scores)

    assert len(_squares_with(repaired, "P")) == MAX_PAWNS
    assert len(_squares_with(repaired, "p")) == 8
    assert len(corrections) == 1
