from __future__ import annotations

import chess


def parse_fen(fen: str) -> chess.Board:
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Ungültige FEN: {exc}") from exc
    return board


def fen_from_board(board: chess.Board) -> str:
    return board.fen()


def board_from_piece_map(
    pieces: dict[chess.Square, chess.Piece | None],
    turn: chess.Color = chess.WHITE,
    castling_fen: str = "-",
    ep_square: chess.Square | None = None,
) -> chess.Board:
    board = chess.Board(None)
    board.clear_board()
    for square, piece in pieces.items():
        if piece is not None:
            board.set_piece_at(square, piece)
    board.turn = turn
    try:
        board.set_castling_fen(castling_fen)
    except ValueError:
        board.castling_rights = 0
    board.ep_square = ep_square
    board.fullmove_number = 1
    board.halfmove_clock = 0
    return board


def describe_validity(board: chess.Board) -> tuple[bool, list[str]]:
    issues: list[str] = []
    white_kings = list(board.pieces(chess.KING, chess.WHITE))
    black_kings = list(board.pieces(chess.KING, chess.BLACK))
    if len(white_kings) != 1:
        issues.append(f"Weißer König: {len(white_kings)} statt 1.")
    if len(black_kings) != 1:
        issues.append(f"Schwarzer König: {len(black_kings)} statt 1.")
    if board.occupied_co[chess.WHITE] == 0:
        issues.append("Keine weißen Figuren erkannt.")
    if board.occupied_co[chess.BLACK] == 0:
        issues.append("Keine schwarzen Figuren erkannt.")
    if not issues:
        try:
            if not board.is_valid():
                issues.append(
                    "python-chess bewertet die Stellung als ungültig "
                    "(z. B. König im Schach beim Gegner am Zug, zu viele Bauern)."
                )
        except ValueError as exc:
            issues.append(str(exc))
    return len(issues) == 0, issues


def apply_manual_piece(
    board: chess.Board, square: chess.Square, symbol: str | None
) -> chess.Board:
    updated = board.copy(stack=False)
    if not symbol:
        updated.remove_piece_at(square)
        return updated
    piece = chess.Piece.from_symbol(symbol)
    updated.set_piece_at(square, piece)
    return updated
