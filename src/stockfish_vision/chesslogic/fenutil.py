from __future__ import annotations

from pathlib import Path

import chess


def parse_fen(fen: str) -> chess.Board:
    try:
        return chess.Board(fen.strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Ungültige FEN: {exc}") from exc


def load_fen(path: Path) -> chess.Board:
    try:
        content = path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise ValueError(f"FEN-Datei konnte nicht gelesen werden: {exc}") from exc
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError("Die FEN-Datei ist leer.")
    if len(lines) > 1:
        raise ValueError("Die FEN-Datei darf genau eine Stellung enthalten.")
    return parse_fen(lines[0])


def save_fen(path: Path, board: chess.Board) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(board.fen() + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"FEN-Datei konnte nicht geschrieben werden: {exc}") from exc


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


def update_board_state(
    board: chess.Board,
    *,
    turn: chess.Color,
    castling_fen: str = "-",
    ep_fen: str = "-",
) -> chess.Board:
    """Copy a board and update fields that cannot be recognised from pixels."""
    updated = board.copy(stack=False)
    updated.turn = turn
    try:
        updated.set_castling_fen(castling_fen.strip() or "-")
    except ValueError as exc:
        raise ValueError(f"Ungültige Rochaderechte: {exc}") from exc
    ep_fen = ep_fen.strip() or "-"
    if ep_fen == "-":
        updated.ep_square = None
    else:
        try:
            updated.ep_square = chess.parse_square(ep_fen.lower())
        except ValueError as exc:
            raise ValueError(f"Ungültiges En-passant-Feld: {ep_fen}") from exc
    return updated


_STATUS_MESSAGES = {
    chess.STATUS_NO_WHITE_KING: "Der weiße König fehlt.",
    chess.STATUS_NO_BLACK_KING: "Der schwarze König fehlt.",
    chess.STATUS_TOO_MANY_KINGS: "Es sind zu viele Könige auf dem Brett.",
    chess.STATUS_TOO_MANY_WHITE_PAWNS: "Weiß hat mehr als acht Bauern.",
    chess.STATUS_TOO_MANY_BLACK_PAWNS: "Schwarz hat mehr als acht Bauern.",
    chess.STATUS_PAWNS_ON_BACKRANK: "Bauern stehen auf der Grundreihe.",
    chess.STATUS_TOO_MANY_WHITE_PIECES: "Weiß hat zu viele Figuren.",
    chess.STATUS_TOO_MANY_BLACK_PIECES: "Schwarz hat zu viele Figuren.",
    chess.STATUS_BAD_CASTLING_RIGHTS: "Die Rochaderechte passen nicht zu König/Turm.",
    chess.STATUS_INVALID_EP_SQUARE: "Das En-passant-Feld ist ungültig.",
    chess.STATUS_OPPOSITE_CHECK: "Der Gegner wäre bereits im Schach, obwohl diese Seite am Zug ist.",
    chess.STATUS_EMPTY: "Das Brett ist leer.",
    chess.STATUS_RACE_CHECK: "Die Stellung enthält einen ungültigen Schachzustand.",
    chess.STATUS_RACE_OVER: "Die Stellung ist für die Engine bereits beendet.",
    chess.STATUS_RACE_MATERIAL: "Das Figurenmaterial ist ungültig.",
}


def describe_validity(board: chess.Board) -> tuple[bool, list[str]]:
    issues: list[str] = []
    status = board.status()
    for flag, message in _STATUS_MESSAGES.items():
        if status & flag and message not in issues:
            issues.append(message)
    if status == chess.STATUS_VALID:
        return True, []
    if not issues:
        issues.append("python-chess bewertet die Stellung als ungültig.")
    return False, issues


def apply_manual_piece(
    board: chess.Board, square: chess.Square, symbol: str | None
) -> chess.Board:
    updated = board.copy(stack=False)
    if not symbol:
        updated.remove_piece_at(square)
        return updated
    try:
        piece = chess.Piece.from_symbol(symbol)
    except ValueError as exc:
        raise ValueError(f"Unbekannte Figur: {symbol}") from exc
    updated.set_piece_at(square, piece)
    return updated
