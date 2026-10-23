from __future__ import annotations

import chess
import chess.engine


def hanging_pieces(board: chess.Board) -> list[str]:
    labels: list[str] = []
    for square, piece in board.piece_map().items():
        attacked = board.is_attacked_by(not piece.color, square)
        defended = board.is_attacked_by(piece.color, square)
        if attacked and not defended and piece.piece_type != chess.KING:
            labels.append(f"{piece.symbol()} auf {chess.square_name(square)}")
    return labels


def check_status(board: chess.Board) -> str:
    if board.is_checkmate():
        side = "Weiß" if board.turn == chess.BLACK else "Schwarz"
        return f"Matt – {side} gewinnt."
    if board.is_stalemate():
        return "Patt."
    if board.is_check():
        return "Schach."
    return "Kein Schach."


def format_score(pov_score: chess.engine.PovScore | None) -> str:
    if pov_score is None:
        return "–"
    score = pov_score.white()
    mate = score.mate()
    if mate is not None:
        sign = "+" if mate > 0 else ""
        return f"M{sign}{mate}"
    cp = score.score()
    if cp is None:
        return "–"
    return f"{cp / 100.0:+.2f}"


def pv_to_san(board: chess.Board, moves: list[chess.Move]) -> str:
    clone = board.copy()
    sans: list[str] = []
    for move in moves:
        if move not in clone.legal_moves:
            sans.append(move.uci())
            break
        sans.append(clone.san(move))
        clone.push(move)
    return " ".join(sans)
