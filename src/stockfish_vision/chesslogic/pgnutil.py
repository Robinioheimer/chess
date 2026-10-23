from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn


def load_pgn_games(path: Path, limit: int = 50) -> list[chess.pgn.Game]:
    games: list[chess.pgn.Game] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while len(games) < limit:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games.append(game)
    if not games:
        raise ValueError("Die PGN-Datei enthält keine lesbaren Partien.")
    return games


def game_positions(game: chess.pgn.Game) -> list[tuple[str, str, chess.Board]]:
    """Return (san, comment, board_after_move) including the start position."""
    boards: list[tuple[str, str, chess.Board]] = [("Start", "", game.board())]
    node = game
    while node.variations:
        node = node.variation(0)
        move = node.move
        parent_board = node.parent.board() if node.parent else game.board()
        san = parent_board.san(move)
        boards.append((san, node.comment, node.board()))
    return boards
