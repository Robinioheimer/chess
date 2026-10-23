from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn

COMPLETED_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def load_pgn_games(path: Path, limit: int = 50) -> list[chess.pgn.Game]:
    if limit < 1:
        raise ValueError("Das PGN-Limit muss mindestens 1 sein.")
    games: list[chess.pgn.Game] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            while len(games) < limit:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                if game.errors:
                    raise ValueError(f"Fehler in Partie {len(games) + 1}: {game.errors[0]}")
                games.append(game)
    except OSError as exc:
        raise ValueError(f"PGN-Datei konnte nicht gelesen werden: {exc}") from exc
    if not games:
        raise ValueError("Die PGN-Datei enthält keine lesbaren Partien.")
    return games


def game_positions(game: chess.pgn.Game) -> list[tuple[str, str, chess.Board]]:
    """Return (san, comment, board_after_move) for the mainline, including start."""
    boards: list[tuple[str, str, chess.Board]] = [("Start", game.comment, game.board())]
    node = game
    while node.variations:
        node = node.variation(0)
        move = node.move
        parent_board = node.parent.board() if node.parent else game.board()
        san = parent_board.san(move)
        boards.append((san, node.comment, node.board()))
    return boards


def game_is_completed(game: chess.pgn.Game) -> bool:
    return game.headers.get("Result", "*") in COMPLETED_RESULTS


def game_label(game: chess.pgn.Game, index: int = 1) -> str:
    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")
    result = game.headers.get("Result", "*")
    suffix = "abgeschlossen" if game_is_completed(game) else "nicht abgeschlossen"
    return f"{index}: {white} – {black} ({result}, {suffix})"


def game_from_board(board: chess.Board) -> chess.pgn.Game:
    """Create a PGN with a SetUp/FEN header for the current training position."""
    game = chess.pgn.Game()
    game.headers["Event"] = "Stockfish Vision Training"
    game.headers["Site"] = "Lokale Analyse"
    game.headers["Result"] = "*"
    if board.fen() != chess.Board().fen():
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "Stockfish Vision Training"
        game.headers["Site"] = "Lokale Analyse"
        game.headers["Result"] = "*"
    return game


def save_pgn_game(path: Path, game: chess.pgn.Game) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            exporter = chess.pgn.FileExporter(handle)
            game.accept(exporter)
            handle.write("\n")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"PGN-Datei konnte nicht geschrieben werden: {exc}") from exc
