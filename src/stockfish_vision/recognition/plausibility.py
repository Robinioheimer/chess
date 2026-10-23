"""Plausibilitätskorrektur für erkannte Stellungen.

Die pixelbasierte Erkennung klassifiziert jedes Feld unabhängig. Dadurch
entstehen gelegentlich Stellungen, die es im Schach nicht geben kann – etwa
zwei weiße Könige oder neun Bauern einer Farbe. Solche Ausreißer werden hier
mit der jeweils zweitbesten Klassifikation repariert, damit die Stellung
gültig bleibt und die Engine sie analysieren kann. Jede Änderung wird als
:class:`Correction` zurückgegeben und in der Oberfläche angezeigt.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

__all__ = ["Correction", "MAX_PAWNS", "MAX_PIECES", "repair_symbols"]

MAX_PAWNS = 8
MAX_PIECES = 16

BACK_RANKS = {chess.WHITE: chess.BB_RANK_1, chess.BLACK: chess.BB_RANK_8}
HOME_KING = {chess.WHITE: chess.E1, chess.BLACK: chess.E8}


@dataclass(frozen=True)
class Correction:
    """Eine automatisch korrigierte Felderkennung."""

    square: int
    previous: str
    current: str | None
    reason: str

    @property
    def description(self) -> str:
        name = chess.square_name(self.square)
        target = self.current or "leer"
        return f"{name}: {self.previous} → {target} ({self.reason})"


def _color_of(symbol: str) -> chess.Color:
    return chess.WHITE if symbol.isupper() else chess.BLACK


def _is_king(symbol: str) -> bool:
    return symbol.lower() == "k"


def _is_pawn(symbol: str) -> bool:
    return symbol.lower() == "p"


def _score(symbols: dict[int, str | None], scores: dict[int, dict[str, float]], square: int) -> float:
    symbol = symbols.get(square)
    if symbol is None:
        return 0.0
    return float((scores.get(square) or {}).get(symbol, 0.0))


def _best_alternative(
    symbol: str,
    scores: dict[str, float] | None,
    *,
    forbid_king: bool = False,
    forbid_pawn: bool = False,
) -> str | None:
    """Wählt die beste Alternative gleicher Farbe, die die Regeln nicht verletzt."""

    best: str | None = None
    best_score = -1.0
    for candidate, value in (scores or {}).items():
        if len(candidate) != 1 or not candidate.isalpha():
            continue
        if candidate.isupper() != symbol.isupper():
            continue
        if forbid_king and _is_king(candidate):
            continue
        if forbid_pawn and _is_pawn(candidate):
            continue
        if float(value) <= best_score:
            continue
        best, best_score = candidate, float(value)
    return best


def _replace(
    symbols: dict[int, str | None],
    corrections: list[Correction],
    square: int,
    replacement: str | None,
    reason: str,
) -> None:
    previous = symbols[square]
    if previous is None or previous == replacement:
        return
    symbols[square] = replacement
    corrections.append(Correction(square, previous, replacement, reason))


def repair_symbols(
    symbols: dict[int, str | None],
    scores: dict[int, dict[str, float]] | None = None,
) -> tuple[dict[int, str | None], list[Correction]]:
    """Erzwingt die Grundregeln der Stellung und liefert die Änderungen.

    ``symbols`` bildet Felder auf Figurensymbole (``"K"``, ``"p"`` …) oder
    ``None`` ab, ``scores`` enthält die Klassifikations-Scores je Feld.
    """
    scores = scores or {}
    repaired = dict(symbols)
    corrections: list[Correction] = []

    # 1. Höchstens ein König je Farbe. Der König auf dem Stammfeld (e1/e8) bzw.
    #    der eigenen Grundreihe gewinnt vor dem reinen Score – ein zweiter König
    #    entsteht fast immer aus einer Fehlklassifikation auf einem Nachbarfeld.
    for color in (chess.WHITE, chess.BLACK):
        kings = [
            square
            for square, symbol in repaired.items()
            if symbol is not None and _is_king(symbol) and _color_of(symbol) == color
        ]
        if len(kings) <= 1:
            continue

        def _rank(square: int) -> tuple[int, int, float]:
            on_home = 1 if chess.BB_SQUARES[square] & BACK_RANKS[color] else 0
            near_home = 1 if square == HOME_KING[color] else 0
            return (near_home, on_home, _score(repaired, scores, square))

        kings.sort(key=_rank, reverse=True)
        for square in kings[1:]:
            symbol = repaired[square]
            assert symbol is not None
            replacement = _best_alternative(
                symbol, scores.get(square), forbid_king=True, forbid_pawn=True
            )
            reason = "nur ein König je Seite"
            if replacement is None:
                # Eine Dame ist materiell immer möglich und erzeugt keinen neuen Fehler.
                replacement = "Q" if color == chess.WHITE else "q"
                reason = "nur ein König je Seite, Ersatz als Dame"
            _replace(repaired, corrections, square, replacement, reason)

    # 2. Keine Bauern auf der eigenen Grundreihe – dort steht eine Umwandlung.
    for square, symbol in sorted(repaired.items()):
        if symbol is None or not _is_pawn(symbol):
            continue
        if not chess.BB_SQUARES[square] & BACK_RANKS[_color_of(symbol)]:
            continue
        replacement = _best_alternative(
            symbol, scores.get(square), forbid_king=True, forbid_pawn=True
        )
        if replacement is None:
            replacement = "Q" if symbol.isupper() else "q"
        _replace(
            repaired,
            corrections,
            square,
            replacement,
            "Bauer auf der Grundreihe",
        )

    # 3. Höchstens 16 Figuren je Farbe; die unsicherste Figur weicht.
    for color in (chess.WHITE, chess.BLACK):
        pieces = [
            square
            for square, symbol in repaired.items()
            if symbol is not None and _color_of(symbol) == color
        ]
        if len(pieces) <= MAX_PIECES:
            continue
        pieces.sort(key=lambda square: _score(repaired, scores, square))
        for square in pieces[: len(pieces) - MAX_PIECES]:
            _replace(repaired, corrections, square, None, "mehr als 16 Figuren")

    # 4. Höchstens acht Bauern je Farbe; der unsicherste Bauer weicht.
    for color in (chess.WHITE, chess.BLACK):
        pawns = [
            square
            for square, symbol in repaired.items()
            if symbol is not None and _is_pawn(symbol) and _color_of(symbol) == color
        ]
        if len(pawns) <= MAX_PAWNS:
            continue
        pawns.sort(key=lambda square: _score(repaired, scores, square))
        for square in pawns[: len(pawns) - MAX_PAWNS]:
            symbol = repaired[square]
            assert symbol is not None
            replacement = _best_alternative(
                symbol, scores.get(square), forbid_king=True, forbid_pawn=True
            )
            _replace(
                repaired,
                corrections,
                square,
                replacement,
                "mehr als acht Bauern",
            )

    return repaired, corrections
