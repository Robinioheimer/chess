"""Figurensatz einmalig anlernen.

Der mitgelieferte Klassifikator kennt nur synthetische Glyphen. Echte
Screenshots (z. B. chess.com) verwenden einen völlig anderen Figurensatz.
Statt ein generisches Modell zu trainieren, lernt das Programm den
tatsächlich sichtbaren Satz direkt aus einer bekannten Stellung – in der
Regel der Grundstellung. Aus jedem Feld wird die Figur freigestellt und als
Vorlage gespeichert; die Erkennung vergleicht danach genau diese Vorlagen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chess

from stockfish_vision.board.geometry import extract_square, square_to_image_rc
from stockfish_vision.recognition.pieces import TemplateBank, default_template_path

__all__ = [
    "LEARNABLE_POSITIONS",
    "LearningReport",
    "learn_templates",
    "standard_start_symbols",
    "template_path",
]

logger = logging.getLogger(__name__)

START_FEN = chess.STARTING_FEN

LEARNABLE_POSITIONS: tuple[tuple[str, str], ...] = (
    (
        "Grundstellung (alle 12 Figuren)",
        START_FEN,
    ),
)


def template_path():
    return default_template_path()


def standard_start_symbols() -> dict[chess.Square, str]:
    """Feld → Figurensymbol der Grundstellung."""

    board = chess.Board(START_FEN)
    return {
        square: piece.symbol()
        for square, piece in board.piece_map().items()
    }


@dataclass(frozen=True)
class LearningReport:
    samples: int
    per_symbol: dict[str, int]
    skipped: int
    path: str = ""

    @property
    def symbols(self) -> int:
        return len(self.per_symbol)

    def summary(self) -> str:
        parts = ", ".join(f"{symbol}×{count}" for symbol, count in sorted(self.per_symbol.items()))
        return f"{self.samples} Vorlagen aus {self.symbols} Figurenklassen ({parts})"


def learn_templates(
    warped,
    orientation: str,
    symbols: dict[chess.Square, str] | None = None,
    *,
    bank: TemplateBank | None = None,
    persist: bool = True,
) -> LearningReport:
    """Lernt Figurenvorlagen aus einem entzerrten Brettbild.

    ``symbols`` beschreibt die bekannte Stellung (Standard: Grundstellung).
    ``bank`` erlaubt das Ergänzen einer bestehenden Sammlung.
    """

    if warped is None or getattr(warped, "size", 0) == 0:
        raise ValueError("Kein Brettbild zum Anlernen vorhanden.")
    known = dict(symbols) if symbols is not None else standard_start_symbols()
    bank = bank if bank is not None else TemplateBank()

    counts: dict[str, int] = {}
    skipped = 0
    for square, symbol in known.items():
        try:
            col, row = square_to_image_rc(square, orientation)
            tile = extract_square(warped, col, row)
        except (ValueError, IndexError):
            skipped += 1
            continue
        if bank.add(symbol, tile):
            counts[symbol] = counts.get(symbol, 0) + 1
        else:
            skipped += 1

    path = ""
    if persist and counts:
        try:
            path = str(bank.save())
        except OSError as exc:
            logger.warning("Figurenvorlagen konnten nicht gespeichert werden: %s", exc)

    logger.info(
        "Figurensatz angelernt: %s Vorlagen, %s Klassen, %s übersprungen",
        sum(counts.values()),
        len(counts),
        skipped,
    )
    return LearningReport(
        samples=sum(counts.values()),
        per_symbol=counts,
        skipped=skipped,
        path=path,
    )
