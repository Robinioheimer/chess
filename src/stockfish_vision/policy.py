from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

ALLOWED_MODES = frozenset({"training", "pgn_completed", "frozen_snapshot"})

LIVE_GAME_BLOCK_MESSAGE = (
    "Live-Engine-Hilfe in laufenden Online-Partien ist deaktiviert. "
    "Analyse ist nur für eigene Trainingsstellungen, eingefrorene Snapshots "
    "und abgeschlossene Partien (PGN) verfügbar."
)

CONFIRMATION_REQUIRED = (
    "Bitte bestätigen Sie, dass es sich um eine eigene Trainingsstellung "
    "oder eine abgeschlossene Partie handelt – nicht um eine laufende Online-Partie."
)

RAPID_CHANGE_MESSAGE = (
    "Das Brett hat sich mehrfach in kurzer Zeit verändert. "
    "Das sieht nach einer laufenden Partie aus; die Engine bleibt gesperrt. "
    "Nehmen Sie nach einer Ruhepause einen neuen Snapshot einer Trainingsstellung."
)


@dataclass
class PositionChange:
    fen: str
    at: datetime


@dataclass
class LiveGameGuard:
    """Conservative guard for frozen snapshots; no live engine automation exists."""

    window: timedelta = timedelta(seconds=12)
    max_changes: int = 3
    history: list[PositionChange] = field(default_factory=list)

    def _prune(self, now: datetime) -> None:
        cutoff = now - self.window
        self.history = [item for item in self.history if item.at >= cutoff]

    def record(self, fen: str) -> None:
        now = datetime.now()
        self._prune(now)
        if self.history and self.history[-1].fen == fen:
            return
        self.history.append(PositionChange(fen=fen, at=now))

    def looks_like_live_game(self) -> bool:
        self._prune(datetime.now())
        unique = {item.fen.split(" ")[0] for item in self.history}
        return len(unique) >= self.max_changes

    def reset(self) -> None:
        self.history.clear()


def can_analyze(
    mode: str,
    *,
    snapshot_confirmed: bool = False,
    live_guard: LiveGameGuard | None = None,
) -> tuple[bool, str]:
    if mode not in ALLOWED_MODES:
        return False, LIVE_GAME_BLOCK_MESSAGE
    if mode == "frozen_snapshot":
        if not snapshot_confirmed:
            return False, CONFIRMATION_REQUIRED
        if live_guard is not None and live_guard.looks_like_live_game():
            return False, RAPID_CHANGE_MESSAGE
    return True, ""
