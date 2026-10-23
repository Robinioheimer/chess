from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

from stockfish_vision.chesslogic.tactics import (
    check_status,
    format_score,
    hanging_pieces,
    pv_to_san,
)
from stockfish_vision.config import EngineSettings

logger = logging.getLogger(__name__)


@dataclass
class AnalysisLine:
    multipv: int
    score_text: str
    pv_san: str
    pv_uci: list[str]
    best_move_uci: str
    best_move_san: str
    depth: int | None = None
    seldepth: int | None = None
    nodes: int | None = None


@dataclass
class AnalysisResult:
    fen: str
    lines: list[AnalysisLine] = field(default_factory=list)
    check: str = ""
    hanging: list[str] = field(default_factory=list)
    engine_id: str = "Stockfish"


class EngineClient:
    def __init__(self) -> None:
        self._engine: chess.engine.SimpleEngine | None = None
        self.engine_id = ""
        self.command: list[str] = []

    def resolve_executable(self, configured: str) -> str:
        if configured:
            expanded = Path(os.path.expandvars(configured)).expanduser()
            if expanded.is_file():
                return str(expanded.resolve())
            raise FileNotFoundError(f"Stockfish nicht gefunden: {configured}")
        env = os.environ.get("STOCKFISH_PATH")
        if env:
            expanded_env = Path(os.path.expandvars(env)).expanduser()
            if expanded_env.is_file():
                return str(expanded_env.resolve())
        found = shutil.which("stockfish") or shutil.which("stockfish.exe")
        if found:
            return found
        common = [
            Path("C:/Program Files/Stockfish/stockfish.exe"),
            Path.home() / "stockfish" / "stockfish.exe",
            Path.cwd() / "stockfish.exe",
            Path.cwd() / "engines" / "stockfish.exe",
        ]
        for candidate in common:
            if candidate.is_file():
                return str(candidate.resolve())
        raise FileNotFoundError(
            "Keine Stockfish-Binary gefunden. Bitte unter Engine-Einstellungen "
            "den Pfad zur offiziellen stockfish.exe setzen. Bezugsquelle: "
            "https://stockfishchess.org/download/"
        )

    def is_running(self) -> bool:
        return self._engine is not None

    def start(self, settings: EngineSettings, command: list[str] | None = None) -> str:
        self.stop()
        if command is None:
            command = [self.resolve_executable(settings.executable)]
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("Der Engine-Kommandoaufruf ist ungültig.")
        self.command = list(command)
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(command)
            self.engine_id = self._engine.id.get("name", "UCI-Engine")
            options: dict[str, int] = {}
            available = self._engine.options
            if "Threads" in available:
                options["Threads"] = _option_value(available["Threads"], settings.threads, 1)
            if "Hash" in available:
                options["Hash"] = _option_value(available["Hash"], settings.hash_mb, 16)
            if options:
                self._engine.configure(options)
            logger.info("Engine gestartet: %s (%s)", self.engine_id, command)
            return self.engine_id
        except Exception:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            self.command = []
            raise

    def stop(self) -> None:
        engine, self._engine = self._engine, None
        self.command = []
        if engine is None:
            return
        try:
            engine.quit()
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError, OSError, TimeoutError):
            logger.warning("Engine war bereits beendet oder antwortete nicht; Prozess wird geschlossen.")
        finally:
            engine.close()

    def analyse(self, board: chess.Board, settings: EngineSettings) -> AnalysisResult:
        if self._engine is None:
            raise RuntimeError("Engine ist nicht gestartet.")
        if not board.is_valid():
            raise ValueError("Die Stellung ist für eine Engine-Analyse ungültig.")
        limit_kwargs: dict[str, float | int] = {}
        if settings.move_time_ms > 0:
            limit_kwargs["time"] = settings.move_time_ms / 1000.0
        if settings.depth > 0:
            limit_kwargs["depth"] = settings.depth
        if not limit_kwargs:
            limit_kwargs["depth"] = 12
        limit = chess.engine.Limit(**limit_kwargs)
        multipv = max(1, min(int(settings.multipv), 5))
        option = self._engine.options.get("MultiPV")
        if option is not None and option.max is not None:
            multipv = min(multipv, option.max)
        raw = self._engine.analyse(board, limit, multipv=multipv)
        infos = [raw] if isinstance(raw, dict) else list(raw)
        lines: list[AnalysisLine] = []
        for info in infos:
            pv = info.get("pv") or []
            score = info.get("score")
            best = pv[0] if pv else None
            san = board.san(best) if best and best in board.legal_moves else (best.uci() if best else "–")
            lines.append(
                AnalysisLine(
                    multipv=int(info.get("multipv", len(lines) + 1)),
                    score_text=format_score(score),
                    pv_san=pv_to_san(board, list(pv)),
                    pv_uci=[move.uci() for move in pv],
                    best_move_uci=best.uci() if best else "",
                    best_move_san=san,
                    depth=info.get("depth"),
                    seldepth=info.get("seldepth"),
                    nodes=info.get("nodes"),
                )
            )
        lines.sort(key=lambda line: line.multipv)
        return AnalysisResult(
            fen=board.fen(),
            lines=lines,
            check=check_status(board),
            hanging=hanging_pieces(board),
            engine_id=self.engine_id,
        )


def _option_value(option: chess.engine.Option, requested: int, minimum: int) -> int:
    value = max(minimum, int(requested))
    if option.min is not None:
        value = max(value, option.min)
    if option.max is not None:
        value = min(value, option.max)
    return value
