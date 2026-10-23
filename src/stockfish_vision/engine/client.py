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
            path = Path(configured)
            if path.is_file():
                return str(path)
            raise FileNotFoundError(f"Stockfish nicht gefunden: {configured}")
        env = os.environ.get("STOCKFISH_PATH")
        if env and Path(env).is_file():
            return env
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
                return str(candidate)
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
            executable = self.resolve_executable(settings.executable)
            command = [executable]
        self.command = command
        self._engine = chess.engine.SimpleEngine.popen_uci(command)
        self.engine_id = self._engine.id.get("name", "UCI-Engine")
        options: dict[str, int] = {}
        available = self._engine.options
        if "Threads" in available:
            options["Threads"] = max(1, settings.threads)
        if "Hash" in available:
            options["Hash"] = max(16, settings.hash_mb)
        if options:
            self._engine.configure(options)
        logger.info("Engine gestartet: %s (%s)", self.engine_id, command)
        return self.engine_id

    def stop(self) -> None:
        if self._engine is None:
            return
        try:
            self._engine.quit()
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError, OSError):
            logger.warning("Engine war bereits beendet.")
        self._engine = None

    def analyse(self, board: chess.Board, settings: EngineSettings) -> AnalysisResult:
        if self._engine is None:
            raise RuntimeError("Engine ist nicht gestartet.")
        limit_kwargs: dict[str, float | int] = {}
        if settings.move_time_ms > 0:
            limit_kwargs["time"] = settings.move_time_ms / 1000.0
        if settings.depth > 0:
            limit_kwargs["depth"] = settings.depth
        if not limit_kwargs:
            limit_kwargs["depth"] = 12
        limit = chess.engine.Limit(**limit_kwargs)
        multipv = max(1, min(settings.multipv, 5))
        raw = self._engine.analyse(board, limit, multipv=multipv)
        if isinstance(raw, dict):
            infos = [raw]
        else:
            infos = list(raw)
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
                )
            )
        return AnalysisResult(
            fen=board.fen(),
            lines=lines,
            check=check_status(board),
            hanging=hanging_pieces(board),
            engine_id=self.engine_id,
        )
