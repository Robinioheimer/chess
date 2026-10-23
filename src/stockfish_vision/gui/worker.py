from __future__ import annotations

import traceback

import chess
from PySide6.QtCore import QObject, Signal, Slot

from stockfish_vision.config import EngineSettings
from stockfish_vision.engine.client import AnalysisResult, EngineClient


class EngineWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    started = Signal(str)
    stopped = Signal()
    request_start = Signal(object, object)
    request_analyse = Signal(object, object)
    request_stop = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.client = EngineClient()

    @Slot(object, object)
    def start_engine(self, settings: EngineSettings, command: object) -> None:
        try:
            name = self.client.start(settings, command=command if command else None)
            self.started.emit(name)
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def stop_engine(self) -> None:
        self.client.stop()
        self.stopped.emit()

    @Slot(object, object)
    def analyse(self, board: chess.Board, settings: EngineSettings) -> None:
        try:
            result: AnalysisResult = self.client.analyse(board, settings)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
