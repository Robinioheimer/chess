from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field

import chess
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from stockfish_vision.capture.screen import ScreenCapture, list_monitors
from stockfish_vision.config import EngineSettings
from stockfish_vision.engine.client import AnalysisResult, EngineClient
from stockfish_vision.recognition.calibrate import learn_templates
from stockfish_vision.recognition.pieces import Detection, PieceClassifier
from stockfish_vision.recognition.plausibility import Correction

logger = logging.getLogger(__name__)


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
            logger.exception("Engine-Start fehlgeschlagen")
            self.failed.emit(str(exc))

    @Slot()
    def stop_engine(self) -> None:
        try:
            self.client.stop()
        except Exception as exc:
            logger.exception("Engine-Stopp fehlgeschlagen")
            self.failed.emit(str(exc))
        finally:
            self.stopped.emit()

    @Slot(object, object)
    def analyse(self, board: chess.Board, settings: EngineSettings) -> None:
        try:
            result: AnalysisResult = self.client.analyse(board, settings)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Engine-Analyse fehlgeschlagen")
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class CaptureWorker(QObject):
    request_grab = Signal(object, object, object)
    finished = Signal(object, object, object)  # BGR frame, global origin, purpose
    failed = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self.capture = ScreenCapture()

    @Slot(object, object, object)
    def grab(self, monitor_index: int, region: dict[str, int] | None, purpose: str) -> None:
        try:
            frame = self.capture.grab(int(monitor_index), region)
            if region is not None:
                origin = (float(region["left"]), float(region["top"]))
            else:
                monitors = list_monitors()
                monitor = next((item for item in monitors if item.index == int(monitor_index)), None)
                if monitor is None:
                    raise ValueError(f"Monitor {monitor_index} ist nicht mehr verfügbar.")
                origin = (float(monitor.left), float(monitor.top))
            self.finished.emit(frame, origin, str(purpose))
        except Exception as exc:
            logger.exception("Bildschirmaufnahme fehlgeschlagen")
            self.failed.emit(str(exc), purpose)


@dataclass
class RecognitionResult:
    board: chess.Board
    detections: list[Detection]
    validity: tuple[bool, list[str]]
    model_error: str | None
    corrections: list[Correction] = field(default_factory=list)
    purpose: str = "snapshot"


class RecognitionWorker(QObject):
    request_recognize = Signal(object, object, object)
    request_correction = Signal(object, object)
    request_configure = Signal(object, object, object)
    request_learn = Signal(object, object)
    request_reload_templates = Signal()
    finished = Signal(object)
    learned = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        occupancy_threshold: float,
        min_confidence: float,
        model_path: str = "",
    ) -> None:
        super().__init__()
        self.occupancy_threshold = occupancy_threshold
        self.min_confidence = min_confidence
        self.model_path = model_path
        self.classifier: PieceClassifier | None = None

    def _ensure_classifier(self) -> PieceClassifier:
        if self.classifier is None:
            self.classifier = PieceClassifier(
                occupancy_threshold=self.occupancy_threshold,
                min_confidence=self.min_confidence,
                model_path=self.model_path or None,
            )
        return self.classifier

    @Slot(object, object, object)
    def recognize(self, warped: np.ndarray, orientation: str, turn: bool) -> None:
        try:
            classifier = self._ensure_classifier()
            board, detections, validity = classifier.detect_board(warped, str(orientation), bool(turn))
            self.finished.emit(
                RecognitionResult(
                    board=board,
                    detections=detections,
                    validity=validity,
                    model_error=classifier.model_error,
                    corrections=list(getattr(classifier, "last_corrections", [])),
                )
            )
        except Exception as exc:
            logger.exception("Figurenerkennung fehlgeschlagen")
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")

    @Slot(object, object, object)
    def configure(self, occupancy_threshold: float, min_confidence: float, model_path: str) -> None:
        model_changed = str(model_path) != self.model_path
        self.occupancy_threshold = float(occupancy_threshold)
        self.min_confidence = float(min_confidence)
        self.model_path = str(model_path)
        if self.classifier is not None:
            self.classifier.occupancy_threshold = self.occupancy_threshold
            self.classifier.min_confidence = self.min_confidence
        if model_changed:
            self.classifier = None

    @Slot(object, object)
    def learn(self, warped: np.ndarray, orientation: str) -> None:
        """Lernt den sichtbaren Figurensatz aus der Grundstellung."""
        try:
            classifier = self._ensure_classifier()
            report = learn_templates(
                warped, str(orientation), bank=classifier.templates
            )
            classifier.reload_templates()
            self.learned.emit(report)
        except Exception as exc:
            logger.exception("Figurensatz konnte nicht angelernt werden")
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")

    @Slot()
    def reload_templates(self) -> None:
        try:
            count = self._ensure_classifier().reload_templates()
            logger.info("Figurenvorlagen neu geladen: %s", count)
        except Exception as exc:
            logger.warning("Figurenvorlagen konnten nicht neu geladen werden: %s", exc)

    @Slot(object, object)
    def remember_correction(self, square_img: np.ndarray, symbol: str | None) -> None:
        try:
            self._ensure_classifier().remember_correction(square_img, symbol)
        except Exception as exc:
            # A correction must still update the board even if template learning fails.
            logger.warning("Korrektur-Template konnte nicht gespeichert werden: %s", exc)
