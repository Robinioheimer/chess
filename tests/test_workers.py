from __future__ import annotations

import sys
import time
from pathlib import Path

import chess
import pytest
from PySide6.QtCore import QCoreApplication, Qt, QThread

from stockfish_vision.config import EngineSettings
from stockfish_vision.gui.worker import EngineWorker, RecognitionResult, RecognitionWorker
from stockfish_vision.recognition.glyphs import render_piece_glyph

FAKE = Path(__file__).resolve().parent / "fake_uci_engine.py"


@pytest.fixture(scope="module")
def core_app():
    return QCoreApplication.instance() or QCoreApplication([])


def _process_until(core_app, predicate, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        core_app.processEvents()
        time.sleep(0.005)
    assert predicate(), "Qt-Worker hat nicht rechtzeitig geantwortet"


def test_engine_worker_runs_uci_in_qthread(core_app) -> None:
    thread = QThread()
    worker = EngineWorker()
    worker.moveToThread(thread)
    worker.request_start.connect(worker.start_engine, Qt.ConnectionType.QueuedConnection)
    worker.request_analyse.connect(worker.analyse, Qt.ConnectionType.QueuedConnection)
    worker.request_stop.connect(worker.stop_engine, Qt.ConnectionType.QueuedConnection)
    settings = EngineSettings(depth=5, move_time_ms=0, threads=1, hash_mb=16, multipv=2)
    results = []
    errors = []
    stopped = []
    worker.failed.connect(errors.append)
    worker.started.connect(
        lambda name: worker.request_analyse.emit(chess.Board(), settings)
    )
    worker.finished.connect(results.append)
    worker.stopped.connect(lambda: stopped.append(True))
    thread.start()
    try:
        worker.request_start.emit(settings, [sys.executable, "-u", str(FAKE)])
        _process_until(core_app, lambda: bool(results) or bool(errors))
        assert not errors
        assert results and results[0].lines[0].best_move_uci == "e2e4"
        worker.request_stop.emit()
        _process_until(core_app, lambda: bool(stopped))
        assert thread.isRunning()
    finally:
        thread.quit()
        assert thread.wait(3000)
        worker.client.stop()


def test_recognition_worker_emits_result_in_qthread(core_app) -> None:
    thread = QThread()
    worker = RecognitionWorker(18.0, 0.45)
    worker.moveToThread(thread)
    worker.request_recognize.connect(worker.recognize, Qt.ConnectionType.QueuedConnection)
    results: list[RecognitionResult] = []
    errors = []
    worker.finished.connect(results.append)
    worker.failed.connect(errors.append)
    # 64 solid tiles are quick and exercise the complete worker/signal path.
    import cv2
    import numpy as np

    tile = render_piece_glyph("K", size=48, jitter=False)
    warped = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    warped = np.tile(warped, (8, 8, 1))
    thread.start()
    try:
        worker.request_recognize.emit(warped, "white_bottom", chess.WHITE)
        _process_until(core_app, lambda: bool(results) or bool(errors))
        assert not errors
        assert len(results) == 1
        assert len(results[0].detections) == 64
        assert results[0].board.piece_map()
    finally:
        thread.quit()
        assert thread.wait(3000)
