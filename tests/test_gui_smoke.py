from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6 import QtWidgets
except ImportError:
    pytest.skip("Native Qt-Bibliotheken nicht verfügbar", allow_module_level=True)

import chess

from stockfish_vision.config import AppConfig
from stockfish_vision.gui.board_widget import BoardWidget


@pytest.fixture(scope="module")
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


def test_board_widget_coordinates_and_detection_display(app) -> None:
    widget = BoardWidget()
    widget.resize(800, 800)
    assert widget.square_at(50, 50) == chess.A8
    assert widget.square_at(750, 750) == chess.H1
    widget.flip()
    assert widget.square_at(50, 50) == chess.H1
    widget.close()


def test_main_window_constructs_and_applies_fen(app, monkeypatch, tmp_path) -> None:
    from stockfish_vision.capture.screen import MonitorInfo
    from stockfish_vision.gui import main_window as module

    monkeypatch.setattr(
        module,
        "list_monitors",
        lambda: [MonitorInfo(1, 0, 0, 800, 600, "Testmonitor")],
    )
    monkeypatch.setattr(module.WindowsHotkeyFilter, "register", lambda self: None)
    monkeypatch.setattr(module, "save_config", lambda config, path=None: None)
    config = AppConfig()
    config.overlay.visible = False
    window = module.MainWindow(config)
    try:
        window.fen_edit.setPlainText("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        window._apply_fen()
        assert window.board.piece_at(chess.E2).symbol() == "P"
        assert window.mode == "training"
        assert window.engine_thread is not None
    finally:
        window.close()
        app.processEvents()
