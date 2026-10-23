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


def test_main_window_overview_auto_frame_and_sizing(app, monkeypatch) -> None:
    import cv2
    import numpy as np

    from stockfish_vision.capture.screen import MonitorInfo
    from stockfish_vision.config import AppConfig
    from stockfish_vision.gui import main_window as module

    monkeypatch.setattr(
        module, "list_monitors", lambda: [MonitorInfo(1, 0, 0, 900, 700, "Testmonitor")]
    )
    monkeypatch.setattr(module.WindowsHotkeyFilter, "register", lambda self: None)
    monkeypatch.setattr(module, "save_config", lambda config, path=None: None)

    size = 320
    tile = size // 8
    frame = np.full((600, 800, 3), 30, dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            color = (235, 205, 160) if (row + col) % 2 == 0 else (120, 80, 55)
            cv2.rectangle(
                frame,
                (80 + col * tile, 60 + row * tile),
                (80 + (col + 1) * tile - 1, 60 + (row + 1) * tile - 1),
                color,
                -1,
            )

    config = AppConfig()
    config.overlay.visible = False
    window = module.MainWindow(config)
    try:
        assert window.tabs.tabText(0) == "Start"
        assert "v" in window.windowTitle()
        window._on_capture(frame, (0.0, 0.0), "snapshot")
        assert len(window.corners) == 4
        top_left, top_right, _bottom_right, _bottom_left = window.corners
        assert top_left[0] < top_right[0]
        assert abs(top_left[0] - 80) < 14 and abs(top_left[1] - 60) < 14
        assert window.overlay.corners == window.corners

        window._apply_size_preset(1200, 820)
        assert window.config.ui.window_width == 1200
        assert window.config.ui.window_height == 820
        window._apply_font_scale(140)
        assert window.config.ui.font_scale_percent == 140
        window._apply_font_scale(500)
        assert window.config.ui.font_scale_percent == 200

        window._test_overlay()
        assert window.overlay.test_mode is True

        window._frame_settings_changed()
        assert window.overlay.settings.show_frame is True

        window._reset_corners()
        assert window.corners == []
        assert window.overlay.corners is None
    finally:
        window.close()
        app.processEvents()


def test_main_window_mouse_region_flow_with_helper_popup(app, monkeypatch) -> None:
    import cv2
    import numpy as np
    from PySide6.QtCore import QRect

    from stockfish_vision.capture.screen import MonitorInfo
    from stockfish_vision.chesslogic.fenutil import describe_validity
    from stockfish_vision.config import AppConfig
    from stockfish_vision.gui import main_window as module
    from stockfish_vision.gui.worker import RecognitionResult

    monkeypatch.setattr(
        module, "list_monitors", lambda: [MonitorInfo(1, 0, 0, 1200, 900, "Testmonitor")]
    )
    monkeypatch.setattr(module.WindowsHotkeyFilter, "register", lambda self: None)
    monkeypatch.setattr(module, "save_config", lambda config, path=None: None)

    size = 400
    tile = size // 8
    frame = np.full((800, 1000, 3), 24, dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            color = (235, 205, 160) if (row + col) % 2 == 0 else (120, 80, 55)
            cv2.rectangle(
                frame,
                (150 + col * tile, 120 + row * tile),
                (150 + (col + 1) * tile - 1, 120 + (row + 1) * tile - 1),
                color,
                -1,
            )

    config = AppConfig()
    config.overlay.visible = False
    config.ui.helper_visible = True
    window = module.MainWindow(config)
    snapshots: list[bool] = []
    window._take_snapshot = lambda: snapshots.append(True)
    try:
        assert window.helper.isVisible() is True
        assert window.topmost.watched_count() >= 2
        assert "Popup" in window.overview_status.text()

        # Ausschnitt wie mit der Maus aufziehen: exakt um das Brett.
        window._on_region_selected(QRect(150, 120, size, size))
        assert snapshots == [True]
        region = window.config.capture.region
        assert region is not None
        assert region["width"] >= size and region["height"] >= size
        assert window._chain_recognize is True

        window.capture_in_flight = False
        window._on_capture(frame, (region["left"], region["top"]), "snapshot")
        assert len(window.corners) == 4
        top_left, top_right, _br, _bl = window.corners
        assert top_left[0] < top_right[0]
        assert abs(top_left[0] - 150) < 14 and abs(top_left[1] - 120) < 14

        # Der Helfer führt weiter: Erkennung meldet Erfolg, Analysieren wird frei.
        window.recognition_in_flight = False
        board = chess.Board()
        window._on_recognition(
            RecognitionResult(
                board=board,
                detections=[],
                validity=describe_validity(board),
                model_error=None,
            )
        )
        assert window.helper.step == 1
        assert window.helper.analyze_button.isEnabled() is True

        # Analysieren startet zuerst die Engine, wenn sie noch nicht läuft.
        started: list[bool] = []
        window._start_engine = lambda: started.append(True)
        window.engine_ready = False
        window._helper_analyze()
        assert started == [True] and window._pending_analysis is True
        window._pending_analysis = False
        window.engine_ready = True
        analysed: list[bool] = []
        window._analyze_snapshot = lambda: analysed.append(True)
        window._helper_analyze()
        assert analysed == [True]

        # Position und Sichtbarkeit des Popups landen in der Konfiguration.
        window.helper.position_changed.emit(48, 64)
        assert (window.config.ui.helper_x, window.config.ui.helper_y) == (48, 64)
        window.helper._hide_by_user()
        assert window.config.ui.helper_visible is False
        window._show_helper()
        assert window.helper.isVisible() is True
        assert window.config.ui.helper_visible is True

        # Gespeicherter Ausschnitt wird angezeigt und wieder eingesetzt.
        assert "Ausschnitt:" in window.region_label.text()
    finally:
        window.close()
        app.processEvents()


def test_version_is_v4() -> None:
    from stockfish_vision.version import APP_VERSION, VERSION_HISTORY, full_version_label

    assert APP_VERSION == "v4"
    assert "v4" in full_version_label()
    labels = [label for label, _text in VERSION_HISTORY]
    assert labels[-1] == "v4"
    assert "v3" not in labels
