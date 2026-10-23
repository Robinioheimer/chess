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


def test_version_is_v7() -> None:
    from stockfish_vision.version import APP_VERSION, VERSION_HISTORY, full_version_label

    assert APP_VERSION == "v7"
    assert "v7" in full_version_label()
    labels = [label for label, _text in VERSION_HISTORY]
    assert labels[-1] == "v7"
    assert labels == ["v1", "v2", "v4", "v5", "v6", "v7"]


def test_popup_width_is_configurable_and_persisted(app, monkeypatch) -> None:
    from stockfish_vision.capture.screen import MonitorInfo
    from stockfish_vision.config import AppConfig
    from stockfish_vision.gui import main_window as module

    monkeypatch.setattr(
        module, "list_monitors", lambda: [MonitorInfo(1, 0, 0, 1200, 900, "Testmonitor")]
    )
    monkeypatch.setattr(module.WindowsHotkeyFilter, "register", lambda self: None)
    monkeypatch.setattr(module, "save_config", lambda config, path=None: None)

    config = AppConfig()
    config.overlay.visible = False
    config.ui.helper_width = 620
    window = module.MainWindow(config)
    try:
        assert window.helper.width() == 620
        assert window.popup_width_spin.value() == 620

        window.popup_width_spin.setValue(760)
        assert window.helper.width() == 760
        assert window.config.ui.helper_width == 760

        # Ziehen am Rand meldet die neue Breite zurück in die Konfiguration.
        window.helper.size_changed.emit(520)
        assert window.config.ui.helper_width == 520
        assert window.popup_width_spin.value() == 520

        window._persist_config()
        assert window.config.ui.helper_width == 520
    finally:
        window.close()
        app.processEvents()


def test_training_consent_is_automatic(app, monkeypatch) -> None:
    from stockfish_vision.capture.screen import MonitorInfo
    from stockfish_vision.config import AppConfig
    from stockfish_vision.gui import main_window as module

    monkeypatch.setattr(
        module, "list_monitors", lambda: [MonitorInfo(1, 0, 0, 1200, 900, "Testmonitor")]
    )
    monkeypatch.setattr(module.WindowsHotkeyFilter, "register", lambda self: None)
    monkeypatch.setattr(module, "save_config", lambda config, path=None: None)

    config = AppConfig()
    config.overlay.visible = False
    window = module.MainWindow(config)
    try:
        # Keine Checkbox mehr, Zustimmung ist gesetzt und dokumentiert.
        assert not hasattr(window, "confirm_box")
        assert window.snapshot_confirmed is True
        assert window.config.training_acknowledged is True
        assert window.config.training_acknowledged_at
        assert "automatisch" in window.training_notice.text()

        # Auch ein alter Konfigurationsstand ohne Zustimmung wird automatisch gesetzt.
        window.config.training_acknowledged = False
        window.config.training_acknowledged_at = ""
        window._record_training_acknowledgement()
        assert window.config.training_acknowledged is True
        assert window.config.training_acknowledged_at
    finally:
        window.close()
        app.processEvents()


def test_theme_is_light_and_grey(app) -> None:
    from stockfish_vision.gui import theme

    assert theme.BACKGROUND.lower() == "#ffffff"
    assert theme.TEXT.lower() == "#111214"
    stylesheet = theme.build_stylesheet()
    lowered = stylesheet.lower()
    # Helles Grunddesign: kein dunkles Fenster, kein Grün als Akzentfarbe.
    for forbidden in ("#121417", "#171a1f", "#2e7d4f", "#1b1f24"):
        assert forbidden not in lowered
    assert theme.BACKGROUND.lower() in lowered
    assert theme.TEXT.lower() in lowered
    # Grün bleibt reine Bedeutungsfarbe, Rot/Orange nur für Warnungen.
    assert theme.SUCCESS.lower() == "#1f7a3f"
    assert theme.ERROR.lower() == "#b3261e"


def test_overlay_surfaces_are_light(app) -> None:
    from stockfish_vision.overlay import helper, selector, window

    for module in (helper, selector):
        bg = module.PANEL_BG
        assert bg.red() > 235 and bg.green() > 235 and bg.blue() > 235
        assert module.TEXT.red() < 60 and module.TEXT.green() < 60
    # Analyse-Panel zeichnet einen hellen Kasten mit dunkler Schrift.
    assert window.PANEL_BG.red() > 235 and window.TEXT.red() < 60


def _tracking_window(app, monkeypatch):
    from stockfish_vision.capture.screen import MonitorInfo
    from stockfish_vision.gui import main_window as module

    monkeypatch.setattr(
        module, "list_monitors", lambda: [MonitorInfo(1, 0, 0, 1200, 900, "Testmonitor")]
    )
    monkeypatch.setattr(module.WindowsHotkeyFilter, "register", lambda self: None)
    monkeypatch.setattr(module, "save_config", lambda config, path=None: None)
    config = AppConfig()
    config.overlay.visible = False
    config.capture.warp_size = 320
    config.capture.change_threshold = 1.0
    return module.MainWindow(config)


def test_live_tracking_reacts_only_to_visible_changes(app, monkeypatch) -> None:
    import numpy as np

    from stockfish_vision.board.geometry import square_to_image_rc
    from stockfish_vision.recognition.glyphs import render_piece_glyph

    window = _tracking_window(app, monkeypatch)
    size = 400
    frame = np.full((size, size, 3), 40, dtype=np.uint8)
    calls: list[tuple[str, object]] = []
    try:
        window.corners = [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]
        window._request_recognition = lambda purpose, warped: calls.append((purpose, warped))

        window._on_capture(frame.copy(), (0, 0), "track")
        assert calls == []  # erste Aufnahme setzt nur den Vergleichsstand
        window._on_capture(frame.copy(), (0, 0), "track")
        assert calls == []  # unverändertes Bild löst keine Erkennung aus

        col, row = square_to_image_rc(chess.E2, "white_bottom")
        tile = size // 8
        glyph = render_piece_glyph("P", size=tile, light_square=True)
        frame[
            row * tile : (row + 1) * tile,
            col * tile : (col + 1) * tile,
        ] = glyph[:, :, ::-1]
        window._on_capture(frame.copy(), (0, 0), "track")
        assert len(calls) == 1
        assert calls[0][0] == "track"
        assert "Änderung" in window.capture_status.text() or "Live-Tracking" in window.capture_status.text()
    finally:
        window.close()
        app.processEvents()


def test_live_tracking_toggle_and_status(app, monkeypatch) -> None:
    window = _tracking_window(app, monkeypatch)
    try:
        assert window.live_tracking is False
        assert window.btn_track.text() == "Live-Tracking starten"
        window._request_capture = lambda purpose: None
        window._start_tracking()
        assert window.live_tracking is True
        assert window.track_timer.isActive() is True
        assert window.btn_track.text() == "Live-Tracking stoppen"
        assert "Tracking aktiv" in window.overview_status.text() or "aktiv" in window.overview_status.text()

        window._stop_tracking("Live-Tracking gestoppt.")
        assert window.live_tracking is False
        assert window.track_timer.isActive() is False
        assert window.btn_track.text() == "Live-Tracking starten"
        assert "gestoppt" in window.capture_status.text()
    finally:
        window.close()
        app.processEvents()


def test_tracking_result_updates_board_history_and_blocks_engine(app, monkeypatch) -> None:
    from stockfish_vision.chesslogic.fenutil import describe_validity
    from stockfish_vision.gui.worker import RecognitionResult
    from stockfish_vision.policy import can_analyze

    window = _tracking_window(app, monkeypatch)
    try:
        board = chess.Board()
        window.recognition_purpose = "track"
        window._on_recognition(
            RecognitionResult(
                board=board,
                detections=[],
                validity=describe_validity(board),
                model_error=None,
            )
        )
        assert window.mode == "live_tracking"
        assert window.track_updates == 1
        assert "Live-Tracking #1" in window.history_view.toPlainText()
        assert window.recognition_purpose == "snapshot"
        # Das Live-Tracking darf die Engine nicht selbst aufrufen.
        allowed, _reason = can_analyze("live_tracking")
        assert allowed is False
    finally:
        window.close()
        app.processEvents()


def test_automatic_corrections_are_reported(app, monkeypatch) -> None:
    from stockfish_vision.chesslogic.fenutil import board_from_piece_map, describe_validity
    from stockfish_vision.gui.worker import RecognitionResult
    from stockfish_vision.recognition.plausibility import Correction

    window = _tracking_window(app, monkeypatch)
    try:
        board = board_from_piece_map(
            {chess.E1: chess.Piece.from_symbol("K"), chess.D1: chess.Piece.from_symbol("K")}
        )
        window._on_recognition(
            RecognitionResult(
                board=board,
                detections=[],
                validity=describe_validity(board),
                model_error=None,
                corrections=[Correction(chess.D1, "K", "Q", "nur ein König je Seite")],
            )
        )
        assert "Automatisch korrigiert" in window.detection_details.text()
        assert "d1" in window.detection_details.text()
    finally:
        window.close()
        app.processEvents()
