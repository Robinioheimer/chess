from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6 import QtWidgets
except ImportError:
    pytest.skip("Native Qt-Bibliotheken nicht verfügbar", allow_module_level=True)

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QMouseEvent

from stockfish_vision.overlay.helper import HelperPopup
from stockfish_vision.overlay.selector import RegionSelector
from stockfish_vision.overlay.topmost import TopmostKeeper


@pytest.fixture(scope="module")
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


def test_helper_popup_steps_and_analyze_gate(app) -> None:
    popup = HelperPopup("v4")
    popup.set_step(0, "Ausschnitt wählen")
    assert popup.step == 0
    assert popup.analyze_button.isEnabled() is False
    popup.set_step(1, "Erkennung fertig", can_analyze=True)
    assert popup.analyze_button.isEnabled() is True
    popup.set_step(1, "Snapshot läuft", busy=True)
    assert popup.snapshot_button.isEnabled() is False
    assert "Snapshot läuft" in popup.snapshot_button.text()
    popup.set_step(9, "geklemmt")
    assert popup.step == 2
    assert "Schritt 3/3" in popup.describe_state()
    popup.close()


def test_helper_popup_hide_signal(app) -> None:
    popup = HelperPopup("v4")
    hidden: list[bool] = []
    popup.hidden_by_user.connect(lambda: hidden.append(True))
    popup.show()
    popup._hide_by_user()
    assert popup.isVisible() is False
    assert hidden == [True]
    popup.close()


def test_helper_popup_position_changes_are_reported(app) -> None:
    popup = HelperPopup("v4")
    popup.show()
    app.processEvents()
    positions: list[tuple[int, int]] = []
    popup.position_changed.connect(lambda x, y: positions.append((x, y)))

    def event(kind: QEvent.Type, x: int, y: int, buttons: Qt.MouseButton):
        local = QPoint(x, y)
        return QMouseEvent(
            kind,
            local,
            popup.mapToGlobal(local),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )

    popup.mousePressEvent(event(QEvent.Type.MouseButtonPress, 20, 20, Qt.MouseButton.LeftButton))
    popup.mouseMoveEvent(event(QEvent.Type.MouseMove, 60, 50, Qt.MouseButton.LeftButton))
    popup.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, 60, 50, Qt.MouseButton.NoButton))
    assert positions, "Position muss nach dem Ziehen gemeldet werden"
    popup.close()


def test_selector_reports_dragged_rectangle(app) -> None:
    selector = RegionSelector()
    selector.start()
    selected: list = []
    selector.selected.connect(selected.append)

    def event(kind: QEvent.Type, x: int, y: int, buttons: Qt.MouseButton):
        local = QPoint(x, y)
        return QMouseEvent(
            kind,
            local,
            selector.mapToGlobal(local),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )

    selector.mousePressEvent(event(QEvent.Type.MouseButtonPress, 120, 90, Qt.MouseButton.LeftButton))
    selector.mouseMoveEvent(event(QEvent.Type.MouseMove, 320, 300, Qt.MouseButton.LeftButton))
    selector.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, 520, 480, Qt.MouseButton.NoButton))
    assert len(selected) == 1
    rect = selected[0]
    assert rect.width() >= 300 and rect.height() >= 300
    assert selector.isVisible() is False
    selector.close()


def test_selector_ignores_tiny_drag_and_supports_cancel(app) -> None:
    selector = RegionSelector()
    selector.start()
    selected: list = []
    cancelled: list[bool] = []
    selector.selected.connect(selected.append)
    selector.cancelled.connect(lambda: cancelled.append(True))

    def event(kind: QEvent.Type, x: int, y: int, buttons: Qt.MouseButton):
        local = QPoint(x, y)
        return QMouseEvent(
            kind,
            local,
            selector.mapToGlobal(local),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )

    selector.mousePressEvent(event(QEvent.Type.MouseButtonPress, 100, 100, Qt.MouseButton.LeftButton))
    selector.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, 104, 104, Qt.MouseButton.NoButton))
    assert selected == []
    selector.cancel()
    assert cancelled == [True]
    assert selector.isVisible() is False
    selector.close()


def test_topmost_keeper_tracks_and_clears_windows(app) -> None:
    keeper = TopmostKeeper(interval_ms=250)
    widget = QtWidgets.QWidget()
    keeper.watch(widget)
    keeper.watch(widget)
    assert keeper.watched_count() == 1
    keeper.start()
    assert keeper.is_active() is True
    assert keeper.reassert() == 0  # nicht sichtbar
    keeper.unwatch(widget)
    assert keeper.watched_count() == 0
    keeper.stop()
    assert keeper.is_active() is False
    widget.close()


def test_helper_popup_width_can_be_resized_and_clamped(app) -> None:
    from stockfish_vision.overlay.helper import MAX_WIDTH, MIN_WIDTH, WIDTH

    popup = HelperPopup("v5")
    assert popup.width() == WIDTH
    assert popup.apply_width(700) == 700
    assert popup.width() == 700
    assert popup.panel_width() == 700
    assert popup.apply_width(10) == MIN_WIDTH
    assert popup.apply_width(99_999) == MAX_WIDTH
    assert popup.clamp_width("unsinn") == WIDTH
    resize_events: list[int] = []
    popup.size_changed.connect(resize_events.append)
    popup.size_changed.emit(popup.width())
    assert resize_events == [popup.width()]
    popup.close()


def test_helper_popup_resize_hit_zone(app) -> None:
    popup = HelperPopup("v5")
    popup.apply_width(400)
    assert popup._is_resize_hit(QPoint(399, 20)) is True
    assert popup._is_resize_hit(QPoint(200, 20)) is False
    popup.close()


def test_selector_covers_full_desktop_and_adds_edge_margin(app) -> None:
    from stockfish_vision.overlay.selector import EDGE_MARGIN, RegionSelector, full_screen_geometry

    expected = full_screen_geometry()
    assert expected.isValid()
    assert expected.width() > 0 and expected.height() > 0
    selector = RegionSelector()
    selector.start()
    assert selector.geometry() == expected
    assert selector.minimumWidth() >= expected.width()
    assert selector.minimumHeight() >= expected.height()
    screen = app.primaryScreen().geometry()
    assert expected.width() >= screen.width() + 2 * EDGE_MARGIN
    selector.close()


def test_region_selection_opens_one_layer_per_screen(app) -> None:
    from stockfish_vision.overlay.selector import RegionSelection

    session = RegionSelection()
    selection = RegionSelection()
    regions = [screen.geometry() for screen in app.screens()]
    selection.start()
    try:
        assert len(selection.selectors) == len(regions)
        assert selection.active() is True
        for selector, region in zip(selection.selectors, regions, strict=True):
            # Jede Ebene deckt ihren Bildschirm vollständig ab (plus 1 px Rand).
            assert selector.geometry().width() >= region.width()
            assert selector.geometry().height() >= region.height()
            assert selector.minimumWidth() >= region.width()
        selection.cancel()
        assert selection.active() is False
        assert all(not selector.isVisible() for selector in selection.selectors)
    finally:
        session.close()
        selection.close()


def test_region_selection_emits_single_selection(app) -> None:
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    from stockfish_vision.overlay.selector import RegionSelection

    session = RegionSelection()
    results: list = []
    session.selected.connect(results.append)
    session.start()
    first = session.selectors[0]

    def event(kind: QEvent.Type, x: int, y: int, buttons: Qt.MouseButton):
        local = QPoint(x, y)
        return QMouseEvent(
            kind,
            local,
            first.mapToGlobal(local),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )

    first.mousePressEvent(event(QEvent.Type.MouseButtonPress, 40, 40, Qt.MouseButton.LeftButton))
    first.mouseMoveEvent(event(QEvent.Type.MouseMove, 140, 130, Qt.MouseButton.LeftButton))
    first.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, 240, 230, Qt.MouseButton.NoButton))

    assert len(results) == 1
    assert results[0].width() >= 150 and results[0].height() >= 150
    assert session.active() is False
    session.close()


def test_helper_tracking_button_reflects_state(app) -> None:
    from stockfish_vision.overlay.helper import HelperPopup

    popup = HelperPopup(version="v7.0.0")
    try:
        assert popup.track_button.text() == "Live-Tracking starten"
        popup.set_tracking(True)
        assert popup.track_button.text() == "Live-Tracking stoppen"
        assert "background:#111214" in popup.track_button.styleSheet()
        popup.set_tracking(False)
        assert popup.track_button.text() == "Live-Tracking starten"
    finally:
        popup.close()
