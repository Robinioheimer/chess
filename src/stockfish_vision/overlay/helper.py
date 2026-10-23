"""Kleines, immer sichtbares Helfer-Fenster für die Einrichtung.

Das Fenster ersetzt die Bedienung im Hauptfenster: Es zeigt die drei Schritte
(Ausschnitt wählen, Snapshot + erkennen, analysieren), bleibt im Vordergrund
und lässt sich an eine freie Stelle ziehen.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

PANEL_BG = QColor(255, 255, 255, 245)
BORDER = QColor(190, 194, 199)
ACCENT = QColor(107, 114, 128)
DONE = QColor(31, 122, 63)
PENDING = QColor(140, 146, 153)
TEXT = QColor(17, 18, 20)
TEXT_MUTED = QColor(91, 96, 103)
FRAME_RED = QColor(214, 48, 40)
WIDTH = 460
MIN_WIDTH = 340
MAX_WIDTH = 1400
RESIZE_MARGIN = 10

STEP_LABELS = (
    "1 · Ausschnitt wählen",
    "2 · Snapshot und erkennen",
    "3 · Analysieren",
)


class HelperPopup(QWidget):
    """Führt durch die Einrichtung und hält die Aktionen erreichbar."""

    select_region_requested = Signal()
    snapshot_requested = Signal()
    tracking_toggled = Signal()
    analyze_requested = Signal()
    settings_requested = Signal()
    overlay_test_requested = Signal()
    hidden_by_user = Signal()
    position_changed = Signal(int, int)
    size_changed = Signal(int)

    def __init__(self, version: str = "v4", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        for button in self.findChildren(QPushButton):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(MIN_WIDTH)
        self.version = version
        self.step = 0
        self.status_text = "Ausschnitt wählen: Rahmen mit der Maus um das Spielfeld ziehen."
        self.busy = False
        self._drag_start: QPoint | None = None
        self._drag_origin: QPoint | None = None
        self._resize_start: QPoint | None = None
        self._resize_width = WIDTH
        self.apply_width(WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 24)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("Brett einrichten")
        title.setStyleSheet("font-weight: 700; font-size: 15px; color: #111214;")
        version_label = QLabel(version)
        version_label.setStyleSheet("color:#5b6067; font-weight:600;")
        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(26, 26)
        self.close_button.setToolTip("Helfer ausblenden (Hotkey oder Start-Tab zeigt ihn wieder)")
        self.close_button.setStyleSheet(
            "QPushButton { background:#f1f2f4; border:1px solid #cfd1d4;"
            " border-radius:13px; color:#33383e; }"
            "QPushButton:hover { background:#e3e5e8; }"
        )
        self.close_button.clicked.connect(self._hide_by_user)
        title_row.addWidget(title)
        title_row.addWidget(version_label)
        title_row.addStretch(1)
        title_row.addWidget(self.close_button)
        layout.addLayout(title_row)

        self.steps_label = QLabel()
        self.steps_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.steps_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(46)
        self.status_label.setStyleSheet("color:#33383e;")
        layout.addWidget(self.status_label)

        self.select_button = QPushButton("Ausschnitt mit der Maus wählen")
        self.select_button.setToolTip("Vollbild-Auswahl: Rahmen um das Spielfeld ziehen (Esc bricht ab)")
        self.select_button.clicked.connect(self.select_region_requested.emit)
        layout.addWidget(self.select_button)

        action_row = QHBoxLayout()
        self.track_button = QPushButton("Live-Tracking starten")
        self.track_button.setToolTip(
            "Erkennt die Stellung laufend neu, ohne die Engine aufzurufen. "
            "Für eine Analyse den Snapshot einfrieren."
        )
        self.track_button.clicked.connect(self.tracking_toggled.emit)
        layout.addWidget(self.track_button)

        self.snapshot_button = QPushButton("Snapshot + erkennen")
        self.snapshot_button.clicked.connect(self.snapshot_requested.emit)
        self.analyze_button = QPushButton("Analysieren")
        self.analyze_button.clicked.connect(self.analyze_requested.emit)
        action_row.addWidget(self.snapshot_button)
        action_row.addWidget(self.analyze_button)
        layout.addLayout(action_row)

        small_row = QHBoxLayout()
        self.settings_button = QPushButton("Einstellungen")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        self.test_button = QPushButton("Overlay testen")
        self.test_button.clicked.connect(self.overlay_test_requested.emit)
        self.hide_button = QPushButton("Ausblenden")
        self.hide_button.clicked.connect(self._hide_by_user)
        for button in (self.settings_button, self.test_button, self.hide_button):
            button.setStyleSheet(
                "QPushButton { background:#f4f5f6; border:1px solid #cfd1d4;"
                " padding:6px 8px; font-size:12px; color:#33383e; }"
                "QPushButton:hover { background:#e7e8eb; }"
            )
            small_row.addWidget(button)
        layout.addLayout(small_row)

        self.hint_label = QLabel("Ziehen verschiebt das Fenster")
        self.hint_label.setStyleSheet("color:#5b6067; font-size:11px;")
        layout.addWidget(self.hint_label)

        self._refresh()

    # ------------------------------------------------------------- Größe
    @staticmethod
    def clamp_width(width: int | float) -> int:
        try:
            value = int(width)
        except (TypeError, ValueError):
            return WIDTH
        return max(MIN_WIDTH, min(MAX_WIDTH, value))

    def apply_width(self, width: int | float) -> int:
        value = self.clamp_width(width)
        self.setFixedWidth(value)
        self._resize_width = value
        self.update()
        return value

    def panel_width(self) -> int:
        return self.width()

    def _is_resize_hit(self, position: QPoint) -> bool:
        return position.x() >= self.width() - RESIZE_MARGIN

    # ------------------------------------------------------------- Zustand
    def set_step(
        self,
        step: int,
        status: str,
        busy: bool = False,
        can_analyze: bool | None = None,
    ) -> None:
        self.step = max(0, min(len(STEP_LABELS) - 1, int(step)))
        self.status_text = status
        self.busy = bool(busy)
        if can_analyze is None:
            can_analyze = self.step >= 2
        self._can_analyze = bool(can_analyze)
        self._refresh()

    def _refresh(self) -> None:
        rows = []
        for index, label in enumerate(STEP_LABELS):
            if index < self.step:
                color = DONE.name()
                marker = "✔"
            elif index == self.step:
                color = "#111214"
                marker = "●"
            else:
                color = PENDING.name()
                marker = "○"
            weight = "600" if index == self.step else "400"
            rows.append(
                f'<span style="color:{color}; font-weight:{weight};">{marker}&nbsp;&nbsp;{label}</span>'
            )
        self.steps_label.setText("<br>".join(rows))
        self.status_label.setText(self.status_text)
        self.select_button.setEnabled(not self.busy)
        self.snapshot_button.setEnabled(not self.busy)
        self.analyze_button.setEnabled(not self.busy and getattr(self, "_can_analyze", False))
        self.snapshot_button.setText(
            "Snapshot läuft …" if self.busy else "Snapshot + erkennen"
        )

    def _hide_by_user(self) -> None:
        self.hide()
        self.hidden_by_user.emit()

    def set_tracking(self, active: bool) -> None:
        """Zeigt an, ob das Live-Tracking gerade läuft."""
        self.track_button.setText(
            "Live-Tracking stoppen" if active else "Live-Tracking starten"
        )
        self.track_button.setStyleSheet(
            "QPushButton { background:#111214; color:#ffffff; font-weight:600; }"
            "QPushButton:hover { background:#2a2d31; }"
            if active
            else ""
        )

    def describe_state(self) -> str:
        return (
            f"Schritt {self.step + 1}/3 · sichtbar={'ja' if self.isVisible() else 'nein'} · "
            f"Analysieren={'frei' if getattr(self, '_can_analyze', False) else 'gesperrt'} · "
            f"Status: {self.status_text}"
        )

    # ------------------------------------------------------------- Ziehen
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._is_resize_hit(event.position().toPoint()):
            self._resize_start = event.globalPosition().toPoint()
            self._resize_width = self.width()
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()
            return
        self._drag_start = event.globalPosition().toPoint()
        self._drag_origin = self.pos()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._resize_start is not None:
            delta = event.globalPosition().toPoint().x() - self._resize_start.x()
            self.apply_width(self._resize_width + delta)
            event.accept()
            return
        if self._drag_start is None or self._drag_origin is None:
            self.setCursor(
                Qt.CursorShape.SizeHorCursor
                if self._is_resize_hit(event.position().toPoint())
                else Qt.CursorShape.ArrowCursor
            )
            return
        delta = event.globalPosition().toPoint() - self._drag_start
        self.move(self._drag_origin + delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._resize_start is not None:
            self._resize_start = None
            self.unsetCursor()
            self.size_changed.emit(self.width())
            event.accept()
            return
        if self._drag_start is None:
            return
        self._drag_start = None
        self._drag_origin = None
        self.position_changed.emit(self.x(), self.y())
        event.accept()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if self._resize_start is None and self._drag_start is None:
            self.unsetCursor()
        super().leaveEvent(event)

    # ------------------------------------------------------------- Malen
    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(PANEL_BG)
        painter.setPen(QPen(BORDER, 1))
        painter.drawRoundedRect(QRect(0, 0, self.width() - 1, self.height() - 1), 12, 12)
        painter.setPen(QPen(FRAME_RED, 3))
        painter.drawLine(16, 2, 120, 2)
        grip = QColor(150, 155, 161, 220)
        painter.setPen(QPen(grip, 2))
        for offset in (0, 5, 10):
            painter.drawLine(
                self.width() - 8 - offset,
                self.height() - 14,
                self.width() - 8,
                self.height() - 14 - offset,
            )
        painter.setPen(TEXT_MUTED)
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        text = QFontMetrics(font).elidedText(
            "Immer im Vordergrund · Esc bricht die Auswahl ab",
            Qt.TextElideMode.ElideRight,
            self.width() - 32,
        )
        painter.drawText(16, self.height() - 4, text)

    def place_default(self) -> None:
        screen = self.screen() or self.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.adjustSize()
        self.move(available.right() - self.width() - 24, available.top() + 24)

    def show_helper(self) -> None:
        if not self.isVisible():
            self.adjustSize()
            self.show()
        self.raise_()
