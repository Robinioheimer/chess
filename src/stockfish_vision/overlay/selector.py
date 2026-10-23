"""Vollbild-Auswahl des Bildschirmausschnitts mit der Maus.

Der Bildschirm wird abgedunkelt, der gewählte Bereich bleibt hell. Ein Zug mit
gedrückter linker Maustaste legt den Aufnahmebereich fest, ``Esc`` bricht ab.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from stockfish_vision.overlay.topmost import ensure_topmost

logger = logging.getLogger(__name__)

PANEL_BG = QColor(16, 19, 23, 235)
ACCENT = QColor(255, 45, 45)
MIN_SIZE = 24
EDGE_MARGIN = 2


def desktop_geometry(screens: list[QRect]) -> QRect:
    """Vereinigung aller Bildschirme (logische Pixel).

    Mehrere Monitore, negative Koordinaten und unterschiedliche DPI-Faktoren
    werden berücksichtigt. Ein leeres Ergebnis bedeutet „keine Bildschirme“.
    """

    union = QRect()
    for rect in screens:
        if rect.isValid() and not rect.isNull():
            union = rect if union.isNull() else union.united(rect)
    return union


def full_screen_geometry() -> QRect:
    """Geometrie, die den gesamten Desktop abdeckt."""

    screens = [screen.geometry() for screen in QGuiApplication.screens()]
    geometry = desktop_geometry(screens)
    if geometry.isNull() or not geometry.isValid():
        primary = QGuiApplication.primaryScreen()
        if primary is not None:
            geometry = primary.geometry()
    if not geometry.isNull() and geometry.isValid():
        # Ein Pixel Rand, damit die Kante unter dem Mauszeiger erreichbar bleibt.
        geometry = geometry.adjusted(-EDGE_MARGIN, -EDGE_MARGIN, EDGE_MARGIN, EDGE_MARGIN)
    return geometry


class RegionSelector(QWidget):
    """Transparente Vollbild-Ebene zum Aufziehen eines Rechtecks."""

    selected = Signal(QRect)  # globale Logikpixel
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.prompt = "Ziehe einen Rahmen um das Spielfeld"
        self.hint = "Esc bricht ab"
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._finished = False

    # ------------------------------------------------------------- Steuerung
    def start(self) -> None:
        geometry = full_screen_geometry()
        self._origin = None
        self._current = None
        self._finished = False
        logger.info(
            "Auswahl-Ebene: %s×%s bei %s/%s (%s Bildschirm(e))",
            geometry.width(),
            geometry.height(),
            geometry.x(),
            geometry.y(),
            len(QGuiApplication.screens()),
        )
        if not geometry.isNull() and geometry.isValid():
            # Mindestgröße verhindert, dass Qt das Fenster auf die Primärseite
            # oder auf die Größenanforderung zusammenstutzt.
            self.setMinimumSize(geometry.size())
            self.setGeometry(geometry)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        if not geometry.isNull() and geometry.isValid():
            # Zweiter Durchlauf: manche Plattformen korrigieren die Geometrie
            # erst nach dem ersten Frame.
            self.setGeometry(geometry)
        ensure_topmost(self)
        self.update()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        geometry = full_screen_geometry()
        if not geometry.isNull() and geometry.isValid() and self.geometry() != geometry:
            self.setMinimumSize(geometry.size())
            self.setGeometry(geometry)

    def selection(self) -> QRect:
        if self._origin is None or self._current is None:
            return QRect()
        return QRect(self._origin, self._current).normalized()

    def selection_global(self) -> QRect:
        local = self.selection()
        if local.isNull():
            return QRect()
        return local.translated(self.geometry().topLeft())

    def cancel(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.hide()
        self.cancelled.emit()

    # ---------------------------------------------------------------- Ereignisse
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self.cancel()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._origin = event.position().toPoint()
        self._current = self._origin
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._origin is None:
            return
        self._current = event.position().toPoint()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._origin is None:
            return
        self._current = event.position().toPoint()
        rect = self.selection()
        self._origin = None
        self._current = None
        if rect.width() < MIN_SIZE or rect.height() < MIN_SIZE:
            self.update()
            return
        final = rect.translated(self.geometry().topLeft())
        self._finished = True
        self.hide()
        logger.info("Ausschnitt gewählt: %s×%s an %s/%s", final.width(), final.height(), final.x(), final.y())
        self.selected.emit(final)
        event.accept()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------- Malen
    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        full = self.rect()
        selection = self.selection()
        dim = QColor(0, 0, 0, 140)
        if selection.isNull() or selection.width() < 2 or selection.height() < 2:
            painter.fillRect(full, dim)
        else:
            painter.fillRect(QRect(full.left(), full.top(), full.width(), selection.top() - full.top()), dim)
            painter.fillRect(
                QRect(full.left(), selection.bottom() + 1, full.width(), full.bottom() - selection.bottom()),
                dim,
            )
            painter.fillRect(
                QRect(full.left(), selection.top(), selection.left() - full.left(), selection.height()),
                dim,
            )
            painter.fillRect(
                QRect(selection.right() + 1, selection.top(), full.right() - selection.right(), selection.height()),
                dim,
            )
            glow = QColor(ACCENT)
            glow.setAlpha(70)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(glow, 9))
            painter.drawRect(selection)
            painter.setPen(QPen(ACCENT, 3))
            painter.drawRect(selection)
            self._draw_size_label(painter, selection)
        self._draw_prompt(painter, full)

    def _draw_prompt(self, painter: QPainter, full: QRect) -> None:
        scale = max(1.0, min(2.0, full.height() / 1080.0))
        font = QFont("Segoe UI", max(12, round(12 * scale)), QFont.Weight.DemiBold)
        small = QFont("Segoe UI", max(9, round(9 * scale)))
        title_width = painter.fontMetrics().horizontalAdvance(self.prompt)
        painter.setFont(font)
        title_width = painter.fontMetrics().horizontalAdvance(self.prompt)
        painter.setFont(small)
        hint_width = painter.fontMetrics().horizontalAdvance(self.hint)
        box_width = max(title_width, hint_width) + round(48 * scale)
        box_height = round(74 * scale)
        box = QRect(0, 0, box_width, box_height)
        box.moveCenter(QPoint(full.center().x(), full.bottom() - round(70 * scale)))
        painter.setBrush(PANEL_BG)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawRoundedRect(box, 10, 10)
        painter.setFont(font)
        painter.setPen(QColor(240, 245, 250))
        painter.drawText(
            QRect(box.left(), box.top() + round(10 * scale), box.width(), round(26 * scale)),
            Qt.AlignmentFlag.AlignCenter,
            self.prompt,
        )
        painter.setFont(small)
        painter.setPen(QColor(160, 172, 184))
        painter.drawText(
            QRect(box.left(), box.top() + round(40 * scale), box.width(), round(22 * scale)),
            Qt.AlignmentFlag.AlignCenter,
            self.hint,
        )

    @staticmethod
    def _draw_size_label(painter: QPainter, selection: QRect) -> None:
        text = f"{selection.width()} × {selection.height()}"
        font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.boundingRect(text).width() + 20
        label = QRect(selection.left(), selection.top() - 28, width, 22)
        if label.top() < 4:
            label.moveTop(selection.top() + 6)
        if label.right() > selection.right() + 2:
            label.moveRight(selection.right())
        if label.left() < 2:
            label.moveLeft(2)
        painter.setBrush(ACCENT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(label, 5, 5)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)
