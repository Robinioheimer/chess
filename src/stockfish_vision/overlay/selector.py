"""Vollbild-Auswahl des Bildschirmausschnitts mit der Maus.

Der Bildschirm wird abgedunkelt, der gewählte Bereich bleibt hell. Ein Zug mit
gedrückter linker Maustaste legt den Aufnahmebereich fest, ``Esc`` bricht ab.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

PANEL_BG = QColor(16, 19, 23, 235)
ACCENT = QColor(255, 45, 45)
MIN_SIZE = 24


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
        geometry = QRect()
        for screen in QGuiApplication.screens():
            geometry = geometry.united(screen.geometry())
        if not geometry.isValid() or geometry.isNull():
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                geometry = screen.geometry()
        self._origin = None
        self._current = None
        self._finished = False
        if geometry.isValid() and not geometry.isNull():
            self.setGeometry(geometry)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

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
        font = QFont("Segoe UI", 12, QFont.Weight.DemiBold)
        small = QFont("Segoe UI", 9)
        title_width = painter.fontMetrics().horizontalAdvance(self.prompt)
        painter.setFont(font)
        title_width = painter.fontMetrics().horizontalAdvance(self.prompt)
        painter.setFont(small)
        hint_width = painter.fontMetrics().horizontalAdvance(self.hint)
        box_width = max(title_width, hint_width) + 48
        box = QRect(0, 0, box_width, 74)
        box.moveCenter(QPoint(full.center().x(), full.bottom() - 70))
        painter.setBrush(PANEL_BG)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawRoundedRect(box, 10, 10)
        painter.setFont(font)
        painter.setPen(QColor(240, 245, 250))
        painter.drawText(
            QRect(box.left(), box.top() + 10, box.width(), 26),
            Qt.AlignmentFlag.AlignCenter,
            self.prompt,
        )
        painter.setFont(small)
        painter.setPen(QColor(160, 172, 184))
        painter.drawText(
            QRect(box.left(), box.top() + 40, box.width(), 22),
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
