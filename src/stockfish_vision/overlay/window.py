from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes
from dataclasses import replace

import chess
import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from stockfish_vision.board.geometry import (
    inverse_matrix,
    perspective_matrix,
    square_center_screen,
    square_polygon_screen,
)
from stockfish_vision.config import OverlaySettings
from stockfish_vision.engine.client import AnalysisResult

logger = logging.getLogger(__name__)

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
SWP_NOACTIVATE = 0x0010

if os.name == "nt":
    user32 = ctypes.windll.user32
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        _get_long = user32.GetWindowLongPtrW
        _set_long = user32.SetWindowLongPtrW
    else:
        _get_long = user32.GetWindowLongW
        _set_long = user32.SetWindowLongW
    _get_long.argtypes = [wintypes.HWND, ctypes.c_int]
    _get_long.restype = ctypes.c_ssize_t
    _set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    _set_long.restype = ctypes.c_ssize_t
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
else:
    user32 = None
    _get_long = None
    _set_long = None

VARIANT_COLORS = [
    QColor(46, 204, 113, 230),
    QColor(52, 152, 219, 230),
    QColor(241, 196, 15, 230),
    QColor(155, 89, 182, 230),
    QColor(230, 126, 34, 230),
]


class OverlayPanel(QWidget):
    """Interactive top-level panel; the separate arrow layer stays click-through."""

    geometry_edited = Signal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(220, 130)
        self.result: AnalysisResult | None = None
        self.scale_percent = 100
        self._drag_start: QPointF | None = None
        self._initial_geometry = QRect()
        self._resizing = False

    def set_result(self, result: AnalysisResult | None) -> None:
        self.result = result
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_start = event.globalPosition()
        self._initial_geometry = self.geometry()
        self._resizing = (
            event.position().x() >= self.width() - 30
            and event.position().y() >= self.height() - 30
        )
        self.setCursor(
            Qt.CursorShape.SizeFDiagCursor if self._resizing else Qt.CursorShape.ClosedHandCursor
        )
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_start is None:
            self.setCursor(
                Qt.CursorShape.SizeFDiagCursor
                if event.position().x() >= self.width() - 30
                and event.position().y() >= self.height() - 30
                else Qt.CursorShape.OpenHandCursor
            )
            return
        delta = (event.globalPosition() - self._drag_start).toPoint()
        if self._resizing:
            width = max(220, self._initial_geometry.width() + delta.x())
            height = max(130, self._initial_geometry.height() + delta.y())
            self.resize(width, height)
        else:
            self.move(self._initial_geometry.topLeft() + delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_start is not None:
            self._drag_start = None
            self._resizing = False
            self.unsetCursor()
            self.geometry_edited.emit()
        event.accept()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self.result is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(18, 22, 28, 218))
        painter.setPen(QPen(QColor(90, 110, 125, 180), 1))
        painter.drawRoundedRect(QRect(0, 0, self.width() - 1, self.height() - 1), 10, 10)
        scale = self.scale_percent / 100.0
        title_font = QFont("Segoe UI", max(8, round(10 * scale)), QFont.Weight.DemiBold)
        body_font = QFont("Segoe UI", max(8, round(9 * scale)))
        painter.setFont(title_font)
        painter.setPen(QColor(118, 220, 156))
        painter.drawText(14, 24, "TRAINING / ANALYSE")
        painter.setPen(QColor(160, 172, 184))
        painter.drawText(self.width() - 70, 24, "ziehen ⠿")
        painter.setFont(body_font)
        painter.setPen(QColor(236, 240, 241))
        painter.drawText(14, 45, f"{self.result.engine_id} · {self.result.check}")
        metrics = QFontMetrics(body_font)
        y = 68
        line_height = max(20, round(23 * scale))
        for line in self.result.lines[:5]:
            if y + line_height > self.height() - 18:
                break
            prefix = f"PV{line.multipv}  {line.score_text:>6}  "
            painter.setPen(VARIANT_COLORS[(line.multipv - 1) % len(VARIANT_COLORS)])
            painter.drawText(14, y, prefix)
            painter.setPen(QColor(236, 240, 241))
            available = max(40, self.width() - 28 - metrics.horizontalAdvance(prefix))
            text = metrics.elidedText(line.pv_san or "Keine Zugfolge", Qt.TextElideMode.ElideRight, available)
            painter.drawText(14 + metrics.horizontalAdvance(prefix), y, text)
            y += line_height
        if not self.result.lines and y < self.height() - 15:
            painter.drawText(14, y, "Keine legalen Züge / keine Engine-Variante.")
        if self.result.hanging and y + line_height < self.height() - 12:
            painter.setPen(QColor(241, 196, 15))
            text = metrics.elidedText(
                "Hängend: " + ", ".join(self.result.hanging),
                Qt.TextElideMode.ElideRight,
                self.width() - 28,
            )
            painter.drawText(14, self.height() - 24, text)
        painter.setPen(QColor(160, 172, 184))
        painter.drawText(self.width() - 24, self.height() - 10, "◢")


class OverlayWindow(QWidget):
    panel_settings_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.result: AnalysisResult | None = None
        self.corners: list[list[float]] | None = None
        self.orientation = "white_bottom"
        self.warp_size = 640
        self.origin = (0.0, 0.0)
        self.settings = OverlaySettings()
        self.click_through = True
        self.user_hidden = False
        self.panel = OverlayPanel()
        self.panel.geometry_edited.connect(self._panel_geometry_edited)
        self._desktop_origin = QPoint(0, 0)
        self._geometry_error_reported = False

    def apply_settings(self, settings: OverlaySettings) -> None:
        self.settings = replace(settings)
        opacity = max(0.2, min(1.0, settings.opacity))
        self.setWindowOpacity(opacity)
        self.panel.setWindowOpacity(opacity)
        self.panel.scale_percent = settings.scale_percent
        scale = settings.scale_percent / 100.0
        self.panel.resize(round(settings.panel_width * scale), round(settings.panel_height * scale))
        self.panel.move(
            self._desktop_origin.x() + settings.panel_offset_x,
            self._desktop_origin.y() + settings.panel_offset_y,
        )
        self.panel.update()

    def set_analysis(
        self,
        result: AnalysisResult | None,
        corners: list[list[float]] | None,
        orientation: str,
        warp_size: int,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.result = result
        self.corners = corners
        self.orientation = orientation
        self.warp_size = warp_size
        self.origin = origin
        self._geometry_error_reported = False
        self.panel.set_result(result)
        self.panel.setVisible(result is not None and not self.user_hidden and self.isVisible())
        self.update()

    def cover_virtual_desktop(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geometry = screen.virtualGeometry()
            self.setGeometry(geometry)
            self._desktop_origin = geometry.topLeft()
            self.panel.move(
                geometry.x() + self.settings.panel_offset_x,
                geometry.y() + self.settings.panel_offset_y,
            )

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.cover_virtual_desktop()
        self._apply_win32_style()
        if self.result is not None and not self.user_hidden:
            self.panel.show()

    def hideEvent(self, event) -> None:  # type: ignore[override]
        self.panel.hide()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.panel.close()
        super().closeEvent(event)

    def set_click_through(self, enabled: bool) -> None:
        self.click_through = enabled
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        self._apply_win32_style()

    def _hwnd(self) -> int:
        return int(self.winId())

    def _apply_win32_style(self) -> None:
        if os.name != "nt" or user32 is None or _get_long is None or _set_long is None:
            return
        hwnd = self._hwnd()
        style = _get_long(hwnd, GWL_EXSTYLE)
        style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        style = style | WS_EX_TRANSPARENT if self.click_through else style & ~WS_EX_TRANSPARENT
        _set_long(hwnd, GWL_EXSTYLE, style)
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW | SWP_NOACTIVATE,
        )

    def _panel_geometry_edited(self) -> None:
        scale = max(1, self.settings.scale_percent) / 100.0
        self.settings.panel_offset_x = self.panel.x() - self._desktop_origin.x()
        self.settings.panel_offset_y = self.panel.y() - self._desktop_origin.y()
        self.settings.panel_width = round(self.panel.width() / scale)
        self.settings.panel_height = round(self.panel.height() / scale)
        self.panel_settings_changed.emit(replace(self.settings))

    def _to_local(self, point: tuple[float, float]) -> tuple[float, float]:
        ox, oy = self.origin
        return (
            point[0] + ox - self.geometry().x(),
            point[1] + oy - self.geometry().y(),
        )

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self.user_hidden or self.result is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.corners is not None and self.result.lines:
            try:
                matrix_inv = inverse_matrix(perspective_matrix(self.corners, self.warp_size))
                for index, line in enumerate(self.result.lines):
                    if not line.pv_uci:
                        continue
                    move = chess.Move.from_uci(line.pv_uci[0])
                    start = self._to_local(
                        square_center_screen(move.from_square, self.orientation, self.warp_size, matrix_inv)
                    )
                    end = self._to_local(
                        square_center_screen(move.to_square, self.orientation, self.warp_size, matrix_inv)
                    )
                    color = VARIANT_COLORS[index % len(VARIANT_COLORS)]
                    self._draw_square_marker(
                        painter,
                        [self._to_local(point) for point in square_polygon_screen(
                            move.from_square, self.orientation, self.warp_size, matrix_inv
                        )],
                        color,
                        fill=True,
                    )
                    self._draw_square_marker(
                        painter,
                        [self._to_local(point) for point in square_polygon_screen(
                            move.to_square, self.orientation, self.warp_size, matrix_inv
                        )],
                        color,
                        fill=False,
                    )
                    self._draw_arrow(painter, start, end, color, width=4 + max(0, 3 - index))
            except (ValueError, cv2.error, chess.InvalidMoveError, np.linalg.LinAlgError) as exc:
                if not self._geometry_error_reported:
                    logger.warning("Overlay-Geometrie konnte nicht gezeichnet werden: %s", exc)
                    self._geometry_error_reported = True

    @staticmethod
    def _draw_square_marker(
        painter: QPainter, points: list[tuple[float, float]], color: QColor, fill: bool
    ) -> None:
        painter.save()
        painter.setPen(QPen(color, 3))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 70) if fill else Qt.BrushStyle.NoBrush)
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        painter.restore()

    @staticmethod
    def _draw_arrow(
        painter: QPainter,
        start: tuple[float, float],
        end: tuple[float, float],
        color: QColor,
        width: int,
    ) -> None:
        painter.save()
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(*start), QPointF(*end))
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        ux, uy = dx / length, dy / length
        left = QPointF(end[0] - ux * 18 + uy * 8, end[1] - uy * 18 - ux * 8)
        right = QPointF(end[0] - ux * 18 - uy * 8, end[1] - uy * 18 + ux * 8)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([QPointF(*end), left, right]))
        painter.restore()
