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
from stockfish_vision.overlay.mapping import physical_to_logical, scaled_to_logical
from stockfish_vision.overlay.screens import screen_rects as collect_screen_rects

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

PANEL_BG = QColor(255, 255, 255, 240)
BORDER = QColor(190, 194, 199)
ACCENT = QColor(107, 114, 128)
TEXT = QColor(17, 18, 20)
TEXT_SOFT = QColor(51, 56, 62)
TEXT_MUTED = QColor(91, 96, 103)
PENDING = QColor(140, 146, 153)
WARNING_TEXT = QColor(154, 91, 0)

PANEL_MIN_WIDTH = 220
PANEL_MIN_HEIGHT = 130


class OverlayPanel(QWidget):
    """Interactives Status-/Analyse-Panel; bleibt unabhängig von den Pfeilen."""

    geometry_edited = Signal()

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(PANEL_MIN_WIDTH, PANEL_MIN_HEIGHT)
        self.result: AnalysisResult | None = None
        self.status_text = "Noch keine Analyse."
        self.scale_percent = 100
        self.accent = QColor("#6b7280")
        self._drag_start: QPointF | None = None
        self._initial_geometry = QRect()
        self._resizing = False

    def set_result(self, result: AnalysisResult | None) -> None:
        self.result = result
        if result is None:
            self.status_text = self.status_text or "Noch keine Analyse."
        self.update()

    def set_status(self, text: str) -> None:
        self.status_text = text
        self.update()

    def set_accent(self, color: QColor) -> None:
        self.accent = QColor(color)
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
            width = max(PANEL_MIN_WIDTH, self._initial_geometry.width() + delta.x())
            height = max(PANEL_MIN_HEIGHT, self._initial_geometry.height() + delta.y())
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
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(PANEL_BG)
        painter.setPen(QPen(BORDER, 1))
        painter.drawRoundedRect(QRect(0, 0, self.width() - 1, self.height() - 1), 10, 10)
        painter.setPen(QPen(self.accent, 3))
        painter.drawLine(12, 2, min(self.width() - 12, 150), 2)

        scale = max(0.6, self.scale_percent / 100.0)
        title_font = QFont("Segoe UI", max(8, round(10 * scale)), QFont.Weight.DemiBold)
        body_font = QFont("Segoe UI", max(8, round(9 * scale)))
        metrics = QFontMetrics(body_font)

        painter.setFont(title_font)
        painter.setPen(self.accent.darker(115))
        painter.drawText(14, 24, "TRAINING / ANALYSE")
        painter.setPen(TEXT_MUTED)
        painter.setFont(body_font)
        painter.drawText(self.width() - metrics.horizontalAdvance("ziehen ⠿") - 12, 24, "ziehen ⠿")

        y = 45
        line_height = max(18, round(23 * scale))
        if self.result is None:
            painter.setPen(TEXT_SOFT)
            text = metrics.elidedText(
                self.status_text or "Noch keine Analyse.",
                Qt.TextElideMode.ElideRight,
                max(40, self.width() - 28),
            )
            painter.drawText(14, y, text)
            painter.setPen(ACCENT)
            hint = metrics.elidedText(
                "Engine starten · Stellung analysieren · Overlay-Test in den Einstellungen",
                Qt.TextElideMode.ElideRight,
                max(40, self.width() - 28),
            )
            if self.height() > 90:
                painter.drawText(14, y + line_height, hint)
        else:
            painter.setPen(TEXT)
            header = f"{self.result.engine_id} · {self.result.check}"
            painter.drawText(
                14,
                y,
                metrics.elidedText(header, Qt.TextElideMode.ElideRight, max(40, self.width() - 28)),
            )
            y += line_height
            for line in self.result.lines[:5]:
                if y + line_height > self.height() - 12:
                    break
                prefix = f"PV{line.multipv}  {line.score_text:>6}  "
                painter.setPen(VARIANT_COLORS[(line.multipv - 1) % len(VARIANT_COLORS)])
                painter.drawText(14, y, prefix)
                painter.setPen(TEXT)
                available = max(40, self.width() - 28 - metrics.horizontalAdvance(prefix))
                text = metrics.elidedText(
                    line.pv_san or "Keine Zugfolge", Qt.TextElideMode.ElideRight, available
                )
                painter.drawText(14 + metrics.horizontalAdvance(prefix), y, text)
                y += line_height
            if not self.result.lines and y < self.height() - 12:
                painter.drawText(14, y, "Keine legalen Züge / keine Engine-Variante.")
            if self.result.hanging and y + line_height < self.height() - 12:
                painter.setPen(WARNING_TEXT)
                text = metrics.elidedText(
                    "Hängend: " + ", ".join(self.result.hanging),
                    Qt.TextElideMode.ElideRight,
                    max(40, self.width() - 28),
                )
                painter.drawText(14, y + line_height, text)

        painter.setPen(PENDING)
        font = QFont("Segoe UI", max(8, round(10 * scale)))
        painter.setFont(font)
        painter.drawText(self.width() - 24, self.height() - 10, "◢")


class OverlayWindow(QWidget):
    """Transparente, klickdurchlässige Ebene für Rahmen und Engine-Pfeile."""

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
        self.test_mode = False
        self.panel = OverlayPanel()
        self.panel.geometry_edited.connect(self._panel_geometry_edited)
        self._desktop_origin = QPoint(0, 0)
        self._geometry_error_reported = False

    # ----------------------------------------------------------- Einstellungen
    def apply_settings(self, settings: OverlaySettings) -> None:
        self.settings = replace(settings)
        opacity = max(0.2, min(1.0, settings.opacity))
        self.setWindowOpacity(opacity)
        self.panel.setWindowOpacity(opacity)
        self.panel.scale_percent = settings.scale_percent
        self.panel.set_accent(QColor(settings.frame_color))
        scale = settings.scale_percent / 100.0
        self.panel.resize(
            max(PANEL_MIN_WIDTH, round(settings.panel_width * scale)),
            max(PANEL_MIN_HEIGHT, round(settings.panel_height * scale)),
        )
        self._place_panel()
        self.panel.update()
        self.update()

    def _place_panel(self) -> None:
        desired_x = self._desktop_origin.x() + self.settings.panel_offset_x
        desired_y = self._desktop_origin.y() + self.settings.panel_offset_y
        geometry = self.geometry()
        if geometry.isValid() and not geometry.isNull():
            max_x = geometry.right() - self.panel.width() - 2
            max_y = geometry.bottom() - self.panel.height() - 2
            desired_x = max(geometry.x() + 2, min(desired_x, max_x))
            desired_y = max(geometry.y() + 2, min(desired_y, max_y))
        self.panel.move(desired_x, desired_y)

    # --------------------------------------------------------------- Inhalt
    def set_analysis(
        self,
        result: AnalysisResult | None,
        corners: list[list[float]] | None,
        orientation: str,
        warp_size: int,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.result = result
        self.set_geometry_inputs(corners, orientation, warp_size, origin)
        self.test_mode = False
        self.panel.set_result(result)
        self.panel.set_status(
            "Analysiere …" if result is None else "Analyse aktualisiert."
        )
        self._sync_panel_visibility()
        self.update()

    def set_geometry_inputs(
        self,
        corners: list[list[float]] | None,
        orientation: str,
        warp_size: int,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.corners = corners if corners is not None and len(corners) == 4 else None
        self.orientation = orientation
        self.warp_size = int(warp_size)
        self.origin = (float(origin[0]), float(origin[1]))
        self._geometry_error_reported = False
        self.update()

    def update_frame(
        self,
        corners: list[list[float]] | None,
        origin: tuple[float, float] = (0.0, 0.0),
        orientation: str | None = None,
        warp_size: int | None = None,
    ) -> None:
        """Aktualisiert nur den roten Brettrahmen (ohne Analyse)."""
        self.corners = corners if corners is not None and len(corners) == 4 else None
        self.origin = (float(origin[0]), float(origin[1]))
        if orientation is not None:
            self.orientation = orientation
        if warp_size is not None:
            self.warp_size = int(warp_size)
        self._geometry_error_reported = False
        self.update()

    def set_status(self, text: str) -> None:
        self.panel.set_status(text)

    def show_test(self) -> None:
        """Zeigt Rahmen, Demo-Pfeil und Panel ohne Engine-Analyse."""
        self.test_mode = True
        self.result = None
        self.panel.set_result(None)
        if self.corners is not None:
            self.panel.set_status("Overlay-Test: Rahmen und Demo-Pfeil (e2–e4) sichtbar.")
        else:
            self.panel.set_status(
                "Overlay-Test: Panel sichtbar. Für Pfeile und Rahmen zuerst das Brett einrahmen."
            )
        self._sync_panel_visibility()
        self.update()

    def describe_state(self) -> str:
        state = [
            f"Overlay sichtbar: {'nein' if self.user_hidden or not self.isVisible() else 'ja'}",
            f"Brettrahmen: {'4 Ecken kalibriert' if self.corners else 'nicht kalibriert'}",
            f"Engine-Ergebnis: {'vorhanden' if self.result else 'keins'}",
            f"Fenstergröße: {self.width()}×{self.height()}",
            f"Aufnahme-Ursprung: {self.origin[0]:.0f}/{self.origin[1]:.0f}",
            f"DPI-Korrektur: {self.settings.coordinate_scale_percent or 'automatisch'}",
        ]
        return " · ".join(state)

    # ------------------------------------------------------------- Fenster
    def _sync_panel_visibility(self) -> None:
        visible = not self.user_hidden and self.isVisible()
        self.panel.setVisible(visible)
        if visible:
            self.panel.raise_()

    def cover_virtual_desktop(self) -> None:
        screens = QGuiApplication.screens()
        geometry = QRect()
        for screen in screens:
            geometry = geometry.united(screen.geometry())
        if not geometry.isValid() or geometry.isNull():
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                geometry = screen.virtualGeometry()
        if not geometry.isValid() or geometry.isNull():
            return
        if self.geometry() != geometry:
            self.setGeometry(geometry)
        self._desktop_origin = geometry.topLeft()
        self._place_panel()

    def present(self) -> None:
        """Macht das Overlay zuverlässig sichtbar und im Vordergrund."""
        self.cover_virtual_desktop()
        self.setVisible(True)
        self.raise_()
        self._apply_win32_style()
        self._sync_panel_visibility()
        self.update()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.cover_virtual_desktop()
        self._apply_win32_style()
        self._sync_panel_visibility()

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

    # ----------------------------------------------------------- Koordinaten
    @staticmethod
    def _screen_rects():
        """Logische und physische Rechtecke je Monitor – siehe overlay.screens."""
        return collect_screen_rects()

    def _to_local(self, point: tuple[float, float]) -> tuple[float, float]:
        physical_x = float(point[0]) + self.origin[0]
        physical_y = float(point[1]) + self.origin[1]
        override = self.settings.coordinate_scale_percent
        if override:
            logical_x, logical_y = scaled_to_logical(physical_x, physical_y, override)
        else:
            logical_x, logical_y = physical_to_logical(
                physical_x, physical_y, self._screen_rects()
            )
        return (
            logical_x - self.geometry().x(),
            logical_y - self.geometry().y(),
        )

    # -------------------------------------------------------------- Malen
    def _frame_color(self) -> QColor:
        color = QColor(self.settings.frame_color)
        return color if color.isValid() else QColor("#ff2d2d")

    def _draw_board_frame(self, painter: QPainter) -> None:
        if not self.settings.show_frame or self.corners is None:
            return
        points = [QPointF(*self._to_local(point)) for point in self.corners]
        color = self._frame_color()
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        glow = QColor(color)
        glow.setAlpha(70)
        painter.setPen(QPen(glow, self.settings.frame_width * 3))
        painter.drawPolygon(QPolygonF(points))
        painter.setPen(QPen(color, self.settings.frame_width))
        painter.drawPolygon(QPolygonF(points))
        painter.restore()

    def _draw_engine_arrows(self, painter: QPainter, matrix_inv: np.ndarray) -> None:
        if self.result is None:
            return
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
                [
                    self._to_local(point)
                    for point in square_polygon_screen(
                        move.from_square, self.orientation, self.warp_size, matrix_inv
                    )
                ],
                color,
                fill=True,
            )
            self._draw_square_marker(
                painter,
                [
                    self._to_local(point)
                    for point in square_polygon_screen(
                        move.to_square, self.orientation, self.warp_size, matrix_inv
                    )
                ],
                color,
                fill=False,
            )
            self._draw_arrow(painter, start, end, color, width=4 + max(0, 3 - index))

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self.user_hidden:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_board_frame(painter)
        if self.corners is None:
            return
        try:
            matrix_inv = inverse_matrix(perspective_matrix(self.corners, self.warp_size))
            if self.result is not None and self.result.lines:
                self._draw_engine_arrows(painter, matrix_inv)
            elif self.test_mode:
                move = chess.Move.from_uci("e2e4")
                start = self._to_local(
                    square_center_screen(
                        move.from_square, self.orientation, self.warp_size, matrix_inv
                    )
                )
                end = self._to_local(
                    square_center_screen(move.to_square, self.orientation, self.warp_size, matrix_inv)
                )
                color = self._frame_color()
                self._draw_square_marker(
                    painter,
                    [
                        self._to_local(point)
                        for point in square_polygon_screen(
                            move.from_square, self.orientation, self.warp_size, matrix_inv
                        )
                    ],
                    color,
                    fill=True,
                )
                self._draw_arrow(painter, start, end, color, width=6)
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
        painter.setBrush(
            QColor(color.red(), color.green(), color.blue(), 70)
            if fill
            else Qt.BrushStyle.NoBrush
        )
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
