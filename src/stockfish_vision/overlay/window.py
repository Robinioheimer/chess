from __future__ import annotations

import ctypes

import chess
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from stockfish_vision.board.geometry import inverse_matrix, square_center_screen
from stockfish_vision.config import OverlaySettings
from stockfish_vision.engine.client import AnalysisResult

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

user32 = ctypes.windll.user32
if ctypes.sizeof(ctypes.c_void_p) == 8:
    _get_long = user32.GetWindowLongPtrW
    _set_long = user32.SetWindowLongPtrW
else:
    _get_long = user32.GetWindowLongW
    _set_long = user32.SetWindowLongW
VARIANT_COLORS = [
    QColor(46, 204, 113, 220),
    QColor(52, 152, 219, 220),
    QColor(241, 196, 15, 220),
    QColor(155, 89, 182, 220),
    QColor(230, 126, 34, 220),
]


class OverlayWindow(QWidget):
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
        self.warp_size = 400
        self.origin = (0.0, 0.0)
        self.settings = OverlaySettings()
        self.click_through = True
        self.user_hidden = False

    def apply_settings(self, settings: OverlaySettings) -> None:
        self.settings = settings
        self.setWindowOpacity(max(0.2, min(1.0, settings.opacity)))

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
        self.update()

    def cover_virtual_desktop(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.virtualGeometry()
            self.setGeometry(geo)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.cover_virtual_desktop()
        self._apply_win32_style()

    def set_click_through(self, enabled: bool) -> None:
        self.click_through = enabled
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        self._apply_win32_style()

    def _hwnd(self) -> int:
        return int(self.winId())

    def _apply_win32_style(self) -> None:
        hwnd = self._hwnd()
        style = _get_long(hwnd, GWL_EXSTYLE)
        style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        if self.click_through:
            style |= WS_EX_TRANSPARENT
        else:
            style &= ~WS_EX_TRANSPARENT
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

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self.user_hidden or self.result is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.corners is not None and self.result.lines:
            try:
                import numpy as np

                matrix = __import__(
                    "stockfish_vision.board.geometry", fromlist=["perspective_matrix"]
                ).perspective_matrix(self.corners, self.warp_size)
                matrix_inv = inverse_matrix(matrix)
                for index, line in enumerate(self.result.lines):
                    if not line.pv_uci:
                        continue
                    move = chess.Move.from_uci(line.pv_uci[0])
                    start = square_center_screen(
                        move.from_square, self.orientation, self.warp_size, matrix_inv
                    )
                    end = square_center_screen(
                        move.to_square, self.orientation, self.warp_size, matrix_inv
                    )
                    ox, oy = self.origin
                    start = (start[0] + ox, start[1] + oy)
                    end = (end[0] + ox, end[1] + oy)
                    color = VARIANT_COLORS[index % len(VARIANT_COLORS)]
                    self._draw_square_marker(painter, start, color, fill=True)
                    self._draw_square_marker(painter, end, color, fill=False)
                    self._draw_arrow(painter, start, end, color, width=4 + max(0, 3 - index))
            except Exception:
                pass
        self._draw_panel(painter)

    def _draw_square_marker(self, painter: QPainter, center: tuple[float, float], color: QColor, fill: bool) -> None:
        painter.save()
        pen = QPen(color, 3)
        painter.setPen(pen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 80) if fill else Qt.BrushStyle.NoBrush)
        painter.drawRect(int(center[0] - 18), int(center[1] - 18), 36, 36)
        painter.restore()

    def _draw_arrow(
        self,
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
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        ux, uy = dx / length, dy / length
        left = QPointF(end[0] - ux * 18 + uy * 8, end[1] - uy * 18 - ux * 8)
        right = QPointF(end[0] - ux * 18 - uy * 8, end[1] - uy * 18 + ux * 8)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([QPointF(*end), left, right]))
        painter.restore()

    def _draw_panel(self, painter: QPainter) -> None:
        if self.result is None:
            return
        x = self.settings.panel_offset_x
        y = self.settings.panel_offset_y
        w = self.settings.panel_width
        h = self.settings.panel_height
        painter.setBrush(QColor(18, 22, 28, 190))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x, y, w, h, 10, 10)
        painter.setPen(QColor(236, 240, 241))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        lines = [self.result.check or "Analyse"]
        if self.result.lines:
            best = self.result.lines[0]
            lines.append(f"{best.score_text}   {best.best_move_san}")
            if best.pv_san:
                lines.append(best.pv_san)
        if self.result.hanging:
            lines.append("Hängend: " + ", ".join(self.result.hanging[:3]))
        for index, text in enumerate(lines[:6]):
            painter.drawText(x + 12, y + 28 + index * 24, text)
