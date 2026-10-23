from __future__ import annotations

import math

import chess
from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}
ARROW_COLORS = (QColor(46, 204, 113, 190), QColor(52, 152, 219, 190), QColor(241, 196, 15, 190))


class BoardWidget(QWidget):
    square_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.board = chess.Board()
        self.orientation_white_bottom = True
        self.highlights: set[int] = set()
        self.selected: chess.Square | None = None
        self.confidences: dict[int, float] = {}
        self.unresolved: set[int] = set()
        self.confidence_threshold = 0.45
        self.arrow_moves: list[tuple[int, int]] = []
        self.setMinimumSize(360, 360)

    def set_board(self, board: chess.Board) -> None:
        self.board = board.copy(stack=False)
        self.arrow_moves = []
        self.update()

    def clear_detections(self) -> None:
        self.confidences = {}
        self.unresolved = set()
        self.update()

    def set_detections(self, detections) -> None:
        self.confidences = {int(item.square): float(item.confidence) for item in detections}
        self.unresolved = {int(item.square) for item in detections if item.occupied and item.piece is None}
        self.update()

    def set_analysis(self, lines) -> None:
        self.arrow_moves = []
        for line in lines[:3]:
            if not line.pv_uci:
                continue
            try:
                move = chess.Move.from_uci(line.pv_uci[0])
            except ValueError:
                continue
            self.arrow_moves.append((move.from_square, move.to_square))
        self.highlights = {square for move in self.arrow_moves for square in move}
        self.update()

    def flip(self) -> None:
        self.orientation_white_bottom = not self.orientation_white_bottom
        self.update()

    def square_at(self, x: int, y: int) -> chess.Square | None:
        side = min(self.width(), self.height())
        ox = (self.width() - side) // 2
        oy = (self.height() - side) // 2
        if not (ox <= x < ox + side and oy <= y < oy + side):
            return None
        col = min(max(int((x - ox) / side * 8), 0), 7)
        row = min(max(int((y - oy) / side * 8), 0), 7)
        if self.orientation_white_bottom:
            file_index, rank_index = col, 7 - row
        else:
            file_index, rank_index = 7 - col, row
        return chess.square(file_index, rank_index)

    def _square_for_rc(self, col: int, row: int) -> chess.Square:
        if self.orientation_white_bottom:
            return chess.square(col, 7 - row)
        return chess.square(7 - col, row)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        square = self.square_at(int(event.position().x()), int(event.position().y()))
        if square is not None:
            self.selected = square
            self.square_clicked.emit(int(square))
            self.update()

    def _tile_geometry(self) -> tuple[int, int, float]:
        side = min(self.width(), self.height())
        return (self.width() - side) // 2, (self.height() - side) // 2, side / 8.0

    def _center(self, square: chess.Square) -> QPointF:
        ox, oy, tile = self._tile_geometry()
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        col, row = (file_index, 7 - rank_index) if self.orientation_white_bottom else (7 - file_index, rank_index)
        return QPointF(ox + (col + 0.5) * tile, oy + (row + 0.5) * tile)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ox, oy, tile = self._tile_geometry()
        light = QColor(240, 217, 181)
        dark = QColor(181, 136, 99)
        for row in range(8):
            for col in range(8):
                square = self._square_for_rc(col, row)
                rect = QRect(int(ox + col * tile), int(oy + row * tile), int(tile) + 1, int(tile) + 1)
                color = light if (chess.square_file(square) + chess.square_rank(square)) % 2 else dark
                if square in self.highlights:
                    color = QColor(118, 186, 112)
                if square == self.selected:
                    color = QColor(246, 198, 86)
                painter.fillRect(rect, color)
                piece = self.board.piece_at(square)
                if piece:
                    painter.setPen(QColor(25, 25, 25) if piece.color == chess.WHITE else QColor(5, 5, 5))
                    painter.setFont(QFont("Segoe UI Symbol", max(8, int(tile * 0.62))))
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, UNICODE[piece.symbol()])
                if square in self.unresolved:
                    painter.setPen(QPen(QColor(230, 126, 34), 3))
                    painter.drawRect(rect.adjusted(2, 2, -2, -2))
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "?")
                conf = self.confidences.get(int(square))
                if conf is not None and (piece or square in self.unresolved):
                    if conf < self.confidence_threshold:
                        painter.setPen(QPen(QColor(192, 57, 43), 2))
                        painter.drawRect(rect.adjusted(2, 2, -2, -2))
                    painter.setPen(QColor(20, 25, 30))
                    painter.setFont(QFont("Segoe UI", max(6, int(tile * 0.12))))
                    painter.drawText(
                        rect.adjusted(2, 2, -3, -2),
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                        f"{conf:.0%}",
                    )
                painter.setPen(QColor(45, 45, 45, 150))
                painter.setFont(QFont("Segoe UI", max(5, int(tile * 0.10))))
                if row == 7:
                    painter.drawText(
                        rect.adjusted(3, 1, -2, -2),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                        chess.FILE_NAMES[chess.square_file(square)],
                    )
                if col == 0:
                    painter.drawText(
                        rect.adjusted(3, 2, -2, -2),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                        str(chess.square_rank(square) + 1),
                    )
        for index, (start, end) in enumerate(self.arrow_moves):
            self._draw_arrow(painter, self._center(start), self._center(end), ARROW_COLORS[index])

    @staticmethod
    def _draw_arrow(painter: QPainter, start: QPointF, end: QPointF, color: QColor) -> None:
        painter.save()
        pen = QPen(color, 6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(start, end)
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        left = QPointF(end.x() - 20 * math.cos(angle - 0.45), end.y() - 20 * math.sin(angle - 0.45))
        right = QPointF(end.x() - 20 * math.cos(angle + 0.45), end.y() - 20 * math.sin(angle + 0.45))
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([end, left, right]))
        painter.restore()
