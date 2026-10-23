from __future__ import annotations

import chess
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
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


class BoardWidget(QWidget):
    square_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.board = chess.Board()
        self.orientation_white_bottom = True
        self.highlights: set[int] = set()
        self.selected: chess.Square | None = None
        self.confidences: dict[int, float] = {}
        self.setMinimumSize(360, 360)

    def set_board(self, board: chess.Board) -> None:
        self.board = board.copy(stack=False)
        self.update()

    def square_at(self, x: int, y: int) -> chess.Square | None:
        side = min(self.width(), self.height())
        ox = (self.width() - side) // 2
        oy = (self.height() - side) // 2
        if not (ox <= x < ox + side and oy <= y < oy + side):
            return None
        col = int((x - ox) / side * 8)
        row = int((y - oy) / side * 8)
        col = min(max(col, 0), 7)
        row = min(max(row, 0), 7)
        if self.orientation_white_bottom:
            file_index, rank_index = col, 7 - row
        else:
            file_index, rank_index = 7 - col, row
        return chess.square(file_index, rank_index)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        square = self.square_at(int(event.position().x()), int(event.position().y()))
        if square is not None:
            self.selected = square
            self.square_clicked.emit(int(square))
            self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        ox = (self.width() - side) // 2
        oy = (self.height() - side) // 2
        tile = side / 8.0
        light = QColor(240, 217, 181)
        dark = QColor(181, 136, 99)
        for row in range(8):
            for col in range(8):
                if self.orientation_white_bottom:
                    square = chess.square(col, 7 - row)
                else:
                    square = chess.square(7 - col, row)
                rect = QRect(int(ox + col * tile), int(oy + row * tile), int(tile) + 1, int(tile) + 1)
                color = light if (col + row) % 2 == 0 else dark
                if square in self.highlights:
                    color = QColor(118, 186, 112)
                if square == self.selected:
                    color = QColor(246, 198, 86)
                painter.fillRect(rect, color)
                piece = self.board.piece_at(square)
                if piece:
                    painter.setPen(QColor(20, 20, 20) if piece.color == chess.WHITE else QColor(10, 10, 10))
                    painter.setFont(QFont("Segoe UI Symbol", int(tile * 0.62)))
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, UNICODE[piece.symbol()])
                conf = self.confidences.get(int(square))
                if conf is not None and conf < 0.5 and piece:
                    painter.setPen(QPen(QColor(192, 57, 43), 2))
                    painter.drawRect(rect.adjusted(2, 2, -2, -2))
