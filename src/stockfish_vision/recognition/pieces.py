from __future__ import annotations

from dataclasses import dataclass, field

import chess
import cv2
import numpy as np

from stockfish_vision.board.geometry import extract_square, image_rc_to_square
from stockfish_vision.chesslogic.fenutil import board_from_piece_map, describe_validity
from stockfish_vision.recognition.hog_svm import PIECE_CLASSES, hog_features, load_svm

__all__ = [
    "PIECE_CLASSES",
    "Detection",
    "PieceClassifier",
    "TemplateBank",
    "occupancy_score",
]


def occupancy_score(square_img: np.ndarray) -> float:
    if square_img.size == 0:
        return 0.0
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY) if square_img.ndim == 3 else square_img
    inner = gray[2:-2, 2:-2] if gray.shape[0] > 6 and gray.shape[1] > 6 else gray
    return float(np.std(inner))


def estimate_piece_color(square_img: np.ndarray) -> chess.Color:
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY) if square_img.ndim == 3 else square_img
    inner = gray[3:-3, 3:-3] if gray.shape[0] > 8 else gray
    background = np.median(inner)
    mask = np.abs(inner.astype(np.float32) - float(background)) > 18
    if mask.sum() < 8:
        return chess.WHITE if float(np.mean(inner)) > 127 else chess.BLACK
    figure_pixels = inner[mask]
    return chess.WHITE if float(np.mean(figure_pixels)) > 127 else chess.BLACK


@dataclass
class Detection:
    square: chess.Square
    piece: chess.Piece | None
    confidence: float
    occupied: bool
    source: str


@dataclass
class TemplateBank:
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, symbol: str, square_img: np.ndarray) -> None:
        gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY) if square_img.ndim == 3 else square_img
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        self.templates.setdefault(symbol, []).append(resized)

    def match(self, square_img: np.ndarray) -> tuple[str | None, float]:
        if not self.templates:
            return None, 0.0
        gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY) if square_img.ndim == 3 else square_img
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        best_symbol = None
        best_score = -1.0
        for symbol, samples in self.templates.items():
            for sample in samples[-4:]:
                result = cv2.matchTemplate(resized, sample, cv2.TM_CCOEFF_NORMED)
                score = float(result.max())
                if score > best_score:
                    best_score = score
                    best_symbol = symbol
        return best_symbol, max(0.0, min(1.0, (best_score + 1.0) / 2.0))


class PieceClassifier:
    def __init__(self, occupancy_threshold: float = 18.0, min_confidence: float = 0.45):
        self.occupancy_threshold = occupancy_threshold
        self.min_confidence = min_confidence
        self.templates = TemplateBank()
        self._svm = None
        self.model_error: str | None = None
        try:
            self._svm = load_svm()
        except (FileNotFoundError, RuntimeError) as exc:
            self.model_error = str(exc)

    def classify_square(self, square_img: np.ndarray) -> tuple[chess.Piece | None, float, bool, str]:
        occupied = occupancy_score(square_img) >= self.occupancy_threshold
        if not occupied:
            return None, 0.95, False, "empty"

        symbol, template_score = self.templates.match(square_img)
        if symbol and template_score >= 0.72:
            return chess.Piece.from_symbol(symbol), template_score, True, "template"

        svm_piece, svm_score = self._svm_predict(square_img)
        if svm_piece is not None:
            color = estimate_piece_color(square_img)
            if svm_piece.color != color:
                svm_piece = chess.Piece(svm_piece.piece_type, color)
            return svm_piece, svm_score, True, "svm"

        color = estimate_piece_color(square_img)
        fallback = chess.Piece(chess.PAWN, color)
        return fallback, 0.25, True, "fallback"

    def _svm_predict(self, square_img: np.ndarray) -> tuple[chess.Piece | None, float]:
        if self._svm is None:
            return None, 0.0
        rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB) if square_img.ndim == 3 else square_img
        feats = hog_features(rgb)
        try:
            _retval, result = self._svm.predict(feats)
            class_index = int(result[0, 0])
            symbol = PIECE_CLASSES[class_index]
            return chess.Piece.from_symbol(symbol), 0.62
        except cv2.error:
            return None, 0.0

    def detect_board(
        self, warped: np.ndarray, orientation: str, turn: chess.Color = chess.WHITE
    ) -> tuple[chess.Board, list[Detection], tuple[bool, list[str]]]:
        pieces: dict[chess.Square, chess.Piece | None] = {}
        detections: list[Detection] = []
        for row in range(8):
            for col in range(8):
                square = image_rc_to_square(col, row, orientation)
                tile = extract_square(warped, col, row)
                piece, confidence, occupied, source = self.classify_square(tile)
                pieces[square] = piece
                detections.append(
                    Detection(
                        square=square,
                        piece=piece,
                        confidence=confidence,
                        occupied=occupied,
                        source=source,
                    )
                )
        board = board_from_piece_map(pieces, turn=turn)
        validity = describe_validity(board)
        return board, detections, validity

    def remember_correction(self, square_img: np.ndarray, symbol: str | None) -> None:
        if not symbol:
            return
        self.templates.add(symbol, square_img)
