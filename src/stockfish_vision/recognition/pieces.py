from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import chess
import cv2
import numpy as np

from stockfish_vision.board.geometry import extract_square, image_rc_to_square
from stockfish_vision.chesslogic.fenutil import board_from_piece_map, describe_validity
from stockfish_vision.recognition.hog_svm import (
    PIECE_CLASSES,
    SvmDecisionScorer,
    hog_features,
    load_svm,
)
from stockfish_vision.recognition.plausibility import Correction, repair_symbols

logger = logging.getLogger(__name__)

__all__ = [
    "PIECE_CLASSES",
    "Detection",
    "PieceClassifier",
    "TemplateBank",
    "occupancy_score",
]


def _gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Leeres Feld kann nicht klassifiziert werden.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def occupancy_score(square_img: np.ndarray) -> float:
    if square_img.size == 0:
        return 0.0
    gray = _gray(square_img)
    inner = gray[2:-2, 2:-2] if gray.shape[0] > 6 and gray.shape[1] > 6 else gray
    if gray.shape[0] > 6 and gray.shape[1] > 6:
        border = np.concatenate(
            (gray[:2].ravel(), gray[-2:].ravel(), gray[2:-2, :2].ravel(), gray[2:-2, -2:].ravel())
        )
        background = float(np.median(border))
        # Standard deviation alone misses flat, solid glyphs. Border contrast
        # detects those pieces while remaining modest on smooth empty squares.
        contrast = float(np.percentile(np.abs(inner.astype(np.float32) - background), 90))
        return max(float(np.std(inner)), 0.65 * contrast)
    return float(np.std(inner))


def estimate_piece_color(square_img: np.ndarray) -> chess.Color:
    return _estimate_piece_color(square_img)[0]


def _estimate_piece_color(square_img: np.ndarray) -> tuple[chess.Color, float]:
    gray = _gray(square_img)
    inner = gray[3:-3, 3:-3] if gray.shape[0] > 8 and gray.shape[1] > 8 else gray
    background = float(np.median(inner))
    mask = np.abs(inner.astype(np.float32) - background) > 18
    if mask.sum() < 8:
        color = chess.WHITE if float(np.mean(inner)) > 127 else chess.BLACK
        return color, 0.35
    figure_mean = float(np.mean(inner[mask]))
    color = chess.WHITE if figure_mean > 127 else chess.BLACK
    # This is only a colour heuristic, not a calibrated probability.
    confidence = float(np.clip(0.55 + abs(figure_mean - 127.0) / 255.0, 0.35, 0.99))
    return color, confidence


@dataclass
class Detection:
    square: chess.Square
    piece: chess.Piece | None
    confidence: float
    occupied: bool
    source: str
    alternatives: dict[str, float] = field(default_factory=dict)


@dataclass
class TemplateBank:
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)
    max_samples_per_class: int = 8

    def add(self, symbol: str, square_img: np.ndarray) -> None:
        if symbol not in PIECE_CLASSES:
            raise ValueError(f"Unbekannte Figurenklasse: {symbol}")
        gray = _gray(square_img)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        samples = self.templates.setdefault(symbol, [])
        samples.append(resized)
        del samples[: -self.max_samples_per_class]

    def match(self, square_img: np.ndarray) -> tuple[str | None, float]:
        if not self.templates:
            return None, 0.0
        resized = cv2.resize(_gray(square_img), (32, 32), interpolation=cv2.INTER_AREA)
        best_symbol: str | None = None
        best_correlation = -1.0
        for symbol, samples in self.templates.items():
            for sample in samples[-4:]:
                result = cv2.matchTemplate(resized, sample, cv2.TM_CCOEFF_NORMED)
                correlation = float(result.max())
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_symbol = symbol
        # Preserve the historic 0..1 correlation score used by the GUI.
        return best_symbol, max(0.0, min(1.0, (best_correlation + 1.0) / 2.0))


class PieceClassifier:
    def __init__(
        self,
        occupancy_threshold: float = 18.0,
        min_confidence: float = 0.45,
        model_path: str | Path | None = None,
    ) -> None:
        self.occupancy_threshold = float(occupancy_threshold)
        self.min_confidence = float(min_confidence)
        self.templates = TemplateBank()
        self.last_corrections: list[Correction] = []
        self._svm = None
        self._scorer: SvmDecisionScorer | None = None
        self.model_error: str | None = None
        try:
            self._svm = load_svm(Path(model_path) if model_path else None)
            self._scorer = SvmDecisionScorer(self._svm)
        except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as exc:
            self.model_error = str(exc)
            logger.error("Figurenmodell nicht verfügbar: %s", exc)

    def classify_square(
        self, square_img: np.ndarray
    ) -> tuple[chess.Piece | None, float, bool, str]:
        piece, confidence, occupied, source, _scores = self._classify(square_img)
        return piece, confidence, occupied, source

    def _classify(
        self, square_img: np.ndarray
    ) -> tuple[chess.Piece | None, float, bool, str, dict[str, float]]:
        score = occupancy_score(square_img)
        occupied = score >= self.occupancy_threshold
        if not occupied:
            margin = (self.occupancy_threshold - score) / max(self.occupancy_threshold, 1.0)
            confidence = float(np.clip(0.5 + 0.5 * margin, 0.5, 0.99))
            return None, confidence, False, "empty", {}

        symbol, template_score = self.templates.match(square_img)
        if symbol and template_score >= 0.72:
            return chess.Piece.from_symbol(symbol), template_score, True, "template", {}

        prediction = self._svm_predict(square_img)
        if prediction is not None:
            piece = chess.Piece.from_symbol(prediction.symbol)
            color, color_confidence = _estimate_piece_color(square_img)
            source = "svm"
            confidence = prediction.confidence
            if piece.color != color:
                piece = chess.Piece(piece.piece_type, color)
                confidence *= 0.85 + 0.15 * color_confidence
            alternatives = dict(
                sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True)[:3]
            )
            return piece, float(confidence), True, source, alternatives

        # Never invent a pawn when no classifier is available or prediction fails.
        return None, 0.0, True, "unresolved", {}

    def _svm_predict(self, square_img: np.ndarray):
        if self._svm is None:
            return None
        rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB) if square_img.ndim == 3 else square_img
        try:
            if self._scorer is None or self._scorer.svm is not self._svm:
                self._scorer = SvmDecisionScorer(self._svm)
            return self._scorer.predict(hog_features(rgb))
        except (cv2.error, ValueError, RuntimeError) as exc:
            logger.warning("SVM-Klassifikation fehlgeschlagen: %s", exc)
            return None

    def detect_board(
        self, warped: np.ndarray, orientation: str, turn: chess.Color = chess.WHITE
    ) -> tuple[chess.Board, list[Detection], tuple[bool, list[str]]]:
        pieces: dict[chess.Square, chess.Piece | None] = {}
        detections: list[Detection] = []
        for row in range(8):
            for col in range(8):
                square = image_rc_to_square(col, row, orientation)
                tile = extract_square(warped, col, row)
                piece, confidence, occupied, source, scores = self._classify(tile)
                pieces[square] = piece
                detections.append(
                    Detection(
                        square=square,
                        piece=piece,
                        confidence=confidence,
                        occupied=occupied,
                        source=source,
                        alternatives=scores,
                    )
                )
        symbols = {
            square: (piece.symbol() if piece is not None else None)
            for square, piece in pieces.items()
        }
        scores = {detection.square: detection.alternatives for detection in detections}
        symbols, corrections = repair_symbols(symbols, scores)
        self.last_corrections = corrections
        if corrections:
            for index, detection in enumerate(detections):
                symbol = symbols.get(detection.square)
                piece = chess.Piece.from_symbol(symbol) if symbol else None
                if piece != detection.piece:
                    detections[index] = replace(detection, piece=piece)
        pieces = {
            square: (chess.Piece.from_symbol(symbol) if symbol else None)
            for square, symbol in symbols.items()
        }
        board = board_from_piece_map(pieces, turn=turn)
        ok, issues = describe_validity(board)
        unresolved = [d.square for d in detections if d.occupied and d.piece is None]
        if corrections:
            issues.append(
                "Plausibilität korrigiert: "
                + "; ".join(item.description for item in corrections)
            )
        if unresolved:
            ok = False
            names = ", ".join(chess.square_name(square) for square in unresolved)
            issues.append(f"Belegte Felder ohne zuverlässige Figurenklasse: {names}.")
        return board, detections, (ok, issues)

    def remember_correction(self, square_img: np.ndarray, symbol: str | None) -> None:
        if symbol:
            self.templates.add(symbol, square_img)
