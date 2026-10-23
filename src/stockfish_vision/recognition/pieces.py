from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import chess
import cv2
import numpy as np

from stockfish_vision.board.geometry import extract_square, image_rc_to_square
from stockfish_vision.chesslogic.fenutil import board_from_piece_map, describe_validity
from stockfish_vision.paths import app_data_dir
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
    "default_template_path",
    "normalize_tile",
    "occupancy_score",
]


def default_template_path() -> Path:
    """Ablage der gelernten Figurenvorlagen im Benutzerprofil."""
    return _default_template_path()


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


def normalize_tile(square_img: np.ndarray, size: int = 32) -> np.ndarray | None:
    """Schneidet die Figur frei und normiert sie auf ``size`` × ``size``.

    Der Hintergrund wird aus den Rändern des Feldes geschätzt, die Figur über
    die Abweichung davon maskiert und auf ihre Bounding-Box beschnitten.
    Dadurch sind Vorlagen unabhängig von Feldgröße, Figurengröße und
    Position im Feld – der entscheidende Punkt bei echten Screenshots.
    """

    gray = _gray(square_img)
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return None
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    border = np.concatenate(
        (
            gray[:3].ravel(),
            gray[-3:].ravel(),
            gray[3:-3, :3].ravel(),
            gray[3:-3, -3:].ravel(),
        )
    )
    background = float(np.median(border))
    difference = np.abs(gray.astype(np.float32) - background)
    threshold = max(10.0, float(np.percentile(difference, 60)))
    mask = difference > threshold
    if float(mask.mean()) < 0.02:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])
    if bottom - top < 3 or right - left < 3:
        return None
    # Quadratisch machen, damit die Seitenverhältnisse erhalten bleiben.
    height = bottom - top + 1
    width = right - left + 1
    side = max(height, width)
    pad_y = (side - height) // 2
    pad_x = (side - width) // 2
    top = max(0, top - pad_y)
    left = max(0, left - pad_x)
    patch = gray[top : top + side, left : left + side]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def _default_template_path() -> Path:
    return app_data_dir() / "templates" / "piece_templates.npz"


@dataclass
class TemplateBank:
    """Gelernte Vorlagen des tatsächlich verwendeten Figurensatzes."""

    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)
    max_samples_per_class: int = 8
    size: int = 32

    def add(self, symbol: str, square_img: np.ndarray) -> bool:
        if symbol not in PIECE_CLASSES:
            raise ValueError(f"Unbekannte Figurenklasse: {symbol}")
        normalized = normalize_tile(square_img, size=self.size)
        if normalized is None:
            return False
        samples = self.templates.setdefault(symbol, [])
        samples.append(normalized)
        del samples[: -self.max_samples_per_class]
        return True

    def match(self, square_img: np.ndarray) -> tuple[str | None, float]:
        """Beste Vorlage und Korrelation (0..1)."""

        if not self.templates:
            return None, 0.0
        normalized = normalize_tile(square_img, size=self.size)
        if normalized is None:
            return None, 0.0
        best_symbol: str | None = None
        best_correlation = -1.0
        for symbol, samples in self.templates.items():
            for sample in samples:
                if sample.shape != normalized.shape:
                    continue
                result = cv2.matchTemplate(normalized, sample, cv2.TM_CCOEFF_NORMED)
                correlation = float(result.max())
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_symbol = symbol
        return best_symbol, max(0.0, min(1.0, (best_correlation + 1.0) / 2.0))

    # ---------------------------------------------------------- Persistenz
    @property
    def symbols(self) -> list[str]:
        return sorted(self.templates)

    def __len__(self) -> int:
        return sum(len(samples) for samples in self.templates.values())

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for symbol, samples in self.templates.items():
            for index, sample in enumerate(samples):
                arrays[f"{symbol}_{index}"] = sample.astype(np.uint8)
        return arrays

    def save(self, path: Path | None = None) -> Path:
        target = Path(path) if path is not None else _default_template_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **self.to_arrays())
        return target

    @classmethod
    def load(cls, path: Path | None = None, size: int = 32) -> "TemplateBank":
        source = Path(path) if path is not None else _default_template_path()
        bank = cls(size=size)
        with np.load(source) as data:
            for key in data.files:
                symbol, _, _index = key.partition("_")
                if symbol not in PIECE_CLASSES:
                    continue
                bank.templates.setdefault(symbol, []).append(np.asarray(data[key], dtype=np.uint8))
        return bank


class PieceClassifier:
    def __init__(
        self,
        occupancy_threshold: float = 18.0,
        min_confidence: float = 0.45,
        model_path: str | Path | None = None,
        template_path: str | Path | None = None,
    ) -> None:
        self.occupancy_threshold = float(occupancy_threshold)
        self.min_confidence = float(min_confidence)
        self.templates = TemplateBank()
        self.template_path = Path(template_path) if template_path else _default_template_path()
        self.template_threshold = 0.78
        self.template_error: str | None = None
        self.last_corrections: list[Correction] = []
        self.reload_templates()
        self._svm = None
        self._scorer: SvmDecisionScorer | None = None
        self.model_error: str | None = None
        try:
            self._svm = load_svm(Path(model_path) if model_path else None)
            self._scorer = SvmDecisionScorer(self._svm)
        except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as exc:
            self.model_error = str(exc)
            logger.error("Figurenmodell nicht verfügbar: %s", exc)

    def reload_templates(self) -> int:
        """Lädt gelernte Vorlagen des Benutzers (falls vorhanden)."""

        try:
            if not self.template_path.exists():
                self.templates = TemplateBank()
                return 0
            self.templates = TemplateBank.load(self.template_path)
            self.template_error = None
            return len(self.templates)
        except (OSError, ValueError, KeyError) as exc:
            self.template_error = f"Vorlagen konnten nicht geladen werden: {exc}"
            logger.warning(self.template_error)
            self.templates = TemplateBank()
            return 0

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
        if symbol and template_score >= self.template_threshold:
            return chess.Piece.from_symbol(symbol), template_score, True, "vorlage", {}

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
