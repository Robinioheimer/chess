from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stockfish_vision.paths import models_dir
from stockfish_vision.recognition.glyphs import render_piece_glyph

PIECE_CLASSES = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
MODEL_NAME = "piece_hog_svm.xml"
HOG_SIZE = (32, 32)


def hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=HOG_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def hog_features(bgr_or_rgb: np.ndarray, hog: cv2.HOGDescriptor | None = None) -> np.ndarray:
    hog = hog or hog_descriptor()
    resized = cv2.resize(bgr_or_rgb, HOG_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY) if resized.ndim == 3 else resized
    feats = hog.compute(gray)
    if feats is None:
        raise RuntimeError("HOG-Berechnung fehlgeschlagen.")
    return feats.reshape(1, -1).astype(np.float32)


def default_model_path() -> Path:
    return models_dir() / MODEL_NAME


@dataclass(frozen=True)
class SvmPrediction:
    symbol: str
    confidence: float
    scores: dict[str, float]


class SvmDecisionScorer:
    """Calculate multiclass scores from OpenCV's pairwise SVM margins.

    OpenCV does not expose calibrated probabilities for a multiclass SVM. The
    confidence is therefore explicitly a heuristic: for every pair, the RBF
    decision margin is mapped through a sigmoid; a class score is its mean
    pairwise win score. The predicted class still follows OpenCV's vote rule.
    """

    def __init__(self, svm: cv2.ml_SVM, classes: tuple[str, ...] = PIECE_CLASSES) -> None:
        if not svm.isTrained():
            raise RuntimeError("Das Figurenmodell ist nicht trainiert.")
        self.svm = svm
        self.classes = classes
        self.support_vectors = np.asarray(svm.getSupportVectors(), dtype=np.float32)
        self.gamma = float(svm.getGamma())
        self._decisions: list[tuple[float, np.ndarray, np.ndarray]] = []
        class_count = len(classes)
        for index in range(class_count * (class_count - 1) // 2):
            rho, alpha, support_indices = svm.getDecisionFunction(index)
            self._decisions.append(
                (
                    float(rho),
                    np.asarray(alpha, dtype=np.float32).ravel(),
                    np.asarray(support_indices, dtype=np.int32).ravel(),
                )
            )
        # OpenCV orders pairs as (1,0), (2,0), ..., (11,0), (2,1), ...
        self._pairs = [
            (higher, lower)
            for lower in range(class_count)
            for higher in range(lower + 1, class_count)
        ]

    def predict(self, features: np.ndarray) -> SvmPrediction:
        sample = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if sample.shape[1] != self.support_vectors.shape[1]:
            raise ValueError("HOG-Merkmale passen nicht zum geladenen SVM-Modell.")
        delta = self.support_vectors - sample
        kernels = np.exp(-self.gamma * np.einsum("ij,ij->i", delta, delta))
        class_count = len(self.classes)
        votes = np.zeros(class_count, dtype=np.int32)
        pairwise_scores = np.zeros(class_count, dtype=np.float64)
        for (rho, alpha, indices), (higher, lower) in zip(self._decisions, self._pairs):
            margin = float(np.dot(alpha, kernels[indices]) - rho)
            lower_probability = 1.0 / (1.0 + np.exp(-float(np.clip(margin, -30.0, 30.0))))
            if margin > 0:
                votes[lower] += 1
            else:
                votes[higher] += 1
            pairwise_scores[lower] += lower_probability
            pairwise_scores[higher] += 1.0 - lower_probability
        pairwise_scores /= class_count - 1
        winner = int(np.argmax(votes))
        scores = {
            symbol: float(np.clip(pairwise_scores[index], 0.0, 1.0))
            for index, symbol in enumerate(self.classes)
        }
        return SvmPrediction(
            symbol=self.classes[winner],
            confidence=float(np.clip(pairwise_scores[winner], 0.0, 1.0)),
            scores=scores,
        )


def train_and_save(
    path: Path | None = None,
    samples_per_class: int = 48,
    seed: int = 7,
) -> Path:
    if samples_per_class < 4:
        raise ValueError("Mindestens vier Trainingsbeispiele pro Klasse sind erforderlich.")
    rng = np.random.default_rng(seed)
    py_rng = __import__("random").Random(seed)
    hog = hog_descriptor()
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, symbol in enumerate(PIECE_CLASSES):
        for _ in range(samples_per_class):
            light = bool(rng.integers(0, 2))
            image = render_piece_glyph(
                symbol, size=32, light_square=light, jitter=True, rng=py_rng
            )
            rows.append(hog_features(image, hog))
            labels.append(class_index)
    samples = np.vstack(rows)
    label_array = np.array(labels, dtype=np.int32)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2.5)
    svm.setGamma(0.03)
    if not svm.train(samples, cv2.ml.ROW_SAMPLE, label_array) or not svm.isTrained():
        raise RuntimeError("Das SVM-Training ist fehlgeschlagen.")
    target = (path or default_model_path()).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(target))
    metadata = {
        "format": 1,
        "model": target.name,
        "classes": list(PIECE_CLASSES),
        "features": "OpenCV HOG, 32x32, 8x8 cells, 9 bins, 324 values",
        "classifier": "OpenCV C-SVC, RBF, C=2.5, gamma=0.03",
        "samples_per_class": samples_per_class,
        "seed": seed,
        "training_data": "Synthetic glyphs generated by recognition.glyphs; no external images",
        "confidence": "Heuristic mean pairwise sigmoid margin; not a calibrated probability",
    }
    target.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target


def load_svm(path: Path | None = None) -> cv2.ml_SVM:
    target = (path or default_model_path()).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"Kein Figurenmodell unter {target}")
    try:
        svm = cv2.ml.SVM_load(str(target))
    except cv2.error as exc:
        raise RuntimeError(f"Modell konnte nicht geladen werden: {target}: {exc}") from exc
    if svm is None or not svm.isTrained():
        raise RuntimeError(f"Modell ist nicht trainiert oder ungültig: {target}")
    expected_features = hog_descriptor().getDescriptorSize()
    if svm.getVarCount() != expected_features:
        raise RuntimeError(
            f"Modell besitzt {svm.getVarCount()} statt {expected_features} HOG-Merkmale: {target}"
        )
    return svm
