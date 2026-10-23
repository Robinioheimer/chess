from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stockfish_vision.paths import models_dir
from stockfish_vision.recognition.glyphs import render_piece_glyph

PIECE_CLASSES = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
MODEL_NAME = "piece_hog_svm.xml"


def hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def hog_features(bgr_or_rgb: np.ndarray, hog: cv2.HOGDescriptor | None = None) -> np.ndarray:
    hog = hog or hog_descriptor()
    resized = cv2.resize(bgr_or_rgb, (32, 32), interpolation=cv2.INTER_AREA)
    if resized.ndim == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = resized
    feats = hog.compute(gray)
    if feats is None:
        raise RuntimeError("HOG-Berechnung fehlgeschlagen.")
    return feats.reshape(1, -1).astype(np.float32)


def default_model_path() -> Path:
    return models_dir() / MODEL_NAME


def train_and_save(
    path: Path | None = None,
    samples_per_class: int = 48,
    seed: int = 7,
) -> Path:
    rng = np.random.default_rng(seed)
    py_rng = __import__("random").Random(seed)
    hog = hog_descriptor()
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, symbol in enumerate(PIECE_CLASSES):
        for _ in range(samples_per_class):
            light = bool(rng.integers(0, 2))
            image = render_piece_glyph(symbol, size=32, light_square=light, jitter=True, rng=py_rng)
            rows.append(hog_features(image, hog))
            labels.append(class_index)
    samples = np.vstack(rows)
    label_array = np.array(labels, dtype=np.int32)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2.5)
    svm.setGamma(0.03)
    svm.train(samples, cv2.ml.ROW_SAMPLE, label_array)
    target = path or default_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(target))
    return target


def load_svm(path: Path | None = None) -> cv2.ml_SVM:
    target = path or default_model_path()
    if not target.exists():
        raise FileNotFoundError(f"Kein Figurenmodell unter {target}")
    svm = cv2.ml.SVM_load(str(target))
    if svm is None:
        raise RuntimeError(f"Modell konnte nicht geladen werden: {target}")
    return svm
