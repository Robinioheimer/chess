from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np

from stockfish_vision.recognition.glyphs import render_piece_glyph
from stockfish_vision.recognition.hog_svm import PIECE_CLASSES, hog_features, train_and_save
from stockfish_vision.recognition.pieces import PieceClassifier, occupancy_score


def test_occupancy_empty_vs_piece() -> None:
    empty = np.full((40, 40, 3), 200, dtype=np.uint8)
    piece = render_piece_glyph("N", size=40, light_square=True, jitter=False)
    piece_bgr = cv2.cvtColor(piece, cv2.COLOR_RGB2BGR)
    assert occupancy_score(empty) < occupancy_score(piece_bgr)


def test_train_svm_and_predict_glyph(tmp_path: Path) -> None:
    model = train_and_save(path=tmp_path / "svm.xml", samples_per_class=20, seed=3)
    assert model.exists()
    clf = PieceClassifier()
    from stockfish_vision.recognition import hog_svm

    clf._svm = hog_svm.load_svm(model)
    image = render_piece_glyph("Q", size=32, light_square=False, jitter=False)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    piece, conf, occupied, source = clf.classify_square(bgr)
    assert occupied
    assert piece is not None
    assert piece.piece_type == chess.QUEEN
    assert source in {"svm", "template", "fallback"}
    assert conf > 0.2
    feats = hog_features(image)
    assert feats.shape[1] > 10
    assert set(PIECE_CLASSES) == {
        "P",
        "N",
        "B",
        "R",
        "Q",
        "K",
        "p",
        "n",
        "b",
        "r",
        "q",
        "k",
    }
