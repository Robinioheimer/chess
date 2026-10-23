from __future__ import annotations

import random
from pathlib import Path

import cv2

from stockfish_vision.recognition.glyphs import render_piece_glyph
from stockfish_vision.recognition.hog_svm import (
    PIECE_CLASSES,
    SvmDecisionScorer,
    hog_features,
    load_svm,
)
from stockfish_vision.recognition.pieces import PieceClassifier


def test_svm_scores_match_native_votes_and_vary() -> None:
    svm = load_svm()
    scorer = SvmDecisionScorer(svm)
    confidences = []
    for index, symbol in enumerate(PIECE_CLASSES):
        image = render_piece_glyph(
            symbol, size=32, jitter=True, rng=random.Random(100 + index)
        )
        features = hog_features(image)
        prediction = scorer.predict(features)
        native_index = int(svm.predict(features)[1][0, 0])
        assert prediction.symbol == PIECE_CLASSES[native_index]
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.scores) == 12
        confidences.append(round(prediction.confidence, 6))
    assert len(set(confidences)) > 4  # no fixed pseudo-confidence


def test_missing_model_never_invents_a_pawn(tmp_path: Path) -> None:
    classifier = PieceClassifier(model_path=tmp_path / "missing.xml")
    image = render_piece_glyph("N", size=40, jitter=False)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    piece, confidence, occupied, source = classifier.classify_square(bgr)
    assert occupied
    assert piece is None
    assert confidence == 0.0
    assert source == "unresolved"
    assert classifier.model_error is not None


def test_manual_template_is_reused() -> None:
    classifier = PieceClassifier(model_path=Path("/definitely/missing/model.xml"))
    image = render_piece_glyph("R", size=40, jitter=False)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    classifier.remember_correction(bgr, "R")
    piece, confidence, occupied, source = classifier.classify_square(bgr)
    assert occupied
    assert piece is not None and piece.symbol() == "R"
    assert source == "template"
    assert confidence > 0.99
