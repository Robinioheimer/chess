from __future__ import annotations

import random

import chess
import cv2
import numpy as np

from stockfish_vision.board.geometry import square_to_image_rc
from stockfish_vision.recognition.calibrate import learn_templates, standard_start_symbols
from stockfish_vision.recognition.glyphs import render_piece_glyph
from stockfish_vision.recognition.pieces import TemplateBank, normalize_tile


def _board_image(
    symbols: dict[chess.Square, str],
    tile_size: int = 48,
    jitter: bool = False,
    scale: float = 1.0,
    seed: int = 3,
) -> np.ndarray:
    warped = np.zeros((tile_size * 8, tile_size * 8, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            light = (row + col) % 2 == 0
            rgb = np.array((226, 214, 187) if light else (96, 122, 158), dtype=np.uint8)
            warped[
                row * tile_size : (row + 1) * tile_size,
                col * tile_size : (col + 1) * tile_size,
            ] = cv2.cvtColor(np.tile(rgb, (tile_size, tile_size, 1)), cv2.COLOR_RGB2BGR)
    for square, symbol in symbols.items():
        col, row = square_to_image_rc(square, "white_bottom")
        light = (row + col) % 2 == 0
        piece_size = max(12, int(tile_size * scale))
        glyph = render_piece_glyph(
            symbol,
            size=piece_size,
            light_square=light,
            jitter=jitter,
            rng=random.Random(seed + square),
        )
        offset = (tile_size - piece_size) // 2
        region = warped[
            row * tile_size + offset : row * tile_size + offset + piece_size,
            col * tile_size + offset : col * tile_size + offset + piece_size,
        ]
        region[:, :] = cv2.cvtColor(glyph, cv2.COLOR_RGB2BGR)
    return warped


def test_normalize_tile_is_position_and_size_invariant() -> None:
    image = render_piece_glyph("Q", size=64, light_square=True)

    def variant(size: int, shift: int) -> np.ndarray:
        canvas = np.full((96, 96), 235, dtype=np.uint8)
        glyph = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (size, size))
        canvas[shift : shift + size, shift : shift + size] = glyph
        return canvas

    first = normalize_tile(variant(48, 8))
    second = normalize_tile(variant(34, 20))
    assert first is not None and second is not None
    correlation = cv2.matchTemplate(first, second, cv2.TM_CCOEFF_NORMED).max()
    assert correlation > 0.9


def test_learned_templates_recognise_a_different_rendering() -> None:
    symbols = standard_start_symbols()
    bank = TemplateBank()
    report = learn_templates(
        _board_image(symbols, tile_size=48),
        "white_bottom",
        symbols,
        bank=bank,
        persist=False,
    )
    assert report.symbols == 12
    assert report.samples == 32

    # Andere Feldgröße, leichte Verschiebung/Verzerrung: neue Vorlage nötig?
    harder = _board_image(symbols, tile_size=56, jitter=True, seed=11)
    hits = 0
    for square, symbol in symbols.items():
        col, row = square_to_image_rc(square, "white_bottom")
        tile = harder[row * 56 : (row + 1) * 56, col * 56 : (col + 1) * 56]
        predicted, score = bank.match(tile)
        hits += predicted == symbol
    assert hits >= 29, f"nur {hits}/32 Figuren wiedererkannt"


def test_templates_survive_roundtrip(tmp_path) -> None:
    symbols = standard_start_symbols()
    bank = TemplateBank()
    learn_templates(
        _board_image(symbols), "white_bottom", symbols, bank=bank, persist=False
    )
    path = bank.save(tmp_path / "templates.npz")
    loaded = TemplateBank.load(path)
    assert len(loaded) == len(bank)
    assert loaded.symbols == bank.symbols

    col, row = square_to_image_rc(chess.E1, "white_bottom")
    tile = _board_image(symbols)[row * 48 : (row + 1) * 48, col * 48 : (col + 1) * 48]
    assert loaded.match(tile)[0] == "K"


def test_learning_rejects_empty_image() -> None:
    with np.errstate(all="ignore"):
        try:
            learn_templates(None, "white_bottom", persist=False)  # type: ignore[arg-type]
        except ValueError:
            return
    raise AssertionError("leeres Bild muss abgelehnt werden")


def _svg_tile(symbol: str, size: int, light: bool) -> np.ndarray:
    """Rendert einen cburnett-Figurensatz (echte, nicht-synthetische Vorlage)."""

    import chess.svg
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    background = (226, 214, 187) if light else (96, 122, 158)
    image = QImage(size, size, QImage.Format_RGB888)
    image.fill(QColor(*background))
    renderer = QSvgRenderer(QByteArray(chess.svg.piece(chess.Piece.from_symbol(symbol), size=size).encode()))
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    buffer = image.constBits()
    array = np.frombuffer(buffer, dtype=np.uint8).reshape(image.height(), image.bytesPerLine() // 3, 3)
    return np.ascontiguousarray(array[:, : image.width(), :])


def _svg_board(symbols: dict[chess.Square, str], tile: int, piece_scale: float, offset: int) -> np.ndarray:
    board = np.zeros((tile * 8, tile * 8, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            light = (row + col) % 2 == 0
            rgb = (226, 214, 187) if light else (96, 122, 158)
            board[row * tile : (row + 1) * tile, col * tile : (col + 1) * tile] = rgb[::-1]
    piece_size = max(8, int(tile * piece_scale))
    for square, symbol in symbols.items():
        col, row = square_to_image_rc(square, "white_bottom")
        tile_img = _svg_tile(symbol, piece_size, (row + col) % 2 == 0)
        top = row * tile + offset
        left = col * tile + offset
        board[top : top + piece_size, left : left + piece_size] = np.ascontiguousarray(tile_img[:, :, ::-1])
    return board


def test_learned_templates_beat_the_generic_model_on_real_piece_set() -> None:
    """Der entscheidende Praxisfall: chess.com-artige Figuren statt Glyphen."""

    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.instance() is None:
        QGuiApplication([])

    start = standard_start_symbols()
    bank = TemplateBank()
    learn_templates(
        _svg_board(start, tile=40, piece_scale=1.0, offset=0),
        "white_bottom",
        start,
        bank=bank,
        persist=False,
    )
    assert report_symbols(bank) == 12

    # Gleiche Figuren, anderer Screenshot: andere Feldgröße und Position.
    mill = {
        chess.A1: "R", chess.B1: "N", chess.C1: "B", chess.D1: "Q", chess.E1: "K",
        chess.F1: "B", chess.G1: "N", chess.H1: "R",
        chess.A8: "r", chess.B8: "n", chess.C8: "b", chess.D8: "q", chess.E8: "k",
        chess.F8: "b", chess.G8: "n", chess.H8: "r",
        chess.A2: "P", chess.B2: "P", chess.C2: "P", chess.D2: "P",
        chess.E4: "P", chess.D5: "P", chess.F2: "P", chess.G2: "P", chess.H2: "P",
        chess.A7: "p", chess.B7: "p", chess.C7: "p", chess.D7: "p",
        chess.E5: "p", chess.D4: "p", chess.F7: "p", chess.G7: "p", chess.H7: "p",
    }
    tile = 64
    board = _svg_board(mill, tile=tile, piece_scale=0.82, offset=5)
    hits = 0
    for square, symbol in mill.items():
        col, row = square_to_image_rc(square, "white_bottom")
        predicted, _score = bank.match(board[row * tile : (row + 1) * tile, col * tile : (col + 1) * tile])
        hits += predicted == symbol
    # Mit dem generischen Modell waren es auf echten Figurensätzen nur ~17 %.
    assert hits >= 28, f"nur {hits}/32 Figuren im zweiten Screenshot erkannt"


def report_symbols(bank: TemplateBank) -> int:
    return len(bank.symbols)
