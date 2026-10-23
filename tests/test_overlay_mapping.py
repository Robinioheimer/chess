from __future__ import annotations

from stockfish_vision.overlay.mapping import physical_to_logical, scaled_to_logical


def test_physical_to_logical_with_150_percent_scaling() -> None:
    screens = [(0.0, 0.0, 1280.0, 720.0, 1.5)]
    assert physical_to_logical(0.0, 0.0, screens) == (0.0, 0.0)
    assert physical_to_logical(150.0, 300.0, screens) == (100.0, 200.0)
    # Rechte untere Ecke bleibt innerhalb des Fensters.
    assert physical_to_logical(1919.0, 1079.0, screens) == (1279.3333333333333, 719.3333333333334)


def test_physical_to_logical_uses_second_monitor_for_negative_origin() -> None:
    screens = [
        (0.0, 0.0, 1280.0, 720.0, 1.25),
        (-1920.0, 0.0, 1920.0, 1080.0, 1.0),
    ]
    assert physical_to_logical(-1000.0, 500.0, screens) == (-1000.0, 500.0)
    assert physical_to_logical(320.0, 320.0, screens) == (256.0, 256.0)


def test_physical_to_logical_falls_back_to_identity_without_match() -> None:
    assert physical_to_logical(4000.0, 4000.0, []) == (4000.0, 4000.0)


def test_scaled_to_logical_manual_override() -> None:
    assert scaled_to_logical(300.0, 150.0, 150) == (200.0, 100.0)
    assert scaled_to_logical(300.0, 150.0, 0) == (300.0, 150.0)
