from __future__ import annotations

import pytest

from stockfish_vision.overlay.mapping import (
    logical_rect_to_region,
    logical_to_physical,
    physical_to_logical,
    region_to_logical_rect,
)

# Zwei Monitore, gemischte Skalierung:
# Primär 1920x1080 physisch, 125 % -> 1536x864 logisch, Ursprung 0/0.
# Sekundär 1920x1080 physisch, 100 % -> 1920x1080 logisch, rechts daneben.
SCREENS = [
    (0.0, 0.0, 1536.0, 864.0, 0.0, 0.0, 1920.0, 1080.0),
    (1536.0, 0.0, 1920.0, 1080.0, 1920.0, 0.0, 1920.0, 1080.0),
]


def test_primary_screen_125_percent() -> None:
    assert physical_to_logical(0, 0, SCREENS) == (0.0, 0.0)
    # Ein Pixel vor dem Rand: dort endet der primäre Monitor.
    assert physical_to_logical(1919, 1079, SCREENS) == pytest.approx((1535.2, 863.2))


def test_second_screen_is_not_offset_by_primary_scaling() -> None:
    # Genau der Fehler des alten Verfahrens: Ursprung * Faktor.
    assert physical_to_logical(1920, 100, SCREENS) == pytest.approx((1536.0, 100.0))
    assert physical_to_logical(3839, 1079, SCREENS) == pytest.approx((3455.0, 1079.0))


def test_roundtrip_physical_logical() -> None:
    for point in ((0, 0), (960, 540), (1920, 10), (3000, 900)):
        logical = physical_to_logical(*point, SCREENS)
        back = logical_to_physical(*logical, SCREENS)
        assert back == pytest.approx(point, abs=0.5)


def test_fallback_without_physical_rect_keeps_dpi_factor() -> None:
    single = [(0.0, 0.0, 1536.0, 864.0, 1.25)]
    assert physical_to_logical(1536, 864, single) == pytest.approx((1228.8, 691.2))


def test_selection_rectangle_is_converted_per_screen() -> None:
    region = logical_rect_to_region((1600.0, 100.0, 400.0, 300.0), SCREENS)
    assert region is not None
    assert region["left"] == 1984
    assert region["top"] == 100
    assert region["width"] == 400
    assert region["height"] == 300


def test_region_roundtrip() -> None:
    region = {"left": 100, "top": 200, "width": 640, "height": 640}
    rect = region_to_logical_rect(region, SCREENS)
    assert rect is not None
    assert rect == pytest.approx((80.0, 160.0, 512.0, 512.0))
    assert logical_rect_to_region(rect, SCREENS) == region
