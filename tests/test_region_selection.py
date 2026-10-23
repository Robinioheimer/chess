from __future__ import annotations

from stockfish_vision.overlay.mapping import (
    logical_rect_to_region,
    logical_to_physical,
    region_to_logical_rect,
)


def test_logical_to_physical_uses_device_pixel_ratio() -> None:
    screens = [(0.0, 0.0, 1280.0, 720.0, 1.5)]
    assert logical_to_physical(100.0, 200.0, screens) == (150.0, 300.0)


def test_logical_rect_to_region_with_150_percent_scaling() -> None:
    screens = [(0.0, 0.0, 1280.0, 720.0, 1.5)]
    region = logical_rect_to_region((100.0, 100.0, 200.0, 150.0), screens)
    assert region == {"left": 150, "top": 150, "width": 300, "height": 225}


def test_logical_rect_to_region_uses_monitor_of_center() -> None:
    screens = [
        (0.0, 0.0, 1280.0, 720.0, 1.25),
        (-1280.0, 0.0, 1280.0, 720.0, 1.0),
    ]
    region = logical_rect_to_region((-1000.0, 100.0, 400.0, 400.0), screens)
    assert region == {"left": -1000, "top": 100, "width": 400, "height": 400}


def test_logical_rect_to_region_rejects_tiny_selection() -> None:
    screens = [(0.0, 0.0, 1280.0, 720.0, 1.0)]
    assert logical_rect_to_region((10.0, 10.0, 5.0, 40.0), screens) is None
    assert logical_rect_to_region((10.0, 10.0, 0.0, 0.0), screens) is None


def test_region_roundtrip_is_stable() -> None:
    screens = [(0.0, 0.0, 1920.0, 1080.0, 1.5)]
    region = {"left": 300, "top": 150, "width": 900, "height": 750}
    rect = region_to_logical_rect(region, screens)
    assert rect == (200.0, 100.0, 600.0, 500.0)
    assert logical_rect_to_region(rect, screens) == region


def test_region_to_logical_rect_rejects_broken_data() -> None:
    assert region_to_logical_rect({"left": 1}, []) is None
