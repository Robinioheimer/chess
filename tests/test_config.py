from __future__ import annotations

import json
from pathlib import Path

from stockfish_vision.config import AppConfig, load_config, save_config


def test_config_roundtrip_with_region_model_and_overlay(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig()
    config.engine.depth = 0
    config.engine.move_time_ms = 1250
    config.capture.region = {"left": -1920, "top": -20, "width": 900, "height": 700}
    config.capture.corners = [[0, 0], [800, 0], [800, 800], [0, 800]]
    config.overlay.scale_percent = 135
    config.model_path = "C:/models/pieces.xml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.engine.depth == 0
    assert loaded.engine.move_time_ms == 1250
    assert loaded.capture.region["left"] == -1920
    assert loaded.capture.corners is not None
    assert loaded.overlay.scale_percent == 135
    assert loaded.model_path.endswith("pieces.xml")
    assert not path.with_suffix(".json.tmp").exists()


def test_config_recovers_from_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{definitely broken", encoding="utf-8")
    loaded = load_config(path)
    assert loaded.engine.multipv == 3
    assert json.loads(path.read_text(encoding="utf-8"))["engine"]["depth"] == 18
    assert list(tmp_path.glob("config.corrupt-*.json"))


def test_config_normalises_out_of_range_and_unknown_values() -> None:
    config = AppConfig.from_dict(
        {
            "engine": {"depth": 900, "multipv": -2, "unknown": "ignored"},
            "overlay": {"opacity": 99, "scale_percent": 1},
            "capture": {"region": {"left": 0, "top": 0, "width": 2, "height": 2}},
            "min_piece_confidence": "not-a-number",
        }
    )
    assert config.engine.depth == 40
    assert config.engine.multipv == 1
    assert config.overlay.opacity == 1.0
    assert config.overlay.scale_percent == 60
    assert config.capture.region is None
    assert config.min_piece_confidence == 0.45


def test_config_stores_ui_size_font_and_frame_options(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig()
    config.ui.window_width = 1600
    config.ui.window_height = 1000
    config.ui.font_scale_percent = 125
    config.ui.start_maximized = True
    config.overlay.show_frame = False
    config.overlay.frame_color = "#00ff00"
    config.overlay.frame_width = 8
    config.overlay.coordinate_scale_percent = 150
    config.capture.auto_frame = False
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.ui.window_width == 1600
    assert loaded.ui.window_height == 1000
    assert loaded.ui.font_scale_percent == 125
    assert loaded.ui.start_maximized is True
    assert loaded.overlay.show_frame is False
    assert loaded.overlay.frame_color == "#00ff00"
    assert loaded.overlay.frame_width == 8
    assert loaded.overlay.coordinate_scale_percent == 150
    assert loaded.capture.auto_frame is False


def test_config_normalises_new_display_options() -> None:
    config = AppConfig.from_dict(
        {
            "ui": {"window_width": 10, "window_height": 99_999, "font_scale_percent": 500},
            "overlay": {"frame_color": "rot", "frame_width": 99, "coordinate_scale_percent": 5},
        }
    )
    assert config.ui.window_width == 720
    assert config.ui.window_height == 4000
    assert config.ui.font_scale_percent == 200
    assert config.overlay.frame_color == "#ff2d2d"
    assert config.overlay.frame_width == 24
    assert config.overlay.coordinate_scale_percent == 0


def test_version_labels_follow_scheme() -> None:
    from stockfish_vision.version import APP_VERSION, full_version_label, version_label, window_title

    assert APP_VERSION.startswith("v")
    assert APP_VERSION[1:].isdigit()
    assert version_label() == APP_VERSION
    assert version_label() in full_version_label()
    assert version_label() in window_title()


def test_config_stores_training_acknowledgement(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig()
    config.training_acknowledged = True
    config.training_acknowledged_at = "2026-10-23T12:00:00"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.training_acknowledged is True
    assert loaded.training_acknowledged_at == "2026-10-23T12:00:00"


def test_config_defaults_to_acknowledged_without_field(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"engine": {"depth": 12}}', encoding="utf-8")
    loaded = load_config(path)
    assert loaded.training_acknowledged is True
    assert loaded.training_acknowledged_at == ""
