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
