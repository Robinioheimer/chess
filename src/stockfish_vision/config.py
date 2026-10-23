from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from stockfish_vision.paths import user_config_path


DEFAULT_HOTKEY = "ctrl+shift+f8"


@dataclass
class EngineSettings:
    executable: str = ""
    depth: int = 18
    move_time_ms: int = 800
    threads: int = 4
    hash_mb: int = 128
    multipv: int = 3


@dataclass
class OverlaySettings:
    opacity: float = 0.55
    panel_width: int = 320
    panel_height: int = 180
    panel_offset_x: int = 16
    panel_offset_y: int = 16
    hotkey: str = DEFAULT_HOTKEY
    visible: bool = True


@dataclass
class CaptureSettings:
    monitor_index: int = 1
    region: dict[str, int] | None = None
    corners: list[list[float]] | None = None
    orientation: str = "white_bottom"
    warp_size: int = 400
    change_threshold: float = 8.0


@dataclass
class AppConfig:
    engine: EngineSettings = field(default_factory=EngineSettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    last_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    side_to_move: str = "w"
    occupancy_std_threshold: float = 18.0
    min_piece_confidence: float = 0.45

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        cfg = cls()
        engine = data.get("engine", {})
        overlay = data.get("overlay", {})
        capture = data.get("capture", {})
        cfg.engine = EngineSettings(**_merge(EngineSettings, engine))
        cfg.overlay = OverlaySettings(**_merge(OverlaySettings, overlay))
        cfg.capture = CaptureSettings(**_merge(CaptureSettings, capture))
        cfg.last_fen = data.get("last_fen", cfg.last_fen)
        cfg.side_to_move = data.get("side_to_move", cfg.side_to_move)
        cfg.occupancy_std_threshold = float(
            data.get("occupancy_std_threshold", cfg.occupancy_std_threshold)
        )
        cfg.min_piece_confidence = float(
            data.get("min_piece_confidence", cfg.min_piece_confidence)
        )
        return cfg


def _merge(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    base = asdict(cls())
    base.update({key: value for key, value in data.items() if key in allowed})
    return base


def default_config() -> AppConfig:
    return AppConfig()


def load_config(path: Path | None = None) -> AppConfig:
    target = path or user_config_path()
    if not target.exists():
        cfg = default_config()
        save_config(cfg, target)
        return cfg
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return AppConfig.from_dict(data)


def save_config(config: AppConfig, path: Path | None = None) -> None:
    target = path or user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(config.to_dict())
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
