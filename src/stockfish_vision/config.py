from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from stockfish_vision.paths import user_config_path

logger = logging.getLogger(__name__)

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
    opacity: float = 0.75
    panel_width: int = 380
    panel_height: int = 230
    panel_offset_x: int = 16
    panel_offset_y: int = 16
    scale_percent: int = 100
    hotkey: str = DEFAULT_HOTKEY
    visible: bool = True
    show_frame: bool = True
    frame_color: str = "#ff2d2d"
    frame_width: int = 5
    coordinate_scale_percent: int = 0


@dataclass
class CaptureSettings:
    monitor_index: int = 1
    region: dict[str, int] | None = None
    corners: list[list[float]] | None = None
    orientation: str = "white_bottom"
    warp_size: int = 640
    change_threshold: float = 8.0
    auto_frame: bool = True


@dataclass
class UiSettings:
    window_width: int = 1380
    window_height: int = 900
    font_scale_percent: int = 100
    start_maximized: bool = False
    helper_visible: bool = True
    helper_x: int = -1
    helper_y: int = -1
    helper_topmost: bool = True


@dataclass
class AppConfig:
    engine: EngineSettings = field(default_factory=EngineSettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    last_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    side_to_move: str = "w"
    occupancy_std_threshold: float = 18.0
    min_piece_confidence: float = 0.45
    model_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        if not isinstance(data, dict):
            raise ValueError("Die Konfiguration muss ein JSON-Objekt enthalten.")
        cfg = cls()
        cfg.engine = EngineSettings(**_merge(EngineSettings, _section(data, "engine")))
        cfg.overlay = OverlaySettings(**_merge(OverlaySettings, _section(data, "overlay")))
        cfg.capture = CaptureSettings(**_merge(CaptureSettings, _section(data, "capture")))
        cfg.ui = UiSettings(**_merge(UiSettings, _section(data, "ui")))
        cfg.last_fen = str(data.get("last_fen", cfg.last_fen))
        cfg.side_to_move = str(data.get("side_to_move", cfg.side_to_move))
        cfg.occupancy_std_threshold = data.get(
            "occupancy_std_threshold", cfg.occupancy_std_threshold
        )
        cfg.min_piece_confidence = data.get(
            "min_piece_confidence", cfg.min_piece_confidence
        )
        cfg.model_path = str(data.get("model_path", cfg.model_path))
        cfg._normalise()
        return cfg

    def _normalise(self) -> None:
        self.engine.depth = _bounded_int(self.engine.depth, 0, 40, 18)
        self.engine.move_time_ms = _bounded_int(self.engine.move_time_ms, 0, 60_000, 800)
        self.engine.threads = _bounded_int(self.engine.threads, 1, 64, 4)
        self.engine.hash_mb = _bounded_int(self.engine.hash_mb, 16, 16_384, 128)
        self.engine.multipv = _bounded_int(self.engine.multipv, 1, 5, 3)
        self.engine.executable = str(self.engine.executable)

        self.overlay.opacity = _bounded_float(self.overlay.opacity, 0.2, 1.0, 0.75)
        self.overlay.panel_width = _bounded_int(self.overlay.panel_width, 240, 900, 380)
        self.overlay.panel_height = _bounded_int(self.overlay.panel_height, 140, 700, 230)
        self.overlay.panel_offset_x = _bounded_int(self.overlay.panel_offset_x, -20_000, 20_000, 16)
        self.overlay.panel_offset_y = _bounded_int(self.overlay.panel_offset_y, -20_000, 20_000, 16)
        self.overlay.scale_percent = _bounded_int(self.overlay.scale_percent, 60, 200, 100)
        self.overlay.hotkey = str(self.overlay.hotkey) or DEFAULT_HOTKEY
        self.overlay.visible = bool(self.overlay.visible)
        self.overlay.show_frame = bool(self.overlay.show_frame)
        self.overlay.frame_color = _normalise_color(self.overlay.frame_color, "#ff2d2d")
        self.overlay.frame_width = _bounded_int(self.overlay.frame_width, 1, 24, 5)
        self.overlay.coordinate_scale_percent = _bounded_int(
            self.overlay.coordinate_scale_percent, 0, 400, 0
        )
        if 0 < self.overlay.coordinate_scale_percent < 25:
            self.overlay.coordinate_scale_percent = 0

        self.capture.monitor_index = _bounded_int(self.capture.monitor_index, 0, 128, 1)
        self.capture.warp_size = _bounded_int(self.capture.warp_size, 128, 1600, 640)
        self.capture.change_threshold = _bounded_float(
            self.capture.change_threshold, 0.0, 255.0, 8.0
        )
        if self.capture.orientation not in {
            "white_bottom",
            "white_top",
            "white_left",
            "white_right",
        }:
            self.capture.orientation = "white_bottom"
        self.capture.region = _normalise_region(self.capture.region)
        self.capture.corners = _normalise_corners(self.capture.corners)
        self.capture.auto_frame = bool(self.capture.auto_frame)

        self.ui.window_width = _bounded_int(self.ui.window_width, 720, 6000, 1380)
        self.ui.window_height = _bounded_int(self.ui.window_height, 520, 4000, 900)
        self.ui.font_scale_percent = _bounded_int(self.ui.font_scale_percent, 70, 200, 100)
        self.ui.start_maximized = bool(self.ui.start_maximized)
        self.ui.helper_visible = bool(self.ui.helper_visible)
        self.ui.helper_x = _bounded_int(self.ui.helper_x, -20_000, 20_000, -1)
        self.ui.helper_y = _bounded_int(self.ui.helper_y, -20_000, 20_000, -1)
        self.ui.helper_topmost = bool(self.ui.helper_topmost)

        self.occupancy_std_threshold = _bounded_float(
            self.occupancy_std_threshold, 1.0, 100.0, 18.0
        )
        self.min_piece_confidence = _bounded_float(
            self.min_piece_confidence, 0.05, 0.99, 0.45
        )
        if self.side_to_move not in {"w", "b"}:
            self.side_to_move = "w"


def _section(data: dict[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Konfigurationsabschnitt '{key}' ist ungültig.")
    return value


def _merge(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    base = asdict(cls())
    base.update({key: value for key, value in data.items() if key in allowed})
    return base


def _bounded_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def _bounded_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
        if number != number:  # NaN
            raise ValueError
        return max(minimum, min(maximum, number))
    except (TypeError, ValueError, OverflowError):
        return default


def _normalise_region(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        region = {key: int(value[key]) for key in ("left", "top", "width", "height")}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if region["width"] < 16 or region["height"] < 16:
        return None
    return region


def _normalise_color(value: Any, default: str) -> str:
    if not isinstance(value, str):
        return default
    text = value.strip()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
        except ValueError:
            return default
        return text.lower()
    return default


def _normalise_corners(value: Any) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        corners = [[float(point[0]), float(point[1])] for point in value]
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if any(number != number or abs(number) == float("inf") for point in corners for number in point):
        return None
    return corners


def default_config() -> AppConfig:
    return AppConfig()


def load_config(path: Path | None = None) -> AppConfig:
    target = path or user_config_path()
    if not target.exists():
        cfg = default_config()
        save_config(cfg, target)
        return cfg
    try:
        with target.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return AppConfig.from_dict(data)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.error("Konfiguration konnte nicht geladen werden: %s", exc)
        cfg = default_config()
        try:
            backup = target.with_name(
                f"{target.stem}.corrupt-{datetime.now():%Y%m%d-%H%M%S}{target.suffix}"
            )
            target.replace(backup)
            logger.warning("Defekte Konfiguration gesichert unter %s", backup)
        except OSError:
            logger.exception("Defekte Konfiguration konnte nicht gesichert werden")
        save_config(cfg, target)
        return cfg


def save_config(config: AppConfig, path: Path | None = None) -> None:
    target = path or user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(config.to_dict())
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
