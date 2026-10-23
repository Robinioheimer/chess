from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import mss
import numpy as np


@dataclass
class MonitorInfo:
    index: int
    left: int
    top: int
    width: int
    height: int
    label: str


def list_monitors() -> list[MonitorInfo]:
    with mss.mss() as sct:
        infos: list[MonitorInfo] = []
        for index, mon in enumerate(sct.monitors):
            label = (
                "Virtueller Desktop (alle Monitore)"
                if index == 0
                else f"Monitor {index} ({mon['width']}x{mon['height']})"
            )
            infos.append(
                MonitorInfo(
                    index=index,
                    left=int(mon["left"]),
                    top=int(mon["top"]),
                    width=int(mon["width"]),
                    height=int(mon["height"]),
                    label=label,
                )
            )
        return infos


def validate_region(region: Mapping[str, int]) -> dict[str, int]:
    try:
        validated = {key: int(region[key]) for key in ("left", "top", "width", "height")}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Bereich benötigt left, top, width und height als ganze Zahlen.") from exc
    if validated["width"] < 16 or validated["height"] < 16:
        raise ValueError("Der Aufnahmebereich muss mindestens 16×16 Pixel groß sein.")
    return validated


class ScreenCapture:
    def grab(
        self,
        monitor_index: int = 1,
        region: dict[str, int] | None = None,
    ) -> np.ndarray:
        with mss.mss() as sct:
            if region is not None:
                grab = validate_region(region)
            else:
                monitors = sct.monitors
                if monitor_index < 0 or monitor_index >= len(monitors):
                    raise ValueError(f"Ungültiger Monitor-Index: {monitor_index}")
                grab = monitors[monitor_index]
            raw = np.array(sct.grab(grab))
        if raw.ndim != 3 or raw.shape[2] not in {3, 4} or raw.size == 0:
            raise RuntimeError("Die Bildschirmaufnahme lieferte kein gültiges Bild.")
        return raw[:, :, :3].copy() if raw.shape[2] == 4 else raw.copy()
