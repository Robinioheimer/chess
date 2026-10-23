"""Bildschirmrechtecke in logischen **und** physischen Pixeln.

Qt liefert Fensterkoordinaten in logischen Pixeln, ``mss`` Aufnahmen in
physischen. Für eine verlässliche Umrechnung (auch bei gemischter
Windows-Skalierung und negativen Koordinaten) werden beide Rechtecke pro
Monitor gepaart. Fehlt die physische Angabe, wird auf den DPI-Faktor
zurückgefallen.
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QGuiApplication

from stockfish_vision.overlay.mapping import ScreenRect

__all__ = ["screen_rects"]

logger = logging.getLogger(__name__)


def _physical_monitors() -> list[tuple[float, float, float, float]] | None:
    try:
        from stockfish_vision.capture.screen import list_monitors
    except ImportError:
        return None
    try:
        monitors = list_monitors()
    except Exception as exc:  # pragma: no cover - hängt vom System ab
        logger.debug("Physische Monitorliste nicht verfügbar: %s", exc)
        return None
    # Index 0 ist der virtuelle Desktop (Union aller Monitore).
    return [
        (float(item.left), float(item.top), float(item.width), float(item.height))
        for item in monitors
        if item.index > 0
    ]


def screen_rects() -> list[ScreenRect]:
    """Sammelt alle Bildschirme mit logischem und physischem Rechteck."""

    screens = list(QGuiApplication.screens())
    physical = _physical_monitors()
    if physical is not None and len(physical) != len(screens):
        logger.debug(
            "Monitoranzahl Qt=%d, Aufnahme=%d – nutze DPI-Faktor.",
            len(screens),
            len(physical),
        )
        physical = None
    rects: list[ScreenRect] = []
    for index, screen in enumerate(screens):
        geometry = screen.geometry()
        factor = float(screen.devicePixelRatio()) or 1.0
        entry = (
            float(geometry.x()),
            float(geometry.y()),
            float(geometry.width()),
            float(geometry.height()),
        )
        if physical is not None:
            px, py, pw, ph = physical[index]
            ratio = geometry.width() / pw if pw else 0.0
            if 0.2 <= ratio <= 5.0 and ph > 0:
                rects.append((*entry, px, py, pw, ph))
                continue
        rects.append((*entry, factor))
    return rects
