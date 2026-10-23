"""Hält Overlay-Fenster zuverlässig im Vordergrund.

Manche Anwendungen (Vollbild-Browser, Spiele, Videoplayer) drängen
``WindowStaysOnTopHint``-Fenster nach hinten. Ein kurzer, periodischer
``SetWindowPos(HWND_TOPMOST)`` bringt sie zurück, ohne den Fokus zu stehlen.
Unter Linux/macOS sind alle Funktionen bewusst No-Ops.
"""

from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes

from PySide6.QtCore import QObject, QTimer

logger = logging.getLogger(__name__)

HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
DEFAULT_INTERVAL_MS = 2000

if os.name == "nt":
    user32 = ctypes.windll.user32
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
else:
    user32 = None


def ensure_topmost(widget) -> bool:
    """Setzt ein Qt-Fenster sofort nach vorn, ohne es zu aktivieren."""
    if os.name != "nt" or user32 is None:
        return False
    try:
        hwnd = int(widget.winId())
    except (RuntimeError, TypeError, ValueError):
        return False
    if not hwnd:
        return False
    try:
        return bool(
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        )
    except OSError:  # pragma: no cover - nur unter Windows möglich
        logger.exception("Fenster konnte nicht nach vorn geholt werden")
        return False


class TopmostKeeper(QObject):
    """Prüft in festem Intervall, ob die überwachten Fenster noch obenauf liegen."""

    def __init__(self, interval_ms: int = DEFAULT_INTERVAL_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(max(250, int(interval_ms)))
        self._timer.timeout.connect(self.reassert)
        self._widgets: list = []
        self.checks = 0

    def watch(self, widget) -> None:
        if widget is None:
            return
        if widget not in self._widgets:
            self._widgets.append(widget)

    def unwatch(self, widget) -> None:
        if widget in self._widgets:
            self._widgets.remove(widget)

    def watched_count(self) -> int:
        return len(self._widgets)

    def start(self) -> None:
        if self._widgets:
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def is_active(self) -> bool:
        return self._timer.isActive()

    def reassert(self) -> int:
        """Holt alle sichtbaren, überwachten Fenster nach vorn. Liefert die Anzahl."""
        self.checks += 1
        restored = 0
        for widget in list(self._widgets):
            try:
                if not widget.isVisible():
                    continue
            except RuntimeError:  # Fenster bereits zerstört
                self.unwatch(widget)
                continue
            if ensure_topmost(widget):
                restored += 1
        return restored
