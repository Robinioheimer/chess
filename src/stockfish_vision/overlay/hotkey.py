from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

if os.name == "nt":
    user32 = ctypes.windll.user32
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    user32.RegisterHotKey.restype = wintypes.BOOL
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UnregisterHotKey.restype = wintypes.BOOL
else:
    user32 = None


def parse_hotkey(spec: str) -> tuple[int, int]:
    modifiers = 0
    key: int | None = None
    parts = spec.lower().replace(" ", "").split("+")
    if not spec.strip() or any(not part for part in parts):
        raise ValueError(f"Ungültiger Hotkey: {spec!r}")
    for part in parts:
        if part in {"ctrl", "control"}:
            modifiers |= MOD_CONTROL
        elif part == "shift":
            modifiers |= MOD_SHIFT
        elif part == "alt":
            modifiers |= MOD_ALT
        elif part in {"win", "meta"}:
            modifiers |= MOD_WIN
        elif len(part) == 1 and part.isalnum():
            if key is not None:
                raise ValueError("Ein Hotkey darf nur eine Taste enthalten.")
            key = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
            if key is not None:
                raise ValueError("Ein Hotkey darf nur eine Taste enthalten.")
            key = 0x70 + int(part[1:]) - 1
        else:
            raise ValueError(f"Unbekannter Hotkey-Teil: {part}")
    if key is None:
        raise ValueError(f"Hotkey ohne Taste: {spec}")
    if modifiers == 0:
        raise ValueError("Bitte mindestens einen Modifikator (Ctrl, Alt, Shift oder Win) angeben.")
    return modifiers, key


class WindowsHotkeyFilter(QAbstractNativeEventFilter, QObject):
    activated = Signal()

    def __init__(self, spec: str, hotkey_id: int = 1) -> None:
        QAbstractNativeEventFilter.__init__(self)
        QObject.__init__(self)
        self.spec = spec
        self.hotkey_id = hotkey_id
        self._registered = False

    def register(self) -> None:
        if os.name != "nt" or user32 is None:
            raise OSError("Globale Hotkeys werden nur unter Windows unterstützt.")
        modifiers, vk = parse_hotkey(self.spec)
        if not user32.RegisterHotKey(None, self.hotkey_id, modifiers | MOD_NOREPEAT, vk):
            raise OSError(f"Hotkey konnte nicht registriert werden: {self.spec}")
        self._registered = True

    def unregister(self) -> None:
        if self._registered and user32 is not None:
            user32.UnregisterHotKey(None, self.hotkey_id)
            self._registered = False

    def nativeEventFilter(self, eventType: bytes | str, message: int) -> tuple[bool, int]:
        if eventType not in (b"windows_generic_MSG", "windows_generic_MSG"):
            return False, 0
        msg = wintypes.MSG.from_address(int(message))
        if msg.message == WM_HOTKEY and msg.wParam == self.hotkey_id:
            self.activated.emit()
            return True, 0
        return False, 0
