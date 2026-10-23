from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, Signal, QObject

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
WM_HOTKEY = 0x0312

user32 = ctypes.windll.user32


def parse_hotkey(spec: str) -> tuple[int, int]:
    modifiers = 0
    key = None
    for part in spec.lower().replace(" ", "").split("+"):
        if part in {"ctrl", "control"}:
            modifiers |= MOD_CONTROL
        elif part == "shift":
            modifiers |= MOD_SHIFT
        elif part == "alt":
            modifiers |= MOD_ALT
        elif part in {"win", "meta"}:
            modifiers |= MOD_WIN
        elif len(part) == 1:
            key = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit():
            key = 0x70 + int(part[1:]) - 1
        else:
            raise ValueError(f"Unbekannter Hotkey-Teil: {part}")
    if key is None:
        raise ValueError(f"Hotkey ohne Taste: {spec}")
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
        modifiers, vk = parse_hotkey(self.spec)
        if not user32.RegisterHotKey(None, self.hotkey_id, modifiers, vk):
            raise OSError(f"Hotkey konnte nicht registriert werden: {self.spec}")
        self._registered = True

    def unregister(self) -> None:
        if self._registered:
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
