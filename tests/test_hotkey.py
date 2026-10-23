from __future__ import annotations

from stockfish_vision.overlay.hotkey import parse_hotkey


def test_parse_ctrl_shift_f8() -> None:
    modifiers, vk = parse_hotkey("ctrl+shift+f8")
    assert modifiers & 0x0002  # control
    assert modifiers & 0x0004  # shift
    assert vk == 0x77  # VK_F8
