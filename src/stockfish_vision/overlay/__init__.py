"""Transparent overlay components (QtGui is imported lazily)."""

__all__ = ["OverlayWindow"]


def __getattr__(name: str):
    if name == "OverlayWindow":
        from stockfish_vision.overlay.window import OverlayWindow

        return OverlayWindow
    raise AttributeError(name)
