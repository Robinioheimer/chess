"""GUI package; the main window is loaded lazily for headless worker tests."""

__all__ = ["MainWindow"]


def __getattr__(name: str):
    if name == "MainWindow":
        from stockfish_vision.gui.main_window import MainWindow

        return MainWindow
    raise AttributeError(name)
