from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from stockfish_vision.config import load_config
from stockfish_vision.gui.main_window import MainWindow
from stockfish_vision.logging_setup import setup_logging

STYLE = """
QWidget { background: #121417; color: #e8edf2; font-family: "Segoe UI"; font-size: 13px; }
QMainWindow { background: #121417; }
QTabWidget::pane { border: 1px solid #2a313a; }
QTabBar::tab { background: #1b2026; padding: 8px 16px; margin-right: 2px; }
QTabBar::tab:selected { background: #2e7d4f; }
QPushButton { background: #2e7d4f; border: none; padding: 8px 12px; border-radius: 6px; }
QPushButton:hover { background: #3b9a63; }
QPushButton:pressed { background: #246540; }
QLineEdit, QTextEdit, QSpinBox, QComboBox, QListWidget {
  background: #1b2026; border: 1px solid #343d48; border-radius: 4px; padding: 4px;
}
QLabel { color: #d5dde5; }
QStatusBar { background: #0e1114; color: #9fb0c0; }
QGroupBox { border: 1px solid #2a313a; margin-top: 12px; }
QCheckBox { spacing: 8px; }
QSlider::groove:horizontal { height: 6px; background: #2a313a; }
QSlider::handle:horizontal { background: #3b9a63; width: 14px; margin: -5px 0; border-radius: 7px; }
"""


def main() -> int:
    setup_logging()
    logging.getLogger(__name__).info("Stockfish Vision Chess AI startet")
    app = QApplication(sys.argv)
    app.setApplicationName("Stockfish Vision Chess AI")
    app.setStyleSheet(STYLE)
    config = load_config()
    window = MainWindow(config)
    window.show()
    code = app.exec()
    return int(code)
