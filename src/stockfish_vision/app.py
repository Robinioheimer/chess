from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from stockfish_vision.config import default_config, load_config
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
QPushButton:disabled { background: #343a40; color: #89929b; }
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTableWidget, QTreeWidget {
  background: #1b2026; border: 1px solid #343d48; border-radius: 4px; padding: 4px;
}
QHeaderView::section { background: #20262d; padding: 5px; border: none; }
QLabel { color: #d5dde5; }
QStatusBar { background: #0e1114; color: #9fb0c0; }
QGroupBox { border: 1px solid #2a313a; margin-top: 12px; }
QCheckBox { spacing: 8px; }
QSlider::groove:horizontal { height: 6px; background: #2a313a; }
QSlider::handle:horizontal { background: #3b9a63; width: 14px; margin: -5px 0; border-radius: 7px; }
"""

logger = logging.getLogger(__name__)


def _exception_hook(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Unbehandelter Anwendungsfehler", exc_info=(exc_type, exc_value, exc_traceback))


def main() -> int:
    logfile = setup_logging()
    sys.excepthook = _exception_hook
    logger.info("Stockfish Vision Chess AI startet")
    app = QApplication(sys.argv)
    app.setApplicationName("Stockfish Vision Chess AI")
    app.setOrganizationName("Robinio")
    app.setStyleSheet(STYLE)
    try:
        config = load_config()
    except OSError as exc:
        logger.error("Konfiguration ist nicht schreibbar: %s", exc)
        config = default_config()
    try:
        window = MainWindow(config)
        window.show()
    except Exception as exc:
        logger.critical("Anwendungsstart fehlgeschlagen", exc_info=True)
        QMessageBox.critical(
            None,
            "Stockfish Vision – Startfehler",
            f"Die Anwendung konnte nicht gestartet werden:\n{exc}\n\nProtokoll: {logfile}",
        )
        return 1
    return int(app.exec())
