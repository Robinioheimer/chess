from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from stockfish_vision.config import default_config, load_config
from stockfish_vision.gui.main_window import MainWindow
from stockfish_vision.gui.theme import build_stylesheet
from stockfish_vision.logging_setup import setup_logging
from stockfish_vision.version import full_version_label

logger = logging.getLogger(__name__)


def _exception_hook(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Unbehandelter Anwendungsfehler", exc_info=(exc_type, exc_value, exc_traceback))


def main() -> int:
    logfile = setup_logging()
    sys.excepthook = _exception_hook
    logger.info("%s startet", full_version_label())
    app = QApplication(sys.argv)
    app.setApplicationName("Stockfish Vision Chess AI")
    app.setOrganizationName("Robinio")
    try:
        config = load_config()
    except OSError as exc:
        logger.error("Konfiguration ist nicht schreibbar: %s", exc)
        config = default_config()
    app.setStyleSheet(build_stylesheet(config.ui.font_scale_percent))
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
