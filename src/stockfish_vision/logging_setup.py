from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from stockfish_vision.paths import logs_dir


def setup_logging(log_dir: Path | None = None) -> Path:
    directory = log_dir or logs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    logfile = directory / f"vision-{datetime.now():%Y%m%d}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    logging.getLogger("stockfish_vision").info("Logging nach %s", logfile)
    return logfile
