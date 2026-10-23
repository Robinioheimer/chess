from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def assets_dir() -> Path:
    bundled = package_dir() / "assets"
    if bundled.exists():
        return bundled
    return project_root() / "assets"


def models_dir() -> Path:
    # Bundled assets can be read-only in an installed package or PyInstaller build.
    return assets_dir() / "models"


def app_data_dir() -> Path:
    roaming = os.environ.get("APPDATA")
    base = Path(roaming) if roaming else Path.home() / "AppData" / "Roaming"
    path = base / "StockfishVisionChessAI"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_config_path() -> Path:
    return app_data_dir() / "config.json"
