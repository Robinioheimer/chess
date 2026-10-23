"""Zentrale Versionsangabe.

Neue Fassungen werden fortlaufend als ``v1``, ``v2``, ``v3`` … benannt.
Die Nummer erscheint im Fenstertitel, im Start-Tab und unter *Hilfe → Version*.
"""

from __future__ import annotations

APP_NAME = "Stockfish Vision Chess AI"
APP_VERSION = "v2"
APP_VERSION_SUFFIX = ""  # z. B. "-beta", leer für eine reguläre Fassung

VERSION_HISTORY: tuple[tuple[str, str], ...] = (
    (
        "v1",
        "Erste Fassung: Bildschirmaufnahme, Vier-Punkte-Kalibrierung, HOG-SVM-Erkennung, "
        "Stockfish über UCI, PGN-Nachbereitung und transparentes Overlay.",
    ),
    (
        "v2",
        "Übersichtlicher Start-Tab mit geführtem Ablauf, einstellbare Programmgröße und "
        "Schriftgröße, automatischer roter Brettrahmen (auch per Rechteck aufziehen), "
        "robustes Overlay mit DPI-Korrektur, Testanzeige und Status-Panel.",
    ),
)


def version_label() -> str:
    """Kurzes Versionskürzel, z. B. ``v2``."""
    return f"{APP_VERSION}{APP_VERSION_SUFFIX}"


def full_version_label() -> str:
    """Anzeigename inklusive Programmname, z. B. ``Stockfish Vision Chess AI v2``."""
    return f"{APP_NAME} {version_label()}"


def window_title(section: str = "Trainingsmodus") -> str:
    return f"{full_version_label()} – {section}"
