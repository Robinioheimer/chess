"""Zentrale Versionsangabe.

Neue Fassungen werden fortlaufend als ``v1``, ``v2``, ``v3`` … benannt;
nicht veröffentlichte Nummern werden übersprungen.
Die Nummer erscheint im Fenstertitel, im Start-Tab und unter *Hilfe → Version*.
"""

from __future__ import annotations

APP_NAME = "Stockfish Vision Chess AI"
APP_VERSION = "v6"
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
    (
        "v4",
        "Einrichtung über ein eigenes Windows-Popup (v3 wurde übersprungen): "
        "Vollbild-Auswahl des Bildschirmausschnitts mit der Maus, kleines Helfer-Fenster "
        "für die drei Schritte, dauerhaft im Vordergrund inklusive Wächter, "
        "vollständig per Maus statt Pixelwerten.",
    ),
    (
        "v5",
        "Größenkorrektur: Die Ausschnitt-Auswahl deckt jetzt garantiert den gesamten "
        "Desktop ab (auch mehrere Monitore und hohe Windows-Skalierung), und das "
        "Helfer-Popup ist breiter, an der Kante frei skalierbar und merkt sich seine Größe.",
    ),
    (
        "v6",
        "Helles Design (weißer Hintergrund, schwarzer Text, graue Akzente) in Fenster, "
        "Popup, Auswahl-Ebene und Overlay-Panel; Auswahl-Ebene jetzt auf jedem Monitor "
        "einzeln über den ganzen Bildschirm; Trainings-Zustimmung wird automatisch erteilt.",
    ),
)


def version_label() -> str:
    """Kurzes Versionskürzel, z. B. ``v4``."""
    return f"{APP_VERSION}{APP_VERSION_SUFFIX}"


def full_version_label() -> str:
    """Anzeigename inklusive Programmname, z. B. ``Stockfish Vision Chess AI v4``."""
    return f"{APP_NAME} {version_label()}"


def window_title(section: str = "Trainingsmodus") -> str:
    return f"{full_version_label()} – {section}"
