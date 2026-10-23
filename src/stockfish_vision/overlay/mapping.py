"""Umrechnung zwischen Aufnahme- und Qt-Koordinaten (ohne Qt-Abhängigkeit)."""

from __future__ import annotations

ScreenRect = tuple[float, float, float, float, float]


def physical_to_logical(
    x: float,
    y: float,
    screens: list[ScreenRect],
) -> tuple[float, float]:
    """Rechnet Aufnahmekoordinaten (physische Pixel) in Qt-Logikpixel um.

    ``screens`` enthält je Bildschirm ``(x, y, breite, hoehe, dpi_faktor)`` in
    logischen Koordinaten. Windows liefert Bildschirmaufnahmen in physischen
    Pixeln, Qt-Fenster dagegen in logischen Pixeln. Ohne diese Umrechnung
    landen Pfeile bei 125 %, 150 % oder 200 % Windows-Skalierung außerhalb des
    sichtbaren Bereichs – das Overlay scheint dann nicht zu funktionieren.
    """

    for logical_x, logical_y, logical_w, logical_h, dpr in screens:
        factor = float(dpr) if dpr and float(dpr) > 0 else 1.0
        physical_x = float(logical_x) * factor
        physical_y = float(logical_y) * factor
        physical_w = float(logical_w) * factor
        physical_h = float(logical_h) * factor
        if (
            physical_x <= x < physical_x + physical_w
            and physical_y <= y < physical_y + physical_h
        ):
            return (
                float(logical_x) + (x - physical_x) / factor,
                float(logical_y) + (y - physical_y) / factor,
            )
    return float(x), float(y)


def scaled_to_logical(
    x: float, y: float, percent: int | float
) -> tuple[float, float]:
    """Manuelle Umrechnung mit festem Prozentfaktor (0 = keine Korrektur)."""
    factor = float(percent) / 100.0 if percent else 1.0
    if factor <= 0:
        factor = 1.0
    return x / factor, y / factor
