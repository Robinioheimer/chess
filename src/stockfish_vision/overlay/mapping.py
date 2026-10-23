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


def logical_to_physical(
    x: float, y: float, screens: list[ScreenRect]
) -> tuple[float, float]:
    """Umkehrung von :func:`physical_to_logical` für einen einzelnen Punkt."""
    for logical_x, logical_y, logical_w, logical_h, dpr in screens:
        if logical_x <= x < logical_x + logical_w and logical_y <= y < logical_y + logical_h:
            factor = float(dpr) if dpr and float(dpr) > 0 else 1.0
            return float(x) * factor, float(y) * factor
    return float(x), float(y)


def logical_rect_to_region(
    rect: tuple[float, float, float, float],
    screens: list[ScreenRect],
    minimum: int = 16,
) -> dict[str, int] | None:
    """Wandelt ein mit der Maus aufgezogenes Rechteck in einen Aufnahmebereich um.

    ``rect`` ist ``(x, y, breite, hoehe)`` in Qt-Logikpixeln (globale Koordinaten).
    Rückgabe sind physische Bildschirmpixel für ``mss`` – passend zum Monitor,
    auf dem der Mittelpunkt des Rechtecks liegt. Zu kleine Flächen ergeben ``None``.
    """

    x, y, width, height = (float(value) for value in rect)
    if width <= 0 or height <= 0:
        return None
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    factor = 1.0
    for logical_x, logical_y, logical_w, logical_h, dpr in screens:
        if logical_x <= center_x < logical_x + logical_w and logical_y <= center_y < logical_y + logical_h:
            factor = float(dpr) if dpr and float(dpr) > 0 else 1.0
            break
    left = round(x * factor)
    top = round(y * factor)
    physical_width = round(width * factor)
    physical_height = round(height * factor)
    if physical_width < minimum or physical_height < minimum:
        return None
    return {
        "left": left,
        "top": top,
        "width": physical_width,
        "height": physical_height,
    }


def region_to_logical_rect(
    region: dict[str, int], screens: list[ScreenRect]
) -> tuple[float, float, float, float] | None:
    """Umkehrung: gespeicherter Aufnahmebereich in Qt-Logikpixel."""
    try:
        left = float(region["left"])
        top = float(region["top"])
        width = float(region["width"])
        height = float(region["height"])
    except (KeyError, TypeError, ValueError):
        return None
    center_x = left + width / 2.0
    center_y = top + height / 2.0
    for logical_x, logical_y, logical_w, logical_h, dpr in screens:
        factor = float(dpr) if dpr and float(dpr) > 0 else 1.0
        physical_x = logical_x * factor
        physical_y = logical_y * factor
        if (
            physical_x <= center_x < physical_x + logical_w * factor
            and physical_y <= center_y < physical_y + logical_h * factor
        ):
            return (left / factor, top / factor, width / factor, height / factor)
    return (left, top, width, height)
