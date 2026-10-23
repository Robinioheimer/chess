"""Umrechnung zwischen Aufnahme- und Qt-Koordinaten (ohne Qt-Abhängigkeit).

Windows liefert Bildschirmaufnahmen in **physischen** Pixeln, Qt rechnet
dagegen in **logischen** Pixeln. Bei 125/150/200 % Skalierung – und erst recht
mit mehreren Monitoren unterschiedlicher Skalierung – ist der physische
Ursprung eines Monitors nicht einfach ``logischer_Ursprung * Faktor``.

Deshalb akzeptiert dieses Modul pro Bildschirm entweder

* ``(x, y, breite, hoehe, dpi_faktor)`` – logische Werte, Faktor geschätzt, oder
* ``(lx, ly, lw, lh, px, py, pw, ph)`` – logische **und** physische Werte.

Mit beiden Rechtecken wird pro Monitor ein exakter linearer Bezug hergestellt:
``logisch = lx + (physisch - px) * lw / pw``. Das ist der einzige Weg, der auch
bei gemischter Skalierung und negativen Koordinaten stimmt.
"""

from __future__ import annotations

ScreenRect = tuple[float, ...]


def _unpack(rect: ScreenRect) -> tuple[float, float, float, float, float, float, float, float]:
    """Liefert (lx, ly, lw, lh, px, py, pw, ph) für beide Tupelformen."""
    values = tuple(float(value) for value in rect)
    if len(values) >= 8:
        lx, ly, lw, lh, px, py, pw, ph = values[:8]
        return lx, ly, lw, lh, px, py, pw, ph
    lx, ly, lw, lh = values[:4]
    factor = values[4] if len(values) > 4 and values[4] > 0 else 1.0
    return lx, ly, lw, lh, lx * factor, ly * factor, lw * factor, lh * factor


def _scales(rect: ScreenRect) -> tuple[float, float, float, float, float, float, float, float]:
    lx, ly, lw, lh, px, py, pw, ph = _unpack(rect)
    scale_x = lw / pw if pw > 0 else 1.0
    scale_y = lh / ph if ph > 0 else 1.0
    return lx, ly, lw, lh, px, py, scale_x, scale_y


def physical_to_logical(
    x: float,
    y: float,
    screens: list[ScreenRect],
) -> tuple[float, float]:
    """Rechnet Aufnahmekoordinaten (physische Pixel) in Qt-Logikpixel um."""

    for rect in screens:
        lx, ly, lw, lh, px, py, scale_x, scale_y = _scales(rect)
        if px <= x < px + lw / scale_x and py <= y < py + lh / scale_y:
            return (lx + (x - px) * scale_x, ly + (y - py) * scale_y)
    return float(x), float(y)


def logical_to_physical(
    x: float, y: float, screens: list[ScreenRect]
) -> tuple[float, float]:
    """Umkehrung von :func:`physical_to_logical` für einen einzelnen Punkt."""

    for rect in screens:
        lx, ly, lw, lh, px, py, scale_x, scale_y = _scales(rect)
        if lx <= x < lx + lw and ly <= y < ly + lh:
            return (px + (x - lx) / scale_x, py + (y - ly) / scale_y)
    return float(x), float(y)


def scaled_to_logical(
    x: float, y: float, percent: int | float
) -> tuple[float, float]:
    """Manuelle Umrechnung mit festem Prozentfaktor (0 = keine Korrektur)."""

    factor = float(percent) / 100.0 if percent else 1.0
    if factor <= 0:
        factor = 1.0
    return x / factor, y / factor


def logical_rect_to_region(
    rect: tuple[float, float, float, float],
    screens: list[ScreenRect],
    minimum: int = 16,
) -> dict[str, int] | None:
    """Wandelt ein mit der Maus aufgezogenes Rechteck in einen Aufnahmebereich um.

    ``rect`` ist ``(x, y, breite, hoehe)`` in Qt-Logikpixeln (globale Koordinaten).
    Rückgabe sind physische Bildschirmpixel für ``mss``.
    """

    x, y, width, height = (float(value) for value in rect)
    if width <= 0 or height <= 0:
        return None
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    target = None
    for index, screen in enumerate(screens):
        lx, ly, lw, lh = screen[:4]
        lx, ly, lw, lh = float(lx), float(ly), float(lw), float(lh)
        if lx <= center_x < lx + lw and ly <= center_y < ly + lh:
            target = index
            break
    if target is None:
        target = 0 if screens else None
    if target is None:
        left, top = round(x), round(y)
        physical_width, physical_height = round(width), round(height)
    else:
        left_px, top_px = logical_to_physical(x, y, screens)
        right_px, bottom_px = logical_to_physical(x + width, y + height, screens)
        left = round(left_px)
        top = round(top_px)
        physical_width = round(right_px - left_px)
        physical_height = round(bottom_px - top_px)
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
    x0, y0 = physical_to_logical(left, top, screens)
    x1, y1 = physical_to_logical(left + width, top + height, screens)
    return (x0, y0, x1 - x0, y1 - y0)
