from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random


def _draw_pawn(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.ellipse((x0 + w * 0.32, y0 + h * 0.12, x0 + w * 0.68, y0 + h * 0.48), fill=color)
    draw.polygon(
        [
            (x0 + w * 0.38, y0 + h * 0.42),
            (x0 + w * 0.62, y0 + h * 0.42),
            (x0 + w * 0.78, y0 + h * 0.88),
            (x0 + w * 0.22, y0 + h * 0.88),
        ],
        fill=color,
    )


def _draw_rook(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cren = [
        (x0 + w * 0.18, y0 + h * 0.18),
        (x0 + w * 0.32, y0 + h * 0.18),
        (x0 + w * 0.32, y0 + h * 0.30),
        (x0 + w * 0.44, y0 + h * 0.30),
        (x0 + w * 0.44, y0 + h * 0.18),
        (x0 + w * 0.56, y0 + h * 0.18),
        (x0 + w * 0.56, y0 + h * 0.30),
        (x0 + w * 0.68, y0 + h * 0.30),
        (x0 + w * 0.68, y0 + h * 0.18),
        (x0 + w * 0.82, y0 + h * 0.18),
        (x0 + w * 0.82, y0 + h * 0.88),
        (x0 + w * 0.18, y0 + h * 0.88),
    ]
    draw.polygon(cren, fill=color)


def _draw_knight(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    pts = [
        (x0 + w * 0.72, y0 + h * 0.82),
        (x0 + w * 0.22, y0 + h * 0.82),
        (x0 + w * 0.28, y0 + h * 0.52),
        (x0 + w * 0.18, y0 + h * 0.40),
        (x0 + w * 0.38, y0 + h * 0.18),
        (x0 + w * 0.62, y0 + h * 0.16),
        (x0 + w * 0.78, y0 + h * 0.34),
        (x0 + w * 0.60, y0 + h * 0.42),
        (x0 + w * 0.70, y0 + h * 0.58),
    ]
    draw.polygon(pts, fill=color)


def _draw_bishop(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.polygon(
        [
            (x0 + w * 0.50, y0 + h * 0.10),
            (x0 + w * 0.78, y0 + h * 0.70),
            (x0 + w * 0.22, y0 + h * 0.70),
        ],
        fill=color,
    )
    draw.rectangle((x0 + w * 0.32, y0 + h * 0.70, x0 + w * 0.68, y0 + h * 0.88), fill=color)
    slit = (min(color[0] + 80, 255), min(color[1] + 80, 255), min(color[2] + 80, 255))
    draw.line((x0 + w * 0.50, y0 + h * 0.22, x0 + w * 0.50, y0 + h * 0.58), fill=slit, width=2)


def _draw_queen(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    pts = [
        (x0 + w * 0.18, y0 + h * 0.82),
        (x0 + w * 0.20, y0 + h * 0.38),
        (x0 + w * 0.35, y0 + h * 0.55),
        (x0 + w * 0.50, y0 + h * 0.16),
        (x0 + w * 0.65, y0 + h * 0.55),
        (x0 + w * 0.80, y0 + h * 0.38),
        (x0 + w * 0.82, y0 + h * 0.82),
    ]
    draw.polygon(pts, fill=color)
    draw.ellipse((x0 + w * 0.42, y0 + h * 0.08, x0 + w * 0.58, y0 + h * 0.24), fill=color)


def _draw_king(draw: ImageDraw.ImageDraw, color: tuple[int, int, int], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw.rectangle((x0 + w * 0.28, y0 + h * 0.42, x0 + w * 0.72, y0 + h * 0.88), fill=color)
    draw.rectangle((x0 + w * 0.44, y0 + h * 0.10, x0 + w * 0.56, y0 + h * 0.42), fill=color)
    draw.rectangle((x0 + w * 0.32, y0 + h * 0.20, x0 + w * 0.68, y0 + h * 0.30), fill=color)


DRAWERS = {
    "P": _draw_pawn,
    "R": _draw_rook,
    "N": _draw_knight,
    "B": _draw_bishop,
    "Q": _draw_queen,
    "K": _draw_king,
}


def render_piece_glyph(
    symbol: str,
    size: int = 32,
    light_square: bool = True,
    jitter: bool = True,
    rng: random.Random | None = None,
) -> np.ndarray:
    rng = rng or random.Random()
    if light_square:
        bg = (rng.randint(210, 240), rng.randint(190, 220), rng.randint(150, 180))
    else:
        bg = (rng.randint(90, 130), rng.randint(60, 95), rng.randint(35, 70))
    image = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(image)
    is_white = symbol.isupper()
    color = (245, 245, 245) if is_white else (25, 25, 25)
    drawer = DRAWERS[symbol.upper()]
    pad = rng.randint(1, 4) if jitter else 3
    drawer(draw, color, (pad, pad, size - pad, size - pad))
    if jitter:
        if rng.random() < 0.4:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
        if rng.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.2))
        angle = rng.uniform(-8, 8)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=bg)
        arr = np.array(image).astype(np.int16)
        noise = rng.randint(0, 12)
        arr += np.random.randint(-noise, noise + 1, arr.shape, dtype=np.int16)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    return np.array(image, dtype=np.uint8)
