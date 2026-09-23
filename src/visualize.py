"""Draw detection boxes with a stable per-class color palette."""

from __future__ import annotations

import hashlib
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .detector import Detection

# Brand-neutral categorical palette (distinct on light and dark images).
_PALETTE = [
    (37, 99, 235),    # blue
    (220, 38, 38),    # red
    (5, 150, 105),    # green
    (217, 119, 6),    # amber
    (124, 58, 237),   # violet
    (8, 145, 178),    # cyan
    (219, 39, 119),   # pink
    (101, 163, 13),   # lime
    (67, 56, 202),    # indigo
    (180, 83, 9),     # brown
]


def class_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    return _PALETTE[int(digest, 16) % len(_PALETTE)]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: Sequence[Detection],
    *,
    thickness: int | None = None,
    font_size: int | None = None,
) -> Image.Image:
    """Return a copy of *image* with boxes and labels drawn on it."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    thickness = thickness or max(2, min(w, h) // 240)
    font = _font(font_size or max(14, min(w, h) // 42))

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = class_color(det.label)
        for t in range(thickness):
            draw.rectangle((x1 - t, y1 - t, x2 + t, y2 + t), outline=color)

        caption = f"{det.label} {det.score:.2f}"
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), caption, font=font)
        tw, th = tx1 - tx0, ty1 - ty0
        pad = 3
        bx1, by1 = x1, max(0, y1 - th - pad * 2)
        bx2, by2 = x1 + tw + pad * 2, by1 + th + pad * 2
        if by1 < 0:
            by1, by2 = y1, y1 + th + pad * 2
        draw.rectangle((bx1, by1, bx2, by2), fill=color)
        draw.text((bx1 + pad, by1 + pad), caption, fill=(255, 255, 255), font=font)

    return canvas
