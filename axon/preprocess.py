"""Convert a freehand digit to the 8x8 or MNIST 28x28 image distribution."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def image_to_features(image: Image.Image, *, size: int = 8, auto_invert: bool = True) -> np.ndarray:
    """Return a batch of normalized pixels. 28x28 mimics MNIST's 20px glyph.

    This method helps match scale and centering; it cannot correct illegible
    handwriting or guarantee accuracy on out-of-distribution samples.
    """
    if size not in (8, 28):
        raise ValueError("size must be 8 or 28")
    gray = ImageOps.grayscale(image)
    pixels = np.asarray(gray)
    if auto_invert:
        corners = np.array([pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]])
        if np.median(corners) > 127:
            gray = ImageOps.invert(gray)
            pixels = np.asarray(gray)
    threshold = max(18, int(pixels.max()) // 8)
    mask = pixels > threshold
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros((1, size * size), dtype=np.float64)
    cropped = gray.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    target_glyph = 20 if size == 28 else 6
    w, h = cropped.size
    scale = min(target_glyph / w, target_glyph / h)
    new_w = max(1, min(size, round(w * scale)))
    new_h = max(1, min(size, round(h * scale)))
    glyph = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(glyph, ((size-new_w)//2, (size-new_h)//2))
    if size == 28:
        # MNIST preprocessing centers the mass, not just the bounding box.
        a = np.asarray(canvas, dtype=np.float64)
        mass = a.sum()
        if mass > 0:
            cy = float(np.sum(a * np.arange(size)[:, None]) / mass)
            cx = float(np.sum(a * np.arange(size)[None, :]) / mass)
            dx = int(round((size - 1) / 2 - cx))
            dy = int(round((size - 1) / 2 - cy))
            if dx or dy:
                shifted = Image.new("L", (size, size), 0)
                shifted.paste(canvas, (dx, dy))
                canvas = shifted
    return np.asarray(canvas, dtype=np.float64).reshape(1, size * size) / 255.0
