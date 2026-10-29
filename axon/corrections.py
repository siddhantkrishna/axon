"""Private, on-device labeled handwriting samples for personal adaptation."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from .preprocess import image_to_features


def save_correction(image: Image.Image, label: int, directory: str | Path = "data/corrections") -> Path:
    if label not in range(10):
        raise ValueError("Label must be between 0 and 9")
    if np.max(image_to_features(image, size=28)) == 0:
        raise ValueError("Cannot save blank image")
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{label}-{uuid4().hex}.png"
    image.convert("L").save(path)
    return path


def load_corrections(*, size: int = 28, directory: str | Path = "data/corrections"):
    x, y = [], []
    for path in sorted(Path(directory).glob("*.png")):
        try:
            digit = int(path.name.split("-", 1)[0])
            if digit not in range(10):
                continue
            with Image.open(path) as image:
                feat = image_to_features(image, size=size)
                if feat.max() > 0:
                    x.append(feat[0])
                    y.append(digit)
        except (ValueError, OSError):
            continue
    return np.asarray(x, dtype=np.float64).reshape(-1, size*size), np.asarray(y, dtype=np.int64)
