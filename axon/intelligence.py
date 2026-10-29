"""Multi-view digit inference and interpretable uncertainty signals.

The underlying neural network remains pure NumPy. Multiple nearby translations
help expose predictions that change when the same glyph is moved a pixel.
These are heuristic diagnostics, NOT calibrated probabilities of correctness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictionReport:
    digit: int | None
    probabilities: np.ndarray
    candidates: tuple[tuple[int, float], ...]
    status: str  # blank, review, or predicted
    score: float
    margin: float
    agreement: float
    entropy: float
    views: int


def shift_batch(images: np.ndarray, dx: int, dy: int, *, size: int = 28) -> np.ndarray:
    """Shift flattened images with zero fill (never wrap pixels around edges)."""
    x = np.asarray(images)
    if x.ndim != 2 or x.shape[1] != size * size:
        raise ValueError(f"expected flattened {size}x{size} images")
    if not (-size < dx < size and -size < dy < size):
        raise ValueError("shift is larger than the image")
    source = x.reshape(-1, size, size)
    target = np.zeros_like(source)
    src_y = slice(max(0, -dy), size - max(0, dy))
    src_x = slice(max(0, -dx), size - max(0, dx))
    dst_y = slice(max(0, dy), size - max(0, -dy))
    dst_x = slice(max(0, dx), size - max(0, -dx))
    target[:, dst_y, dst_x] = source[:, src_y, src_x]
    return target.reshape(-1, size * size)


def predict_views(model, images: np.ndarray, *, smart: bool = True, batch_size: int = 256):
    """Return ensemble class scores and agreement across nearby translations.

    Center gets half the vote and each one-pixel translation gets one eighth.
    For 8x8 legacy models or smart=False, prediction is unchanged.
    """
    x = np.asarray(images, dtype=model.dtype)
    if x.ndim != 2 or x.shape[1] != model.sizes[0] or len(x) == 0:
        raise ValueError(f"expected nonempty batch of {model.sizes[0]} features")
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("features must be finite and in [0, 1]")
    views = [(0, 0)]
    if smart and model.sizes[0] == 784:
        views += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    probs = np.zeros((len(x), model.sizes[-1]), dtype=np.float64)
    votes = np.empty((len(views), len(x)), dtype=np.int64)
    weights = [0.5, 0.125, 0.125, 0.125, 0.125] if len(views) == 5 else [1.0]
    for v, ((dx, dy), weight) in enumerate(zip(views, weights)):
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size]
            shifted = shift_batch(chunk, dx, dy) if (dx or dy) else chunk
            prediction = model.predict_proba(shifted)
            probs[start:start + len(chunk)] += weight * prediction
            votes[v, start:start + len(chunk)] = prediction.argmax(axis=1)
    best = probs.argmax(axis=1)
    agreement = np.mean(votes == best[np.newaxis, :], axis=0)
    return probs, agreement, len(views)


def analyze_digit(model, image: np.ndarray, *, smart: bool = True,
                  min_score: float = 0.65, min_margin: float = 0.20,
                  min_agreement: float = 0.60) -> PredictionReport:
    """Classify one digit, or abstain when the image/prediction is unreliable."""
    x = np.asarray(image)
    if x.ndim != 2 or x.shape != (1, model.sizes[0]):
        raise ValueError(f"expected (1, {model.sizes[0]})")
    if not 0 <= min_score <= 1 or not 0 <= min_margin <= 1 or not 0 <= min_agreement <= 1:
        raise ValueError("thresholds must be in [0, 1]")
    # A tiny amount of noise should not be classified as an actual digit.
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("features must be finite and in [0, 1]")
    if np.count_nonzero(x > 0.18) < 3 or float(x.sum()) < 1.0:
        return PredictionReport(None, np.zeros(model.sizes[-1], dtype=np.float64),
                                (), "blank", 0.0, 0.0, 0.0, 0.0, 0)
    prob, agreement, views = predict_views(model, x, smart=smart)
    p = prob[0]
    order = np.argsort(-p, kind="stable")
    top = int(order[0]); score = float(p[top]); margin = score - float(p[order[1]])
    normalized_entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-15))) / np.log(len(p)))
    review = score < min_score or margin < min_margin or agreement[0] < min_agreement
    return PredictionReport(top, p.copy(),
                            tuple((int(i), float(p[i])) for i in order[:3]),
                            "review" if review else "predicted", score, margin,
                            float(agreement[0]), normalized_entropy, views)
