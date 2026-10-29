"""Datasets: scikit-learn supplies *pixels only*, never the neural network."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DigitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def load_digits_data(*, seed: int = 42, test_ratio: float = 0.15,
                     val_ratio: float = 0.15) -> DigitData:
    try:
        from sklearn.datasets import load_digits
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to load the built-in digits dataset") from exc
    if not (0 < test_ratio < 1 and 0 < val_ratio < 1 and test_ratio + val_ratio < 0.8):
        raise ValueError("invalid validation/test split fractions")
    digits = load_digits()
    x = np.asarray(digits.data, dtype=np.float64) / 16.0
    y = np.asarray(digits.target, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []
    for category in np.unique(y):
        indices = np.flatnonzero(y == category)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_ratio)))
        n_val = max(1, int(round(len(indices) * val_ratio)))
        test_ids.extend(indices[:n_test])
        val_ids.extend(indices[n_test:n_test + n_val])
        train_ids.extend(indices[n_test + n_val:])
    train_ids = rng.permutation(train_ids)
    val_ids = rng.permutation(val_ids)
    test_ids = rng.permutation(test_ids)
    return DigitData(x[train_ids], y[train_ids], x[val_ids], y[val_ids],
                     x[test_ids], y[test_ids], train_ids, val_ids, test_ids)


def make_xor(*, repeats: int = 100, jitter: float = 0.04, seed: int = 42):
    """Four nonlinearly separable patterns, repeated with small Gaussian noise."""
    if repeats < 1 or jitter < 0:
        raise ValueError("repeats must be >= 1; jitter must be nonnegative")
    rng = np.random.default_rng(seed)
    base = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    x = np.tile(base, (repeats, 1))
    y = np.tile(np.array([0, 1, 1, 0]), repeats)
    x += rng.normal(0.0, jitter, size=x.shape)
    ids = rng.permutation(len(x))
    return x[ids], y[ids]
