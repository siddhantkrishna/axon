"""Official MNIST IDX data, with fixed holdout and no model/framework dependency.

Four gzip files are downloaded once. Images are flattened to rows of 784 pixels
in [0,1]. Official 10,000-image test partition is NEVER used for training.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import struct
from urllib.request import urlretrieve

import numpy as np

ROOT = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

@dataclass(frozen=True)
class MNISTData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _get_file(cache: Path, filename: str) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / filename
    if not target.exists():
        print(f"Downloading official MNIST file: {filename}", flush=True)
        part = target.with_suffix(target.suffix + ".part")
        try:
            urlretrieve(ROOT + filename, part)
            part.replace(target)
        except Exception as exc:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download MNIST. Check your internet connection and try again. "
                f"Dataset URL: {ROOT}{filename}"
            ) from exc
    return target


def _read_idx(path: Path) -> np.ndarray:
    try:
        with gzip.open(path, "rb") as stream:
            magic = struct.unpack(">I", stream.read(4))[0]
            if magic == 2051:
                n, rows, cols = struct.unpack(">III", stream.read(12))
                if rows != 28 or cols != 28:
                    raise ValueError("MNIST images must be 28x28")
                payload = stream.read()
                if len(payload) != n * 784:
                    raise ValueError("MNIST image file is truncated")
                return np.frombuffer(payload, dtype=np.uint8).reshape(n, 784).copy()
            if magic == 2049:
                n = struct.unpack(">I", stream.read(4))[0]
                payload = stream.read()
                if len(payload) != n:
                    raise ValueError("MNIST label file is truncated")
                return np.frombuffer(payload, dtype=np.uint8).copy()
            raise ValueError(f"Invalid MNIST magic number: {magic}")
    except (OSError, EOFError, struct.error) as exc:
        raise ValueError(f"Unreadable MNIST file: {path}. Delete it and retry.") from exc


def load_mnist(*, cache_dir: str | Path = "data/mnist", seed: int = 42, val_size: int = 5000) -> MNISTData:
    cache = Path(cache_dir)
    items = {key: _read_idx(_get_file(cache, name)) for key, name in FILES.items()}
    x = items["train_images"]
    y = items["train_labels"].astype(np.int64)
    x_test = items["test_images"]
    y_test = items["test_labels"].astype(np.int64)
    if x.shape != (60000, 784) or x_test.shape != (10000, 784) or y.shape != (60000,) or y_test.shape != (10000,):
        raise ValueError("MNIST dataset shapes are incorrect; remove corrupted files")
    if not (1 <= val_size < 20000):
        raise ValueError("val_size must be between 1 and 19999")
    ids = np.random.default_rng(seed).permutation(len(y))
    val = ids[:val_size]
    train = ids[val_size:]
    return MNISTData(x[train].astype(np.float32) / 255.0, y[train],
                     x[val].astype(np.float32) / 255.0, y[val],
                     x_test.astype(np.float32) / 255.0, y_test)


def augment_shifts(x: np.ndarray, y: np.ndarray, *, copies: int = 1, seed: int = 42):
    """Translate MNIST glyphs by at most two pixels, without wrapping edges."""
    if copies < 0 or copies > 3:
        raise ValueError("copies must be 0..3")
    if copies == 0:
        return x, y
    rng = np.random.default_rng(seed)
    base = x.reshape(-1, 28, 28)
    shifted_sets = [x]
    for _ in range(copies):
        result = np.zeros_like(base)
        # Group by shifts so array slicing stays vectorized.
        sx = rng.integers(-2, 3, size=len(base))
        sy = rng.integers(-2, 3, size=len(base))
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                ids = np.flatnonzero((sx == dx) & (sy == dy))
                if not len(ids):
                    continue
                src_y = slice(max(0, -dy), 28 - max(0, dy))
                src_x = slice(max(0, -dx), 28 - max(0, dx))
                dst_y = slice(max(0, dy), 28 - max(0, -dy))
                dst_x = slice(max(0, dx), 28 - max(0, -dx))
                result[ids, dst_y, dst_x] = base[ids, src_y, src_x]
        shifted_sets.append(result.reshape(-1, 784))
    return np.concatenate(shifted_sets, axis=0), np.tile(y, copies + 1)
