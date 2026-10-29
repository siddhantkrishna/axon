"""A fully-connected neural network implemented directly with NumPy.

No PyTorch, TensorFlow, JAX, autograd, or scikit-learn estimators are used.
Shapes: examples are rows, activations are (batch, features).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class History:
    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    epochs_completed: int

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "epochs_completed": self.epochs_completed,
        }


class NeuralNetwork:
    """Dense, multi-class neural network with backprop and Adam/SGD.

    The final layer emits unnormalized logits; the cross-entropy loss applies
    softmax directly, using a numerically stable log-sum-exp calculation.
    """

    def __init__(self, sizes: list[int] | tuple[int, ...], *,
                 activation: str = "relu", l2: float = 0.0001,
                 seed: int = 42, dtype: str = "float64"):
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        self.dtype = np.dtype(dtype)
        if len(sizes) < 2 or any(not isinstance(n, int) or n <= 0 for n in sizes):
            raise ValueError("sizes must contain at least two positive integers")
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")
        if l2 < 0:
            raise ValueError("l2 cannot be negative")
        self.sizes = list(sizes)
        self.activation = activation
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt((2.0 if activation == "relu" else 1.0) / fan_in)
            self.weights.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(self.dtype))
            self.biases.append(np.zeros((1, fan_out), dtype=self.dtype))
        self.metadata: dict = {}

    def _activate(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z) if self.activation == "relu" else np.tanh(z)

    def _activation_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(np.float64)
        a = np.tanh(z)
        return 1.0 - a * a

    def forward(self, x: np.ndarray, *, cache: bool = False):
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 2 or x.shape[1] != self.sizes[0]:
            raise ValueError(f"expected input (batch, {self.sizes[0]}), got {x.shape}")
        activations = [x]
        preactivations = []
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            preactivations.append(z)
            a = z if i == len(self.weights) - 1 else self._activate(z)
            activations.append(a)
        if cache:
            return a, (activations, preactivations)
        return a

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray):
        """Return (cross-entropy + L2 loss, list of (dW, db)).

        dL/dlogits = (softmax(logits) - one_hot(y)) / batch_size.
        Then backpropagate using chain rule, layer by layer.
        """
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        logits, (activations, zs) = self.forward(x, cache=True)
        batch_size, classes = logits.shape
        if y.shape != (batch_size,) or np.any((y < 0) | (y >= classes)):
            raise ValueError("labels must be class integers matching the input batch")

        shifted = logits - logits.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=1))
        data_loss = np.mean(logsumexp - shifted[np.arange(batch_size), y])
        reg_loss = 0.5 * self.l2 * sum(np.sum(w * w) for w in self.weights)
        loss = float(data_loss + reg_loss)

        delta = self.softmax(logits)
        delta[np.arange(batch_size), y] -= 1.0
        delta /= batch_size
        grads = [None] * len(self.weights)
        for layer in reversed(range(len(self.weights))):
            d_w = activations[layer].T @ delta + self.l2 * self.weights[layer]
            d_b = delta.sum(axis=0, keepdims=True)
            grads[layer] = (d_w, d_b)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * self._activation_grad(zs[layer - 1])
        return loss, grads

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        logits = self.forward(x)
        if len(y) != len(logits) or np.any((y < 0) | (y >= self.sizes[-1])):
            raise ValueError("invalid labels")
        shifted = logits - logits.max(axis=1, keepdims=True)
        data_loss = (np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]).mean()
        return float(data_loss + 0.5 * self.l2 * sum((w * w).sum() for w in self.weights))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.softmax(self.forward(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y).reshape(-1)))

    def fit(self, x: np.ndarray, y: np.ndarray, *,
            x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            epochs: int = 100, batch_size: int = 64, lr: float = 0.003,
            optimizer: str = "adam", patience: int = 20, seed: int = 42,
            on_epoch: Callable[[int, int, dict], None] | None = None) -> History:
        """Train mini-batches; restore best validation weights if provided.

        Early stopping requires validation data and stops after `patience`
        successive epochs without a decrease in validation loss.
        """
        x = np.asarray(x, dtype=self.dtype)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.sizes[0] or len(x) == 0 or len(x) != len(y):
            raise ValueError("invalid training data shapes")
        if np.any((y < 0) | (y >= self.sizes[-1])):
            raise ValueError("training labels out of range")
        if (x_val is None) != (y_val is None):
            raise ValueError("x_val and y_val must both be provided")
        if x_val is not None:
            x_val = np.asarray(x_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.int64).reshape(-1)
            if x_val.ndim != 2 or x_val.shape[1] != self.sizes[0] or len(x_val) != len(y_val) or not len(x_val):
                raise ValueError("invalid validation data shapes")
            if np.any((y_val < 0) | (y_val >= self.sizes[-1])):
                raise ValueError("validation labels out of range")
        if epochs < 1 or batch_size < 1 or lr <= 0 or patience < 1:
            raise ValueError("epochs, batch_size, lr, and patience must be positive")
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'")

        rng = np.random.default_rng(seed)
        params = [v for pair in zip(self.weights, self.biases) for v in pair]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        best_loss = float("inf")
        best_params = None
        since_improvement = 0
        hist = History([], [], [], [], 0)
        for epoch in range(1, epochs + 1):
            indices = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                batch_ids = indices[start:start + batch_size]
                _, grads = self.loss_and_gradients(x[batch_ids], y[batch_ids])
                grad_list = [g for pair in grads for g in pair]
                t += 1
                for j, (p, g) in enumerate(zip(params, grad_list)):
                    if optimizer == "adam":
                        m[j] = 0.9 * m[j] + 0.1 * g
                        v[j] = 0.999 * v[j] + 0.001 * (g * g)
                        p -= lr * (m[j] / (1.0 - 0.9**t)) / (np.sqrt(v[j] / (1.0 - 0.999**t)) + 1e-8)
                    else:
                        p -= lr * g
            hist.train_loss.append(self.loss(x, y))
            hist.train_accuracy.append(self.accuracy(x, y))
            if x_val is not None:
                val_loss = self.loss(x_val, y_val)
                val_acc = self.accuracy(x_val, y_val)
                hist.val_loss.append(val_loss)
                hist.val_accuracy.append(val_acc)
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_params = [p.copy() for p in params]
                    since_improvement = 0
                else:
                    since_improvement += 1
            hist.epochs_completed = epoch
            if on_epoch is not None:
                on_epoch(epoch, epochs, {
                    "train_loss": hist.train_loss[-1],
                    "train_accuracy": hist.train_accuracy[-1],
                    "val_loss": hist.val_loss[-1] if x_val is not None else None,
                    "val_accuracy": hist.val_accuracy[-1] if x_val is not None else None,
                })
            if x_val is not None and since_improvement >= patience:
                break
        if best_params is not None:
            for p, best_p in zip(params, best_params):
                p[...] = best_p
        return hist

    def save(self, path: str | Path, *, metadata: dict | None = None) -> None:
        """Portable .npz archive with JSON metadata; no pickled objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format": "axon-v1", "sizes": self.sizes,
                "activation": self.activation, "l2": self.l2, "dtype": str(self.dtype),
                "extra": metadata if metadata is not None else self.metadata}
        arrays = {"metadata": np.array(json.dumps(meta))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["metadata"].item())
            if meta.get("format") != "axon-v1":
                raise ValueError("not an AXON v1 model")
            net = cls(meta["sizes"], activation=meta["activation"], l2=meta["l2"], dtype=meta.get("dtype", "float64"))
            for i in range(len(net.weights)):
                w = archive[f"W{i}"]
                b = archive[f"b{i}"]
                if w.shape != net.weights[i].shape or b.shape != net.biases[i].shape:
                    raise ValueError("corrupt model: incompatible layer shapes")
                net.weights[i][...] = w
                net.biases[i][...] = b
            net.metadata = meta.get("extra", {})
        return net
