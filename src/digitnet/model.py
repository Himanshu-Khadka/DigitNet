"""The neural network: forward pass, backpropagation, and parameter updates."""

from pathlib import Path

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, Y: np.ndarray) -> float:
    return float(-np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1)))


class NeuralNetwork:
    """input -> hidden (ReLU) -> output (softmax)."""

    def __init__(self, n_input=784, n_hidden=128, n_output=10, seed=42):
        rng = np.random.default_rng(seed)
        # He initialization keeps activations from exploding or vanishing
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / n_input), (n_input, n_hidden)).astype(np.float32),
            "b1": np.zeros(n_hidden, dtype=np.float32),
            "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_output)).astype(np.float32),
            "b2": np.zeros(n_output, dtype=np.float32),
        }
        self._cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = relu(z1)
        probs = softmax(a1 @ p["W2"] + p["b2"])
        self._cache = {"X": X, "z1": z1, "a1": a1, "probs": probs}
        return probs

    def backward(self, Y: np.ndarray) -> dict:
        """Gradients of cross-entropy loss w.r.t. every parameter (call forward first)."""
        c, p = self._cache, self.params
        n = len(Y)
        dz2 = (c["probs"] - Y) / n  # softmax + cross-entropy combined
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * (c["z1"] > 0)   # ReLU gate
        return {
            "W2": c["a1"].T @ dz2,
            "b2": dz2.sum(axis=0),
            "W1": c["X"].T @ dz1,
            "b1": dz1.sum(axis=0),
        }

    def step(self, grads: dict, lr: float) -> None:
        for k in self.params:
            self.params[k] -= lr * grads[k]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        w = np.load(path)
        net = cls(w["W1"].shape[0], w["W1"].shape[1], w["W2"].shape[1])
        net.params = {k: w[k] for k in w.files}
        return net
