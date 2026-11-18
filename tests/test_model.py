"""Unit tests — run with: pytest"""

import numpy as np

from digitnet.data import one_hot
from digitnet.model import NeuralNetwork, cross_entropy, softmax
from digitnet.train import train


def test_softmax_rows_sum_to_one():
    z = np.random.default_rng(0).normal(size=(5, 10)) * 50
    assert np.allclose(softmax(z).sum(axis=1), 1.0)


def test_one_hot():
    assert one_hot(np.array([3]))[0].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


def test_forward_shapes():
    net = NeuralNetwork(n_input=20, n_hidden=8, n_output=4)
    probs = net.forward(np.zeros((7, 20), dtype=np.float32))
    assert probs.shape == (7, 4)


def test_gradients_match_numerical():
    """Gradient check: backprop must agree with finite differences."""
    rng = np.random.default_rng(1)
    net = NeuralNetwork(n_input=6, n_hidden=5, n_output=3, seed=1)
    net.params = {k: v.astype(np.float64) for k, v in net.params.items()}
    X = rng.normal(size=(4, 6))
    Y = one_hot(np.array([0, 2, 1, 2]), 3).astype(np.float64)

    net.forward(X)
    grads = net.backward(Y)

    eps = 1e-6
    for name, param in net.params.items():
        for idx in np.ndindex(param.shape):
            orig = param[idx]
            param[idx] = orig + eps
            plus = cross_entropy(net.forward(X), Y)
            param[idx] = orig - eps
            minus = cross_entropy(net.forward(X), Y)
            param[idx] = orig
            numeric = (plus - minus) / (2 * eps)
            assert abs(numeric - grads[name][idx]) < 1e-5, (name, idx)


def test_training_reduces_loss_on_toy_data():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 10)) * 3
    y = rng.integers(0, 3, 300)
    X = (centers[y] + rng.normal(size=(300, 10))).astype(np.float32)
    net = NeuralNetwork(n_input=10, n_hidden=16, n_output=3)
    history = train(net, X, y, X, y, epochs=5, lr=0.1, batch_size=32)
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["test_acc"] > 0.9


def test_save_and_load(tmp_path):
    net = NeuralNetwork(n_input=6, n_hidden=5, n_output=3)
    net.save(tmp_path / "w.npz")
    loaded = NeuralNetwork.load(tmp_path / "w.npz")
    X = np.ones((2, 6), dtype=np.float32)
    assert np.allclose(net.forward(X), loaded.forward(X))
