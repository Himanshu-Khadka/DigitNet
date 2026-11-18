"""Load a trained model and show predictions on random test images.

Usage:
    python -m digitnet.predict --n 5
"""

import argparse

import numpy as np

from .data import load_mnist
from .model import NeuralNetwork


def render(img: np.ndarray) -> str:
    """Draw a 28x28 image as ASCII art."""
    chars = " .:-=+*#%@"
    rows = img.reshape(28, 28)[::2]
    return "\n".join("".join(chars[int(p * 9)] for p in row) for row in rows)


def main():
    parser = argparse.ArgumentParser(description="Predict digits with a trained model")
    parser.add_argument("--model", default="models/digitnet_weights.npz")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    model = NeuralNetwork.load(args.model)
    _, _, X_test, y_test = load_mnist(args.data_dir)
    print(f"Test accuracy: {model.accuracy(X_test, y_test):.2%}\n")

    for k in np.random.default_rng().choice(len(X_test), args.n, replace=False):
        probs = model.forward(X_test[k:k + 1])[0]
        print(render(X_test[k]))
        print(f"-> predicted {probs.argmax()} ({probs.max():.0%} sure), actual {y_test[k]}\n")


if __name__ == "__main__":
    main()
