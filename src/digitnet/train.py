"""Train the network.

Usage:
    python -m digitnet.train --epochs 10 --lr 0.1 --hidden 128
"""

import argparse

import numpy as np

from .data import load_mnist, one_hot
from .model import NeuralNetwork, cross_entropy


def train(model, X_train, y_train, X_test, y_test, epochs=10, lr=0.1, batch_size=64, seed=0):
    rng = np.random.default_rng(seed)
    Y_train = one_hot(y_train, model.params["b2"].shape[0])
    history = []
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            model.forward(X_train[idx])
            model.step(model.backward(Y_train[idx]), lr)

        loss = cross_entropy(model.forward(X_train), Y_train)
        acc = model.accuracy(X_test, y_test)
        history.append({"epoch": epoch, "loss": loss, "test_acc": acc})
        print(f"epoch {epoch:2d}  loss {loss:.4f}  test accuracy {acc:.2%}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train an MNIST digit classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="models/digitnet_weights.npz")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    model = NeuralNetwork(n_hidden=args.hidden)
    train(model, X_train, y_train, X_test, y_test, args.epochs, args.lr, args.batch_size)
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
