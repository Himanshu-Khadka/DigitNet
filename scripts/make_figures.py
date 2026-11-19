"""Train DigitNet and save the figures used in the README to assets/.

Usage:
    pip install -e ".[viz]"
    python scripts/make_figures.py
"""

import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from digitnet.data import load_mnist
from digitnet.model import NeuralNetwork
from digitnet.train import train

OUT = Path("assets")
BG, PANEL, GRID = "#131218", "#1b1a21", "#2a2830"
TEXT, MUTED = "#d9d5cc", "#938f87"
GOOD, BAD = "#9ec800", "#e8705a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "monospace", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"saved {OUT / name}")


def training_curve(history):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(epochs, [h["loss"] for h in history], color=MUTED, lw=2, marker="o", ms=4)
    ax1.set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")
    acc = [100 * h["test_acc"] for h in history]
    ax2.plot(epochs, acc, color=GOOD, lw=2, marker="o", ms=4)
    ax2.set(title="Test accuracy", xlabel="epoch", ylabel="%")
    ax2.annotate(f"{acc[-1]:.2f}%", (epochs[-1], acc[-1]), xytext=(-46, -18),
                 textcoords="offset points", color=GOOD)
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.6)
        ax.set_xticks(epochs)
        ax.title.set_color(TEXT)
    save(fig, "training_curve.png")


def digit_grid(images, true, pred, probs, title, name, cols=8):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.55))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, t, p, pr in zip(np.ravel(axes), images, true, pred, probs):
        ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ok = t == p
        ax.set_title(f"pred {p}  {pr:.0%}" + ("" if ok else f"\ntrue {t}"),
                     color=GOOD if ok else BAD, fontsize=8.5)
    two_line = any(t != p for t, p in zip(true, pred))
    fig.suptitle(title, color=TEXT, y=1.06 if two_line else 1.0)
    save(fig, name)


def learned_weights(model):
    W1 = model.params["W1"]
    fig, axes = plt.subplots(4, 8, figsize=(9, 4.8))
    for i, ax in enumerate(np.ravel(axes)):
        w = W1[:, i].reshape(28, 28)
        lim = np.abs(w).max()
        ax.imshow(w, cmap="BrBG", vmin=-lim, vmax=lim)
        ax.axis("off")
    fig.suptitle("What 32 of the 128 hidden neurons learned to look for", color=TEXT)
    save(fig, "hidden_weights.png")


def terminal(log: str):
    lines = ["$ digitnet-train --epochs 10", *log.strip().splitlines()]
    fig = plt.figure(figsize=(7.2, 0.27 * len(lines) + 0.6))
    fig.patch.set_facecolor(PANEL)
    for i, line in enumerate(lines):
        color = GOOD if line.startswith("$") or "Saved" in line else TEXT
        fig.text(0.04, 1 - (i + 1.2) / (len(lines) + 1.4), line, color=color,
                 family="monospace", fontsize=10.5, va="center")
    save(fig, "terminal.png")


def main():
    OUT.mkdir(exist_ok=True)
    X_train, y_train, X_test, y_test = load_mnist("data")
    model = NeuralNetwork()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        history = train(model, X_train, y_train, X_test, y_test, epochs=10, lr=0.1, batch_size=64)
    log = buf.getvalue() + "Saved model to models/digitnet_weights.npz\n"
    print(log)

    probs = model.forward(X_test)
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)

    rng = np.random.default_rng(7)
    idx = rng.choice(len(X_test), 24, replace=False)
    digit_grid(X_test[idx], y_test[idx], pred[idx], conf[idx],
               "Random test images (never seen during training)", "predictions.png")

    wrong = np.flatnonzero(pred != y_test)
    idx = rng.choice(wrong, min(16, len(wrong)), replace=False)
    digit_grid(X_test[idx], y_test[idx], pred[idx], conf[idx],
               f"Some of the {len(wrong)} mistakes out of {len(y_test):,}", "mistakes.png")

    training_curve(history)
    learned_weights(model)
    terminal(log)


if __name__ == "__main__":
    main()
