# DigitNet

A neural network that recognizes handwritten digits (trained on the MNIST dataset), written in pure NumPy, with no deep-learning framework. Forward pass, backpropagation and gradient descent are all implemented by hand.

**Architecture:** 784 inputs → 128 hidden (ReLU) → 10 outputs (softmax), trained with mini-batch SGD on cross-entropy loss.
**Result:** ~97–98% test accuracy after 10 epochs on a laptop CPU.

## Project structure

```
digitnet/
├── src/digitnet/
│   ├── __init__.py     # package entry, exposes NeuralNetwork
│   ├── data.py         # download + load MNIST, one-hot encoding
│   ├── model.py        # the network: forward, backward, step, save/load
│   ├── train.py        # training loop + CLI
│   └── predict.py      # load a trained model, show predictions
├── tests/
│   └── test_model.py   # unit tests incl. numerical gradient check
├── data/               # MNIST downloads here (git-ignored)
├── models/             # trained weights saved here (git-ignored)
├── pyproject.toml      # package metadata + dependencies
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/digitnet.git
cd digitnet
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```bash
# Train (downloads MNIST on first run, ~11 MB)
digitnet-train --epochs 10 --lr 0.1 --hidden 128

# See predictions from the trained model
digitnet-predict --n 5

# Run tests
pytest
```

You can also run them as modules: `python -m digitnet.train`.

## How it works

1. **Forward:** `h = ReLU(x·W1 + b1)`, `p = softmax(h·W2 + b2)`
2. **Loss:** cross-entropy, `−log p[true digit]`
3. **Backward:** chain rule from the output back. The combined softmax + cross-entropy gradient is simply `p − y`
4. **Update:** `W ← W − lr · ∇W`

`tests/test_model.py` checks the backprop gradients against numerical finite differences.

## Ideas to extend

- Add a second hidden layer, momentum, or the Adam optimizer
- Plot loss/accuracy curves with matplotlib
- Reimplement in PyTorch and compare
- Build a convolutional network (CNN) for 99%+
