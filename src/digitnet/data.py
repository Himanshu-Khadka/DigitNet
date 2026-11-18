"""Download, load and preprocess the MNIST dataset."""

import gzip
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def download(data_dir: Path) -> None:
    """Download any MNIST files that aren't already in data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(BASE_URL + name, path)


def _read_idx(path: Path, is_images: bool) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if is_images:
        # 16-byte header, then 28*28 bytes per image
        return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784)
    # 8-byte header, then 1 byte per label
    return np.frombuffer(raw, np.uint8, offset=8)


def load_mnist(data_dir: str | Path = "data"):
    """Return (X_train, y_train, X_test, y_test), pixels scaled to 0..1."""
    data_dir = Path(data_dir)
    download(data_dir)
    d = {k: _read_idx(data_dir / f, k.startswith("x")) for k, f in FILES.items()}
    X_train = d["x_train"].astype(np.float32) / 255.0
    X_test = d["x_test"].astype(np.float32) / 255.0
    return X_train, d["y_train"], X_test, d["y_test"]


def one_hot(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Label 3 -> [0,0,0,1,0,0,0,0,0,0]."""
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1
    return out
