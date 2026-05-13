"""Convert KMNIST .npz files into MNIST-style idx-ubyte files.

Expects `data/KMNIST/kmnist-{train,test}-{imgs,labels}.npz` and writes the
uncompressed `data/KMNIST/raw/{train,t10k}-{images-idx3,labels-idx1}-ubyte`,
which is exactly what torchvision.datasets.KMNIST checks for via
`_check_exists` (it strips the .gz suffix from its resource URLs).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _write_idx_images(path: Path, arr: np.ndarray) -> None:
    assert arr.ndim == 3 and arr.dtype == np.uint8
    n, h, w = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 0x00000803, n, h, w))
        f.write(arr.tobytes(order="C"))


def _write_idx_labels(path: Path, arr: np.ndarray) -> None:
    assert arr.ndim == 1
    arr = arr.astype(np.uint8, copy=False)
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 0x00000801, arr.shape[0]))
        f.write(arr.tobytes(order="C"))


def main() -> None:
    root = Path("data/KMNIST")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("kmnist-train-imgs.npz",   "train-images-idx3-ubyte", _write_idx_images),
        ("kmnist-train-labels.npz", "train-labels-idx1-ubyte", _write_idx_labels),
        ("kmnist-test-imgs.npz",    "t10k-images-idx3-ubyte",  _write_idx_images),
        ("kmnist-test-labels.npz",  "t10k-labels-idx1-ubyte",  _write_idx_labels),
    ]
    for src_name, dst_name, writer in pairs:
        src = root / src_name
        with np.load(src) as z:
            arr = z["arr_0"]
        writer(raw / dst_name, arr)
        print(f"{src_name} -> {dst_name}  shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
