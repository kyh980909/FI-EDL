"""CIFAR-10-C (Hendrycks & Dietterich 2019) distribution-shift loader.

The CIFAR-10-C benchmark applies 19 corruption types × 5 severity levels
to the CIFAR-10 test set. Each `<corruption>.npy` file contains 50 000
images stored as uint8 (shape `[50000, 32, 32, 3]`); they are laid out as
five blocks of 10 000 images, ordered by severity 1→5. Labels are shared
across all corruptions in `labels.npy` (shape `[50000]`).

Used by both DAEDL (Yoon et al., ICML 2024, Table 4) and F-EDL (Yoon &
Kim, NeurIPS 2025, Table 2) as the canonical robustness / distribution-
shift detection benchmark for an evidential CIFAR-10 classifier.

Download (manual):
    curl -sSL -o data/CIFAR-10-C.tar \
         https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
    tar -xf data/CIFAR-10-C.tar -C data/

After extraction the file structure is::
    data/CIFAR-10-C/<corruption_name>.npy   (19 corruption files)
    data/CIFAR-10-C/labels.npy
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# The 19 standard corruption types (alphabetical order matches the
# Hendrycks zenodo archive).
CORRUPTION_TYPES = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
    "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
]


class CIFAR10CDataset(Dataset):
    """One (corruption, severity) slice of CIFAR-10-C.

    Args:
        root: path containing the unpacked ``CIFAR-10-C/`` directory.
        corruption: corruption name (must be in :data:`CORRUPTION_TYPES`).
        severity: severity level in 1..5.
        transform: PIL→tensor transform (typically the CIFAR-10 eval transform).
    """

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        transform: Optional[object] = None,
    ) -> None:
        if corruption not in CORRUPTION_TYPES:
            raise ValueError(
                f"Unknown corruption {corruption!r}. Valid: {CORRUPTION_TYPES}"
            )
        if severity not in (1, 2, 3, 4, 5):
            raise ValueError(f"severity must be in 1..5, got {severity}")

        base = Path(root) / "CIFAR-10-C"
        npy = base / f"{corruption}.npy"
        labels = base / "labels.npy"
        if not npy.exists() or not labels.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C data missing at {base}. Download via the curl/tar "
                "commands in this module's docstring."
            )
        images = np.load(npy, mmap_mode="r")  # mmap to avoid loading 1.9 GB
        all_labels = np.load(labels)

        # Each severity is a contiguous block of 10 000 images. Severity 1 -> [0,10k),
        # severity 5 -> [40k, 50k). Labels repeat across severities, so we slice both.
        start = (severity - 1) * 10000
        end = severity * 10000
        self.images = images[start:end]
        self.labels = all_labels[start:end]
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # mmap returns a memory-mapped array; convert to a regular array
        # before constructing the PIL image so it's safe across workers.
        arr = np.asarray(self.images[idx])
        img = Image.fromarray(arr)
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label
