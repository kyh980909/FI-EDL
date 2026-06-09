"""TinyImageNet val-set wrapper for OOD evaluation.

Used by the CIFAR-100 ID classifier (F-EDL paper, Yoon NeurIPS 2025 setup):
the TinyImageNet val/images directory holds 10 000 unlabeled 64x64 RGB/L
JPEGs that serve as a far-OOD probe for CIFAR-100. We resize to 32x32 and
convert grayscale → RGB so the CIFAR pretrained backbone can consume them
without modification.

Download (manual, ~250 MB):
    curl -sSL -o data/tiny-imagenet/tiny-imagenet-200.zip \
         http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip -q data/tiny-imagenet/tiny-imagenet-200.zip \
          -d data/tiny-imagenet/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetOOD(Dataset):
    """OOD-only wrapper over the TinyImageNet val/images directory."""

    def __init__(
        self,
        root: str = "./data/tiny-imagenet",
        transform: Optional[object] = None,
    ) -> None:
        base = Path(root) / "tiny-imagenet-200" / "val" / "images"
        if not base.exists():
            raise FileNotFoundError(
                f"TinyImageNet val images not found at {base}. "
                "Download via the curl + unzip commands in this module's docstring."
            )
        self.files = sorted(str(p) for p in base.iterdir() if p.suffix.upper() == ".JPEG")
        if not self.files:
            raise RuntimeError(f"No JPEG files under {base}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0  # OOD label is unused; return 0 as placeholder
