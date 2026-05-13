"""MNIST-as-ID adapter with KMNIST / FashionMNIST as OOD."""
from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.data.adapters.base import DatasetAdapter, NormalizationSpec, make_deterministic_loader


class MNISTAdapter(DatasetAdapter):
    def __init__(
        self,
        root: str = "./data",
        val_from_train: bool = False,
        val_split: float = 0.0,
        seed: int = 0,
        normalize: bool = True,
        image_size: int = 32,
        grayscale_to_rgb: bool = True,
        random_crop_padding: int = 2,
    ) -> None:
        self.root = root
        self.val_from_train = bool(val_from_train)
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.image_size = int(image_size)
        self.grayscale_to_rgb = bool(grayscale_to_rgb)
        self.random_crop_padding = int(random_crop_padding)
        self._norm = NormalizationSpec(
            mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081)
        )

    def num_classes(self) -> int:
        return 10

    def normalization_spec(self) -> NormalizationSpec:
        return self._norm

    def _compose(self, train: bool) -> transforms.Compose:
        tf: list = []
        if self.image_size != 28:
            tf.append(transforms.Resize((self.image_size, self.image_size)))
        if self.grayscale_to_rgb:
            tf.append(transforms.Grayscale(num_output_channels=3))
        if train and self.random_crop_padding > 0:
            tf.append(transforms.RandomCrop(self.image_size, padding=self.random_crop_padding))
        tf.append(transforms.ToTensor())
        if self.normalize:
            mean = self._norm.mean if self.grayscale_to_rgb else (self._norm.mean[0],)
            std = self._norm.std if self.grayscale_to_rgb else (self._norm.std[0],)
            tf.append(transforms.Normalize(mean, std))
        return transforms.Compose(tf)

    def _split_indices(self, n_samples: int) -> tuple[list[int], list[int]]:
        val_size = int(round(n_samples * self.val_split))
        if val_size <= 0 or val_size >= n_samples:
            raise ValueError(f"Invalid data.val_split={self.val_split} for {n_samples} samples")
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n_samples, generator=generator).tolist()
        return indices[val_size:], indices[:val_size]

    def id_dataloaders(self, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
        train_tf = self._compose(train=True)
        eval_tf = self._compose(train=False)
        if self.val_from_train and self.val_split > 0.0:
            train_base = datasets.MNIST(self.root, train=True, download=True, transform=train_tf)
            val_base = datasets.MNIST(self.root, train=True, download=True, transform=eval_tf)
            train_idx, val_idx = self._split_indices(len(train_base))
            train_ds = Subset(train_base, train_idx)
            val_ds = Subset(val_base, val_idx)
        else:
            train_ds = datasets.MNIST(self.root, train=True, download=True, transform=train_tf)
            val_ds = datasets.MNIST(self.root, train=False, download=True, transform=eval_tf)
        test_ds = datasets.MNIST(self.root, train=False, download=True, transform=eval_tf)
        return {
            "train": make_deterministic_loader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, seed=self.seed),
            "val": make_deterministic_loader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, seed=self.seed),
            "test": make_deterministic_loader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, seed=self.seed),
        }

    def ood_dataloaders(
        self, names: Iterable[str], batch_size: int, num_workers: int
    ) -> Dict[str, DataLoader]:
        eval_tf = self._compose(train=False)
        out: Dict[str, DataLoader] = {}
        for name in names:
            key = str(name).lower()
            if key == "kmnist":
                ds = datasets.KMNIST(self.root, train=False, download=True, transform=eval_tf)
            elif key in {"fmnist", "fashionmnist"}:
                ds = datasets.FashionMNIST(self.root, train=False, download=True, transform=eval_tf)
            else:
                raise ValueError(f"Unsupported OOD dataset for MNIST ID: {name}")
            out[name] = make_deterministic_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, seed=self.seed)
        return out
