r"""CIFAR-100-as-ID adapter with SVHN / CIFAR-10 / DTD / Places365 as OOD.

CIFAR-100 is the canonical "harder" ID benchmark in modern evidential-EDL
literature (used by DAEDL, F-EDL, R-EDL, Re-EDL). Same 32x32 image format
as CIFAR-10 but with 100 classes instead of 10, so K=100 in
\eqref{eq:fimtrace} and the trigamma-based controller saturates at a
different magnitude --- relevant to §4.1's gate diagnosis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.data.adapters.base import DatasetAdapter, NormalizationSpec, make_deterministic_loader


class CIFAR100Adapter(DatasetAdapter):
    def __init__(
        self,
        root: str = "./data",
        val_from_train: bool = False,
        val_split: float = 0.0,
        seed: int = 0,
        normalize: bool = True,
        random_rotation_degrees: float = 0.0,
        val_use_train_transform: bool = False,
    ) -> None:
        self.root = root
        self.val_from_train = bool(val_from_train)
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.random_rotation_degrees = float(random_rotation_degrees)
        self.val_use_train_transform = bool(val_use_train_transform)
        # CIFAR-100 has slightly different per-channel statistics than CIFAR-10;
        # values match the standard cifar-100 mean/std used in the literature.
        self._norm = NormalizationSpec(
            mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)
        )

    def num_classes(self) -> int:
        return 100

    def normalization_spec(self) -> NormalizationSpec:
        return self._norm

    def _train_tf(self) -> transforms.Compose:
        items: list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]
        if self.random_rotation_degrees > 0.0:
            items.append(transforms.RandomRotation(degrees=self.random_rotation_degrees))
        items.append(transforms.ToTensor())
        if self.normalize:
            items.append(transforms.Normalize(self._norm.mean, self._norm.std))
        return transforms.Compose(items)

    def _eval_tf(self) -> transforms.Compose:
        items: list = [transforms.ToTensor()]
        if self.normalize:
            items.append(transforms.Normalize(self._norm.mean, self._norm.std))
        return transforms.Compose(items)

    def _split_indices(self, n_samples: int) -> tuple[list[int], list[int]]:
        val_size = int(round(n_samples * self.val_split))
        if val_size <= 0 or val_size >= n_samples:
            raise ValueError(f"Invalid data.val_split={self.val_split} for {n_samples} samples")
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n_samples, generator=generator).tolist()
        return indices[val_size:], indices[:val_size]

    def id_dataloaders(self, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
        train_tf = self._train_tf()
        eval_tf = self._eval_tf()
        if self.val_from_train and self.val_split > 0.0:
            train_base = datasets.CIFAR100(self.root, train=True, download=True, transform=train_tf)
            val_tf = train_tf if self.val_use_train_transform else eval_tf
            val_base = datasets.CIFAR100(self.root, train=True, download=True, transform=val_tf)
            train_idx, val_idx = self._split_indices(len(train_base))
            train_ds = Subset(train_base, train_idx)
            val_ds = Subset(val_base, val_idx)
        else:
            train_ds = datasets.CIFAR100(self.root, train=True, download=True, transform=train_tf)
            val_ds = datasets.CIFAR100(self.root, train=False, download=True, transform=eval_tf)
        test_ds = datasets.CIFAR100(self.root, train=False, download=True, transform=eval_tf)
        return {
            "train": make_deterministic_loader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, seed=self.seed),
            "val": make_deterministic_loader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, seed=self.seed),
            "test": make_deterministic_loader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, seed=self.seed),
        }

    def _ood_resize_eval_tf(self) -> transforms.Compose:
        items: list = [
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
        if self.normalize:
            items.append(transforms.Normalize(self._norm.mean, self._norm.std))
        return transforms.Compose(items)

    def ood_dataloaders(
        self, names: Iterable[str], batch_size: int, num_workers: int
    ) -> Dict[str, DataLoader]:
        from src.data.tiny_imagenet import TinyImageNetOOD

        eval_tf = self._eval_tf()
        resize_tf = self._ood_resize_eval_tf()
        out: Dict[str, DataLoader] = {}
        for name in names:
            key = str(name).lower()
            if key == "svhn":
                ds = datasets.SVHN(self.root, split="test", download=True, transform=eval_tf)
            elif key == "cifar10":
                ds = datasets.CIFAR10(self.root, train=False, download=True, transform=eval_tf)
            elif key in ("dtd", "textures"):
                ds = datasets.DTD(self.root, split="test", download=True, transform=resize_tf)
            elif key in ("places365", "places"):
                ds = datasets.Places365(
                    self.root, split="val", small=True, download=True, transform=resize_tf
                )
            elif key in ("tinyimagenet", "tiny_imagenet", "tin"):
                # F-EDL CIFAR-100 OOD set: 10 000 val images at 64x64,
                # resized to 32x32 to match the CIFAR-100 input pipeline.
                ds = TinyImageNetOOD(
                    root=str(Path(self.root) / "tiny-imagenet"), transform=resize_tf,
                )
            else:
                raise ValueError(f"Unsupported OOD dataset for CIFAR-100 ID: {name}")
            out[name] = make_deterministic_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, seed=self.seed)
        return out
