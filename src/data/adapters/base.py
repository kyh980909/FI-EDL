"""Dataset adapter contract and deterministic DataLoader helper."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class NormalizationSpec:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_deterministic_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator if shuffle else None,
        drop_last=drop_last,
        persistent_workers=bool(num_workers > 0),
    )


class DatasetAdapter(ABC):
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def normalization_spec(self) -> NormalizationSpec:
        raise NotImplementedError

    @abstractmethod
    def id_dataloaders(self, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
        raise NotImplementedError

    @abstractmethod
    def ood_dataloaders(
        self, names: Iterable[str], batch_size: int, num_workers: int
    ) -> Dict[str, DataLoader]:
        raise NotImplementedError
