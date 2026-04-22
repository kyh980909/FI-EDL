"""Stop training immediately when a monitored loss becomes NaN/Inf."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pytorch_lightning as pl


class NaNStopCallback(pl.Callback):
    def __init__(
        self,
        monitor_keys: Iterable[str] = (
            "train/loss",
            "val/loss",
            "train/loss_epoch",
            "val/loss_epoch",
        ),
        flag_path: str | None = None,
    ) -> None:
        super().__init__()
        self.keys = tuple(monitor_keys)
        self.flag_path = flag_path
        self.triggered = False

    def _write_flag(self, stage: str, key: str, value: float) -> None:
        if self.flag_path is None:
            return
        path = Path(self.flag_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"stage={stage} key={key} value={value}\n", encoding="utf-8")

    def _check(self, trainer: pl.Trainer, stage: str) -> None:
        if self.triggered:
            return
        for key in self.keys:
            raw = trainer.callback_metrics.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isnan(value) or math.isinf(value):
                self.triggered = True
                trainer.should_stop = True
                self._write_flag(stage, key, value)
                return

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._check(trainer, "train")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._check(trainer, "val")
