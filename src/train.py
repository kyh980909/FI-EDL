"""Training entry point. Invoked via `python -m src.train experiment=<name>`.

Layout:
  helpers (determinism, checkpoint safe globals) → `run_train` → `main`
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.callbacks.nan_detector import NaNStopCallback
from src.data.datamodule import FIEDLDataModule
from src.models.lit_module import FIEDLLightningModule
from src.registry.validators import validate_registry_bindings
from src.reporting.collector import LocalCollector


def _apply_determinism(cfg: DictConfig) -> None:
    if not bool(OmegaConf.select(cfg, "trainer.deterministic", default=True)):
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(int(cfg.seed)))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_checkpoint_safe_globals() -> None:
    # PyTorch 2.6 weights_only loader rejects unknown globals; OmegaConf
    # classes may be referenced inside Lightning checkpoints.
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([OmegaDictConfig, OmegaListConfig])


def run_train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    _apply_determinism(cfg)
    _enable_checkpoint_safe_globals()
    validate_registry_bindings(cfg)

    collector = LocalCollector(cfg)
    datamodule = FIEDLDataModule(cfg)
    model = FIEDLLightningModule(cfg)

    monitor = str(cfg.trainer.early_stopping_monitor) if bool(cfg.trainer.early_stopping) else "val/acc"
    mode = str(cfg.trainer.early_stopping_mode) if bool(cfg.trainer.early_stopping) else "max"
    ckpt = ModelCheckpoint(
        dirpath=str(Path(collector.run_dir) / "checkpoints"),
        save_top_k=1,
        monitor=monitor,
        mode=mode,
        filename="best",
    )
    callbacks = [
        ckpt,
        NaNStopCallback(flag_path=str(Path(collector.run_dir) / "NAN_STOP.flag")),
    ]
    if bool(cfg.trainer.early_stopping):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.trainer.early_stopping_monitor),
                mode=str(cfg.trainer.early_stopping_mode),
                patience=int(cfg.trainer.early_stopping_patience),
            )
        )

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=str(cfg.trainer.precision),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        limit_train_batches=OmegaConf.select(cfg, "trainer.limit_train_batches", default=1.0),
        limit_val_batches=OmegaConf.select(cfg, "trainer.limit_val_batches", default=1.0),
        logger=CSVLogger(save_dir=cfg.logging.local_dir, name=cfg.experiment.name),
        callbacks=callbacks,
        deterministic=bool(OmegaConf.select(cfg, "trainer.deterministic", default=True)),
    )

    trainer.fit(model, datamodule=datamodule)
    test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    metrics = test_metrics[0] if test_metrics else {}
    collector.append_metric(
        method=cfg.experiment.name,
        seed=cfg.seed,
        dataset=cfg.data.id,
        split="test",
        metrics={"accuracy": float(metrics.get("test/acc", 0.0))},
        method_variant=str(cfg.experiment.method_variant),
        score_type=str(cfg.score.name),
        calibration_type=str(cfg.eval.calibration),
    )
    collector.write_summary(
        {
            "best_model_path": ckpt.best_model_path,
            "seed": cfg.seed,
            "experiment": cfg.experiment.name,
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        }
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_train(cfg)


if __name__ == "__main__":
    main()
