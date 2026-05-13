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
from src.reporting.collector import LocalCollector, _slug


def _find_resumable_run(cfg: DictConfig) -> Path | None:
    """Locate an interrupted train run for this (method, seed, dataset, backbone, variant).

    Criteria: a `train_*_<dataset>_<backbone>[_<variant>]/` directory that
    contains `checkpoints/last.ckpt` (or `checkpoints/best.ckpt` as a fallback)
    but lacks `summary.json` — i.e., training started but did not finish.
    The newest matching directory wins.
    """
    root = Path(cfg.logging.local_dir) / cfg.experiment.name / f"seed_{cfg.seed}"
    if not root.exists():
        return None

    dataset = _slug(cfg.data.id)
    backbone = _slug(cfg.model.backbone)
    method = str(cfg.experiment.name)
    variant = OmegaConf.select(cfg, "experiment.method_variant", default="")
    variant_slug = _slug(variant) if variant and str(variant) != method else ""

    expected_suffix = f"_{dataset}_{backbone}"
    if variant_slug:
        expected_suffix += f"_{variant_slug}"

    candidates = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("train_"):
            continue
        if not child.name.endswith(expected_suffix):
            continue
        if (child / "summary.json").exists():
            continue
        last = child / "checkpoints" / "last.ckpt"
        best = child / "checkpoints" / "best.ckpt"
        ckpt = last if last.exists() else (best if best.exists() else None)
        if ckpt is None:
            continue
        candidates.append((child.stat().st_mtime, child, ckpt))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _resume_checkpoint_path(run_dir: Path) -> str | None:
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return str(last)
    best = run_dir / "checkpoints" / "best.ckpt"
    if best.exists():
        return str(best)
    return None


def _wandb_enabled(cfg: DictConfig) -> bool:
    if bool(OmegaConf.select(cfg, "logging.wandb.enabled", default=False)):
        return True
    # Convenience env fallbacks so users can enable wandb without Hydra overrides.
    flag = os.environ.get("FIEDL_WANDB", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if os.environ.get("WANDB_PROJECT"):
        return True
    return False


def _build_loggers(cfg: DictConfig, run_name: str, run_dir):
    # `version=run_name` keeps the per-run CSV log directory unique even when
    # two src.train processes start concurrently (e.g., MNIST + CIFAR seeds of
    # the same experiment). Without it both processes auto-version to
    # `version_0` and clobber each other's metrics.csv headers.
    loggers = [CSVLogger(
        save_dir=cfg.logging.local_dir,
        name=cfg.experiment.name,
        version=run_name,
    )]
    wandb_cfg = OmegaConf.select(cfg, "logging.wandb", default=None)
    if wandb_cfg is None or not _wandb_enabled(cfg):
        print("[WANDB] disabled (pass logging.wandb.enabled=true, or set FIEDL_WANDB=1)")
        return loggers
    mode = str(OmegaConf.select(wandb_cfg, "mode", default="online"))
    if mode == "disabled":
        print("[WANDB] disabled via logging.wandb.mode=disabled")
        return loggers
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as exc:
        raise RuntimeError(
            "logging.wandb.enabled=true requires `wandb` installed. "
            "Run `uv sync --dev` after pulling the latest pyproject.toml."
        ) from exc
    project = os.environ.get("WANDB_PROJECT") or str(
        OmegaConf.select(wandb_cfg, "project", default="fi-edl")
    )
    entity = os.environ.get("WANDB_ENTITY") or OmegaConf.select(wandb_cfg, "entity", default=None)
    tags = list(OmegaConf.select(wandb_cfg, "tags", default=[]) or [])
    print(f"[WANDB] enabled  project={project}  entity={entity}  mode={mode}  run={run_name}")
    loggers.append(
        WandbLogger(
            project=project,
            entity=entity,
            name=run_name,
            save_dir=str(run_dir),
            tags=tags,
            mode=mode,
            log_model=bool(OmegaConf.select(wandb_cfg, "log_model", default=False)),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    )
    return loggers


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

    resume_dir = _find_resumable_run(cfg)
    collector = LocalCollector(cfg, kind="train", resume_dir=resume_dir)
    resume_ckpt = _resume_checkpoint_path(collector.run_dir) if resume_dir is not None else None
    if resume_ckpt:
        print(f"[RESUME] {cfg.experiment.name} seed={cfg.seed} from {resume_ckpt}")
    datamodule = FIEDLDataModule(cfg)
    model = FIEDLLightningModule(cfg)

    monitor = str(cfg.trainer.early_stopping_monitor) if bool(cfg.trainer.early_stopping) else "val/acc"
    mode = str(cfg.trainer.early_stopping_mode) if bool(cfg.trainer.early_stopping) else "max"
    ckpt = ModelCheckpoint(
        dirpath=str(Path(collector.run_dir) / "checkpoints"),
        save_top_k=1,
        save_last=True,
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
        logger=_build_loggers(cfg, run_name=collector.run_name, run_dir=collector.run_dir),
        callbacks=callbacks,
        deterministic=bool(OmegaConf.select(cfg, "trainer.deterministic", default=True)),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
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
