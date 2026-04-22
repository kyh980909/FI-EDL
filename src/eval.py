"""Evaluation entry point. Invoked via `python -m src.eval checkpoint=<path>`.

Computes OOD detection metrics per-score (maxp, alpha0, vacuity) and
confidence metrics for the in-distribution test set, then appends rows to the
run's `metrics.jsonl`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig as OmegaListConfig
from sklearn.metrics import average_precision_score

from src.data.datamodule import FIEDLDataModule
from src.metrics.ood_metrics import (
    aupr,
    aurc_from_confidence,
    auroc_and_fpr95,
    binary_auroc_and_fpr95,
    multiclass_ece,
    multiclass_nll,
)
from src.models.lit_module import FIEDLLightningModule
from src.reporting.collector import LocalCollector


def _enable_checkpoint_safe_globals() -> None:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([OmegaDictConfig, OmegaListConfig])


def _temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("eval.temperature must be > 0")
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / float(temperature)
    return F.softmax(torch.from_numpy(scaled).float(), dim=1).numpy()


def _collect_outputs(
    model: FIEDLLightningModule, loader, temperature: float
) -> Dict[str, np.ndarray]:
    model.eval()
    all_alpha: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(model.device))
            all_alpha.append(out["alpha"].detach().cpu().numpy())
            all_probs.append(out["probs"].detach().cpu().numpy())
            all_labels.append(y.numpy())
    probs = _temperature_scale_probs(np.concatenate(all_probs), temperature=temperature)
    return {
        "alpha": np.concatenate(all_alpha),
        "probs": probs,
        "labels": np.concatenate(all_labels),
    }


def _score_map(alpha: np.ndarray, probs: np.ndarray) -> Dict[str, np.ndarray]:
    alpha0 = alpha.sum(axis=1)
    vacuity = alpha.shape[1] / np.clip(alpha0, 1e-12, None)
    return {
        "maxp": probs.max(axis=1),
        "maxalpha": alpha.max(axis=1),
        "alpha0": alpha0,
        "vacuity": vacuity,
    }


def _ood_score_from_raw(score_name: str, raw: np.ndarray) -> np.ndarray:
    # Convention: higher => more OOD. `maxp` and `alpha0` are confidence-like,
    # so we negate them. `vacuity` is already uncertainty-like.
    if score_name in {"maxp", "alpha0"}:
        return -raw
    return raw


def _confidence_aupr(conf_scores: np.ndarray, correct: np.ndarray) -> float:
    labels = correct.astype(np.int32)
    if labels.min() == labels.max():
        return float("nan")
    return float(average_precision_score(labels, conf_scores))


def run_eval(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    _enable_checkpoint_safe_globals()

    if not cfg.checkpoint:
        raise ValueError("checkpoint must be provided for eval")

    collector = LocalCollector(cfg)
    datamodule = FIEDLDataModule(cfg)
    datamodule.setup()

    model = FIEDLLightningModule.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
    device = "cuda" if torch.cuda.is_available() and cfg.trainer.accelerator != "cpu" else "cpu"
    model.to(device)

    temperature = float(cfg.eval.temperature if cfg.eval.calibration == "temperature" else 1.0)
    id_out = _collect_outputs(model, datamodule.test_dataloader(), temperature=temperature)
    id_scores_raw = _score_map(alpha=id_out["alpha"], probs=id_out["probs"])
    id_pred = id_out["probs"].argmax(axis=1)
    id_correct = (id_pred == id_out["labels"]).astype(np.float32)

    accuracy = float(id_correct.mean())
    nll = multiclass_nll(id_out["probs"], id_out["labels"])
    ece = multiclass_ece(id_out["probs"], id_out["labels"])
    aurc = aurc_from_confidence(id_out["probs"].max(axis=1), id_correct)

    # ID confidence metrics (correct-vs-wrong discrimination)
    for score_name in list(getattr(cfg.eval, "confidence_scores", ["maxp", "maxalpha"])):
        if score_name not in id_scores_raw:
            continue
        conf_score = id_scores_raw[score_name]
        conf_aupr = _confidence_aupr(conf_score, id_correct)
        conf_auroc, conf_fpr95, conf_meta = binary_auroc_and_fpr95(id_correct, conf_score)
        collector.append_metric(
            method=cfg.experiment.name,
            seed=cfg.seed,
            dataset=str(cfg.data.id),
            split="conf_eval",
            metrics={
                "accuracy": accuracy,
                "nll": nll,
                "ece": ece,
                "aurc": aurc,
                "auroc": conf_auroc,
                "aupr": conf_aupr,
                "fpr95": conf_fpr95,
            },
            method_variant=str(cfg.experiment.method_variant),
            score_type=score_name,
            calibration_type=str(cfg.eval.calibration),
            extra={"threshold_meta": conf_meta, "positive_class": "correct_prediction"},
        )

    # OOD detection metrics per (dataset × score)
    for name, loader in datamodule.ood_dataloaders().items():
        ood_out = _collect_outputs(model, loader, temperature=temperature)
        ood_scores_raw = _score_map(alpha=ood_out["alpha"], probs=ood_out["probs"])
        for score_name in list(cfg.eval.scores):
            id_score = _ood_score_from_raw(score_name, id_scores_raw[score_name])
            ood_score = _ood_score_from_raw(score_name, ood_scores_raw[score_name])
            auroc, fpr95, meta = auroc_and_fpr95(id_scores=id_score, ood_scores=ood_score)
            aupr_v = aupr(id_scores=id_score, ood_scores=ood_score)
            collector.append_metric(
                method=cfg.experiment.name,
                seed=cfg.seed,
                dataset=name,
                split="eval",
                metrics={
                    "accuracy": accuracy,
                    "nll": nll,
                    "ece": ece,
                    "aurc": aurc,
                    "auroc": auroc,
                    "aupr": aupr_v,
                    "fpr95": fpr95,
                },
                method_variant=str(cfg.experiment.method_variant),
                score_type=score_name,
                calibration_type=str(cfg.eval.calibration),
                extra={"threshold_meta": meta},
            )

    collector.write_summary(
        {
            "checkpoint": str(Path(cfg.checkpoint).resolve()),
            "seed": cfg.seed,
            "experiment": cfg.experiment.name,
            "id_accuracy": accuracy,
            "calibration": str(cfg.eval.calibration),
            "temperature": temperature,
            "scores": list(cfg.eval.scores),
        }
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
