"""Per-run artifact collector.

Writes the resolved config, per-metric JSONL rows, and a final summary JSON
under `runs/<experiment>/seed_<n>/<kind>_<timestamp>_<dataset>_<backbone>[_<variant>]/`.
The directory name embeds key run attributes so that a plain `ls` tells you
what each directory contains without opening files. All outputs use UTF-8 and
are safe to read with the table/plot scripts in `scripts/`.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from omegaconf import OmegaConf

from src.contracts.schemas import RESULTS_SCHEMA_VERSION


_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: Any) -> str:
    text = str(value).strip()
    text = _SAFE_TOKEN.sub("-", text)
    return text or "x"


def build_run_name(cfg, kind: str) -> str:
    """Human-readable run identifier used for both dir name and wandb run name."""
    parts = [
        _slug(kind),
        _slug(OmegaConf.select(cfg, "experiment.name", default="run")),
        _slug(OmegaConf.select(cfg, "data.id", default="data")),
        _slug(OmegaConf.select(cfg, "model.backbone", default="backbone")),
        f"seed{int(OmegaConf.select(cfg, 'seed', default=0))}",
    ]
    variant = OmegaConf.select(cfg, "experiment.method_variant", default="")
    if variant and str(variant) != OmegaConf.select(cfg, "experiment.name", default=""):
        parts.append(_slug(variant))
    return "_".join(parts)


class LocalCollector:
    def __init__(self, cfg, kind: str = "train") -> None:
        method = cfg.experiment.name
        seed = cfg.seed
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.kind = _slug(kind)
        self.run_name = build_run_name(cfg, kind=self.kind)
        dir_name = f"{self.kind}_{ts}_{_slug(cfg.data.id)}_{_slug(cfg.model.backbone)}"
        variant = OmegaConf.select(cfg, "experiment.method_variant", default="")
        if variant and str(variant) != str(method):
            dir_name = f"{dir_name}_{_slug(variant)}"
        self.run_dir = Path(cfg.logging.local_dir) / method / f"seed_{seed}" / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        resolved = OmegaConf.to_container(cfg, resolve=True)
        (self.run_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        (self.run_dir / "env.json").write_text(
            json.dumps(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "results_schema_version": RESULTS_SCHEMA_VERSION,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cfg_bytes = json.dumps(resolved, sort_keys=True).encode("utf-8")
        self.config_hash = hashlib.sha256(cfg_bytes).hexdigest()[:12]

    def append_metric(
        self,
        method: str,
        seed: int,
        dataset: str,
        split: str,
        metrics: Dict[str, float],
        method_variant: str = "",
        score_type: str = "",
        calibration_type: str = "",
        step: int | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        row = {
            "results_schema_version": RESULTS_SCHEMA_VERSION,
            "method": method,
            "seed": seed,
            "dataset": dataset,
            "split": split,
            "metrics": metrics,
            "method_variant": method_variant,
            "score_type": score_type,
            "calibration_type": calibration_type,
            "config_hash": self.config_hash,
            "step": step,
            "extra": extra or {},
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        out = {
            "results_schema_version": RESULTS_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "summary": summary,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
