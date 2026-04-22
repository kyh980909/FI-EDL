"""Shared schema versions and dataclass shapes for losses / metric records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


LOSS_SCHEMA_VERSION = "v1"
RESULTS_SCHEMA_VERSION = "v1"


@dataclass
class LossOutput:
    total: float
    fit: float
    reg: float
    aux: Dict[str, float] = field(default_factory=dict)
    schema_version: str = LOSS_SCHEMA_VERSION


@dataclass
class MetricRecord:
    method: str
    seed: int
    dataset: str
    split: str
    metrics: Dict[str, float]
    method_variant: str = ""
    score_type: str = ""
    calibration_type: str = ""
    step: int | None = None
    extra: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULTS_SCHEMA_VERSION
