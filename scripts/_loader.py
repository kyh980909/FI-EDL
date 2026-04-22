"""Shared helper for loading metric rows written by `LocalCollector`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

import pandas as pd


def iter_metric_rows(runs_dir: Path) -> Iterator[dict]:
    for metrics_file in sorted(runs_dir.rglob("metrics.jsonl")):
        with metrics_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield row


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """Flatten nested `metrics` into top-level columns alongside identifiers."""
    records: List[dict] = []
    for row in iter_metric_rows(runs_dir):
        flat = {
            "method": row.get("method"),
            "method_variant": row.get("method_variant", ""),
            "seed": row.get("seed"),
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "score_type": row.get("score_type", ""),
            "calibration_type": row.get("calibration_type", ""),
            **{k: v for k, v in (row.get("metrics") or {}).items()},
        }
        records.append(flat)
    if not records:
        return pd.DataFrame(columns=["method", "seed", "dataset", "split", "score_type"])
    return pd.DataFrame.from_records(records)
