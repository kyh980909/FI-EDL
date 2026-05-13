"""Preset-driven experiment runner.

Usage:
    python run.py preset <preset_name> [hydra_override ...]

Each preset (under `configs/paper/`) names a list of `methods`, a list of
`seeds`, and common Hydra overrides. The runner trains each (method, seed)
pair and optionally evaluates every resulting checkpoint.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value) -> str:
    text = str(value).strip()
    text = _SAFE_TOKEN.sub("-", text)
    return text or "x"


def load_preset(name: str) -> Dict:
    path = Path("configs/paper") / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _override_value(overrides: List[str], key: str) -> Optional[str]:
    value = None
    prefix = f"{key}="
    for item in overrides:
        if item.startswith(prefix):
            value = item[len(prefix):]
    return value


def _run_suffix(overrides: List[str], method: str) -> Optional[str]:
    """Build the `_<dataset>_<backbone>[_<variant>]` suffix used in run dirs.

    Returns None if dataset or backbone override is missing — in that case the
    runner cannot safely disambiguate between preset variants and falls back to
    the legacy "latest only" behavior.
    """
    dataset = _override_value(overrides, "dataset") or _override_value(
        overrides, "data.id"
    )
    backbone = _override_value(overrides, "backbone") or _override_value(
        overrides, "model.backbone"
    )
    if not dataset or not backbone:
        return None
    parts = [_slug(dataset), _slug(backbone)]
    variant = _override_value(overrides, "experiment.method_variant")
    if variant and variant != method:
        parts.append(_slug(variant))
    return "_".join(parts)


def _dir_matches_suffix(name: str, prefix: str, suffix: Optional[str]) -> bool:
    if not name.startswith(f"{prefix}_"):
        return False
    if suffix is None:
        return True
    # Dir name is either `<prefix>_<ts>_<suffix>` or
    # `<prefix>_<ts>_<suffix>_<variant>` — accept both.
    return name.endswith(f"_{suffix}") or f"_{suffix}_" in name


def _iter_run_dirs(run_root: Path, prefix: str, suffix: Optional[str] = None):
    if not run_root.exists():
        return
    # Prefer the new `<kind>_<ts>_<dataset>_<backbone>[_<variant>]` layout, but
    # also match legacy timestamp directories that contain a summary of the
    # matching kind. `suffix` restricts to dirs from the same preset context.
    seen: set = set()
    for child in sorted(run_root.iterdir()):
        if not child.is_dir() or child.name in seen:
            continue
        if _dir_matches_suffix(child.name, prefix, suffix):
            seen.add(child.name)
            yield child
            continue
        if child.name.startswith(f"{prefix}_"):
            # New-layout but suffix mismatch — skip (belongs to a different
            # preset variant, e.g. MNIST run found while evaluating CIFAR10).
            continue
        summary = child / "summary.json"
        if not summary.exists():
            continue
        try:
            payload = yaml.safe_load(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        body = (payload or {}).get("summary", {})
        is_train = "best_model_path" in body
        is_eval = "id_accuracy" in body
        if (prefix == "train" and is_train) or (prefix == "eval" and is_eval):
            seen.add(child.name)
            yield child


def _find_latest_train_summary(run_root: Path, suffix: Optional[str] = None):
    train_dirs = sorted(_iter_run_dirs(run_root, "train", suffix))
    for candidate in reversed(train_dirs):
        summary = candidate / "summary.json"
        if summary.exists():
            return summary
    return None


def _has_eval_run(run_root: Path, suffix: Optional[str] = None) -> bool:
    for candidate in _iter_run_dirs(run_root, "eval", suffix):
        if (candidate / "summary.json").exists():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="FI-EDL preset runner")
    parser.add_argument("mode", choices=["preset"], help="Run mode")
    parser.add_argument("preset", help="Preset name under configs/paper/")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides")
    args = parser.parse_args()

    preset = load_preset(args.preset)
    methods = preset["methods"]
    seeds = preset["seeds"]
    preset_overrides = list(preset.get("overrides", []))
    merged = preset_overrides + list(args.overrides)

    # Train (skip seeds that already have a completed training run whose
    # dataset/backbone match this preset).
    for method in methods:
        suffix = _run_suffix(merged, method)
        for seed in seeds:
            run_root = Path("runs") / method / f"seed_{seed}"
            if _find_latest_train_summary(run_root, suffix) is not None:
                print(f"[SKIP train] {method} seed={seed} (checkpoint exists)")
                continue
            run_cmd(
                [sys.executable, "-m", "src.train", f"experiment={method}", f"seed={seed}"]
                + merged
            )

    if not preset.get("run_eval", True):
        return

    # Eval (locates the most recent train run per (method, seed) whose
    # dataset/backbone match this preset; skips if a matching eval run exists).
    for method in methods:
        suffix = _run_suffix(merged, method)
        for seed in seeds:
            run_root = Path("runs") / method / f"seed_{seed}"
            if not run_root.exists():
                print(f"[SKIP eval] {method} seed={seed} (no run dir)")
                continue
            if _has_eval_run(run_root, suffix):
                print(f"[SKIP eval] {method} seed={seed} (already evaluated)")
                continue
            summary_path = _find_latest_train_summary(run_root, suffix)
            if summary_path is None:
                print(f"[SKIP eval] {method} seed={seed} (no training checkpoint)")
                continue
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            ckpt = summary["summary"]["best_model_path"]
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "src.eval",
                    f"experiment={method}",
                    f"seed={seed}",
                    f"checkpoint={ckpt}",
                ]
                + merged
            )


if __name__ == "__main__":
    main()
