"""Preset-driven experiment runner.

Usage:
    python run.py preset <preset_name> [hydra_override ...]

Each preset (under `configs/paper/`) names a list of `methods`, a list of
`seeds`, and common Hydra overrides. The runner trains each (method, seed)
pair and optionally evaluates every resulting checkpoint.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


def load_preset(name: str) -> Dict:
    path = Path("configs/paper") / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _iter_run_dirs(run_root: Path, prefix: str):
    if not run_root.exists():
        return
    # Prefer the new `<kind>_<ts>_...` layout, but also match legacy timestamp
    # directories that contain a summary of the corresponding kind.
    seen: set = set()
    for child in sorted(run_root.iterdir()):
        if not child.is_dir() or child.name in seen:
            continue
        if child.name.startswith(f"{prefix}_"):
            seen.add(child.name)
            yield child
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


def _find_latest_train_summary(run_root: Path):
    train_dirs = sorted(_iter_run_dirs(run_root, "train"))
    for candidate in reversed(train_dirs):
        summary = candidate / "summary.json"
        if summary.exists():
            return summary
    return None


def _has_eval_run(run_root: Path) -> bool:
    for candidate in _iter_run_dirs(run_root, "eval"):
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

    # Train (skip seeds that already have a completed training run)
    for method in methods:
        for seed in seeds:
            run_root = Path("runs") / method / f"seed_{seed}"
            if _find_latest_train_summary(run_root) is not None:
                print(f"[SKIP train] {method} seed={seed} (checkpoint exists)")
                continue
            run_cmd(
                [sys.executable, "-m", "src.train", f"experiment={method}", f"seed={seed}"]
                + merged
            )

    if not preset.get("run_eval", True):
        return

    # Eval (locates the most recent train run per (method, seed); skips if an
    # eval run already exists).
    for method in methods:
        for seed in seeds:
            run_root = Path("runs") / method / f"seed_{seed}"
            if not run_root.exists():
                print(f"[SKIP eval] {method} seed={seed} (no run dir)")
                continue
            if _has_eval_run(run_root):
                print(f"[SKIP eval] {method} seed={seed} (already evaluated)")
                continue
            summary_path = _find_latest_train_summary(run_root)
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
