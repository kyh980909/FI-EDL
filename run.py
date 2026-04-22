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

    # Train
    for method in methods:
        for seed in seeds:
            run_cmd(
                [sys.executable, "-m", "src.train", f"experiment={method}", f"seed={seed}"]
                + merged
            )

    if not preset.get("run_eval", True):
        return

    # Eval (locates the most recent run per (method, seed) from runs/)
    for method in methods:
        for seed in seeds:
            run_root = Path("runs") / method / f"seed_{seed}"
            if not run_root.exists():
                continue
            summaries = sorted(run_root.glob("*/summary.json"))
            if not summaries:
                continue
            summary = yaml.safe_load(summaries[-1].read_text(encoding="utf-8"))
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
