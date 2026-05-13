"""Reliability diagrams from evaluation checkpoints.

Reloads each `best` checkpoint listed in `runs/<method>/seed_<n>/<ts>/summary.json`,
re-runs the ID test loader, and draws one reliability diagram per method
averaged across seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from src.data.datamodule import FIEDLDataModule
from src.metrics.ood_metrics import reliability_bins
from src.models.lit_module import FIEDLLightningModule


def _iter_summaries(runs_dir: Path):
    for path in sorted(runs_dir.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        cfg_dict = summary.get("resolved_config")
        ckpt = summary.get("best_model_path")
        if not cfg_dict or not ckpt or not Path(ckpt).exists():
            continue
        yield summary, OmegaConf.create(cfg_dict), ckpt


@torch.no_grad()
def _probs_labels(cfg, ckpt: str) -> tuple[np.ndarray, np.ndarray]:
    dm = FIEDLDataModule(cfg)
    dm.setup()
    model = FIEDLLightningModule.load_from_checkpoint(ckpt, cfg=cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for x, y in dm.test_dataloader():
        out = model(x.to(device))
        probs.append(out["probs"].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results/reliability.pdf")
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    buckets: Dict[str, List[dict]] = {}
    for summary, cfg, ckpt in _iter_summaries(Path(args.runs)):
        probs, labels = _probs_labels(cfg, ckpt)
        bins = reliability_bins(probs=probs, labels=labels, n_bins=args.n_bins)
        method = str(cfg.experiment.name)
        buckets.setdefault(method, []).append(bins)

    if not buckets:
        raise SystemExit("No runs with loadable checkpoints found under " + args.runs)

    fig, axes = plt.subplots(1, len(buckets), figsize=(4 * len(buckets), 4), squeeze=False)
    for ax, (method, runs) in zip(axes[0], sorted(buckets.items())):
        centers = runs[0]["bin_centers"]
        accs = np.nanmean(np.stack([r["accuracies"] for r in runs]), axis=0)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.bar(centers, accs, width=1.0 / args.n_bins, alpha=0.6, edgecolor="black", label=method)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(method)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Wrote {args.out} with {sum(len(v) for v in buckets.values())} runs")


if __name__ == "__main__":
    main()
