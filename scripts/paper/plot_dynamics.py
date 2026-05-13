"""Plot training dynamics (vFIM, lambda) per epoch for FI-EDL across seeds.

Reads CSVLogger metrics.csv from `runs/{cifar10,mnist}/fi_edl/
train_fi_edl_{ds}_<bb>_seed{N}/metrics.csv`. Skips controller/sensitivity
ablation runs (suffix on the train dir).

Outputs PNG figures to ``results/paper/dynamics/``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABEL = "FI-EDL"
EXCLUDE_KEYS = ("constant", "alpha0", "nodetach", "fim_", "bg_b")


def is_main_train(name: str) -> bool:
    return not any(k in name for k in EXCLUDE_KEYS)


def load_seeds(runs_root: Path, dataset: str, backbone: str, seeds: Iterable[int]):
    """Return dict: seed -> {col: np.array} for FI-EDL main runs."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        # CSVLogger writes to runs/<root>/fi_edl/train_fi_edl_<ds>_<bb>_seed<N>/metrics.csv
        # The dir name has no ablation suffix for the main run.
        candidates = sorted(runs_root.glob(f"fi_edl/train_fi_edl_{dataset}_{backbone}_seed{seed}/metrics.csv"))
        candidates = [p for p in candidates if is_main_train(p.parent.name)]
        if not candidates:
            continue
        path = candidates[-1]
        rows = list(csv.DictReader(open(path)))
        cols = [
            "epoch",
            "Metric/Fisher_Trace_epoch",
            "Metric/Lambda_Mean_epoch",
            "Metric/Lambda_Std_epoch",
            "Metric/Lambda_Min_epoch",
            "Metric/Lambda_Max_epoch",
            "Loss/KL_raw_epoch",
            "Loss/KL_weighted_epoch",
            "Loss/Risk_epoch",
            "Loss/Total_epoch",
            "val/loss",
            "val/acc",
        ]
        # Index rows by epoch, taking the last non-empty entry per epoch per col
        per_epoch: dict[int, dict[str, float]] = {}
        for r in rows:
            try:
                e = int(float(r["epoch"]))
            except Exception:
                continue
            slot = per_epoch.setdefault(e, {})
            for c in cols[1:]:
                v = r.get(c, "")
                if v not in (None, ""):
                    try:
                        slot[c] = float(v)
                    except Exception:
                        pass
        if not per_epoch:
            continue
        epochs = sorted(per_epoch.keys())
        arr: dict[str, np.ndarray] = {"epoch": np.array(epochs, dtype=float)}
        for c in cols[1:]:
            arr[c] = np.array([per_epoch[e].get(c, np.nan) for e in epochs])
        out[seed] = arr
    return out


def _safe_log(ax, values_list):
    """Set y-scale to log only when all plotted values are strictly positive."""
    cat = np.concatenate([np.asarray(v, dtype=float) for v in values_list]) if values_list else np.array([])
    cat = cat[~np.isnan(cat)]
    if len(cat) and np.all(cat > 0):
        ax.set_yscale("log")


def _plot_per_seed(ax, data, col, label, ylabel, log=False):
    plotted = []
    for seed, d in sorted(data.items()):
        mask = ~np.isnan(d[col])
        if not mask.any():
            continue
        ax.plot(d["epoch"][mask], d[col][mask], label=f"seed={seed}", alpha=0.85)
        plotted.append(d[col][mask])
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    if log:
        _safe_log(ax, plotted)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def _plot_lambda_band(ax, data, label):
    plotted = []
    for seed, d in sorted(data.items()):
        mask = ~np.isnan(d["Metric/Lambda_Mean_epoch"])
        if not mask.any():
            continue
        ax.plot(d["epoch"][mask], d["Metric/Lambda_Mean_epoch"][mask],
                alpha=0.7, label=f"seed={seed}")
        plotted.append(d["Metric/Lambda_Mean_epoch"][mask])
    ax.set_xlabel("epoch")
    ax.set_ylabel("λ mean")
    ax.set_title(label)
    _safe_log(ax, plotted)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def render(runs_root: Path, dataset: str, backbone: str, seeds, out_dir: Path):
    data = load_seeds(runs_root, dataset, backbone, seeds)
    if not data:
        print(f"[WARN] no data for {dataset} {backbone} seeds={seeds}")
        return

    # 4-panel figure: Fisher trace, λ mean, KL components, val/loss
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    title = f"FI-EDL Training Dynamics — {dataset.upper()} ({backbone}), seeds={sorted(data.keys())}"
    fig.suptitle(title, fontsize=13, y=0.99)

    _plot_per_seed(axes[0, 0], data, "Metric/Fisher_Trace_epoch",
                   "vFIM (Fisher Trace) per epoch", "Fisher Trace", log=True)
    _plot_lambda_band(axes[0, 1], data, "λ mean per epoch (log scale)")
    _plot_per_seed(axes[1, 0], data, "Loss/KL_weighted_epoch",
                   "KL (weighted) per epoch", "KL × λ", log=True)
    _plot_per_seed(axes[1, 1], data, "val/loss",
                   "Validation loss per epoch", "val/loss")

    fig.tight_layout()
    out_path = out_dir / f"fi_edl_dynamics_{dataset}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")

    # Combined Fisher + λ trends (mean ± min/max across seeds)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, ylabel, log in [
        (axes[0], "Metric/Fisher_Trace_epoch", "vFIM (Fisher Trace)", True),
        (axes[1], "Metric/Lambda_Mean_epoch", "λ mean", True),
    ]:
        seqs = [d[col] for d in data.values()]
        n = min(len(s) for s in seqs)
        stacked = np.array([s[:n] for s in seqs], dtype=float)
        epochs = next(iter(data.values()))["epoch"][:n]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stacked, axis=0)
            lo = np.nanmin(stacked, axis=0)
            hi = np.nanmax(stacked, axis=0)
        ax.plot(epochs, mean, label="seed-mean", color="C0")
        ax.fill_between(epochs, lo, hi, alpha=0.2, color="C0", label="seed min-max")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} ({dataset}, n={len(seqs)} seeds)")
        # Use log only when all visible values are strictly positive.
        with np.errstate(invalid="ignore"):
            min_pos = np.nanmin(np.where(stacked > 0, stacked, np.inf))
        if log and np.isfinite(min_pos):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = out_dir / f"fi_edl_dynamics_{dataset}_summary.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper/dynamics"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    render(Path("runs/cifar10"), "cifar10", "vgg16",
           seeds=[0, 1, 2, 3, 4], out_dir=args.out_dir)
    render(Path("runs/mnist"), "mnist", "convnet",
           seeds=[1, 2, 3, 4, 5], out_dir=args.out_dir)


if __name__ == "__main__":
    main()
