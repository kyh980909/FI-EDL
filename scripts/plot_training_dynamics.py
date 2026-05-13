"""Plot training dynamics (loss, Fisher trace, lambda) from each run's CSV log."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


KEYS = ("Loss/Total", "Metric/Fisher_Trace", "Metric/Lambda_Mean")


def _collect(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    keep = [c for c in df.columns if c in KEYS or c == "epoch"]
    if "epoch" not in keep:
        return pd.DataFrame()
    return df[keep].dropna(how="all", subset=[c for c in keep if c != "epoch"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results/dynamics.pdf")
    args = parser.parse_args()

    per_method: Dict[str, List[pd.DataFrame]] = {}
    for csv_path in sorted(Path(args.runs).rglob("metrics.csv")):
        parts = csv_path.parts
        method = next((p for p in parts[::-1] if p in {"edl_l1", "edl_l01", "edl_l0001", "fi_edl", "i_edl"}), None)
        if method is None:
            continue
        df = _collect(csv_path)
        if not df.empty:
            per_method.setdefault(method, []).append(df)

    if not per_method:
        raise SystemExit("No metrics.csv found under " + args.runs)

    fig, axes = plt.subplots(1, len(KEYS), figsize=(5 * len(KEYS), 4), squeeze=False)
    for method, frames in sorted(per_method.items()):
        avg = pd.concat(frames).groupby("epoch").mean()
        for ax, key in zip(axes[0], KEYS):
            if key in avg.columns:
                ax.plot(avg.index, avg[key], label=method)
    for ax, key in zip(axes[0], KEYS):
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.legend(fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
