"""Build Table 2 — OOD detection (AUROC / AUPR / FPR95) per (method × dataset × score).

Rows with `split == "eval"` are aggregated across seeds by mean ± std.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._loader import load_runs


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    eval_df = df[df["split"] == "eval"].copy()
    if eval_df.empty:
        return eval_df
    metric_cols = ["auroc", "aupr", "fpr95"]
    group = eval_df.groupby(["method", "dataset", "score_type"])
    agg = group[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs", help="Runs directory")
    parser.add_argument("--out", default="results/table_ood.csv", help="Output CSV path")
    args = parser.parse_args()

    df = load_runs(Path(args.runs))
    table = summarize(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out} with {len(table)} rows")


if __name__ == "__main__":
    main()
