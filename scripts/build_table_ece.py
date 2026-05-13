"""Build Table 4 — ECE summary per (method × ID dataset).

ID-side ECE is identical across (score_type, OOD dataset) rows of the same
(method, seed), so we take the first eval row per (method, seed, dataset) and
average across seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._loader import load_runs


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["split"].isin(["eval", "conf_eval"])].copy()
    if subset.empty:
        return subset
    per_run = (
        subset.drop_duplicates(subset=["method", "seed", "dataset"])
        .groupby(["method", "dataset"])[["ece", "accuracy", "nll", "aurc"]]
        .agg(["mean", "std", "count"])
    )
    per_run.columns = [f"{m}_{stat}" for m, stat in per_run.columns]
    return per_run.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results/table_ece.csv")
    args = parser.parse_args()

    df = load_runs(Path(args.runs))
    table = summarize(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out} with {len(table)} rows")


if __name__ == "__main__":
    main()
