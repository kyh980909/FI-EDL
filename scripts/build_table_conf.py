"""Build Table 3 — ID confidence evaluation (AUROC / AUPR / FPR95 on correct-vs-wrong).

Uses rows with `split == "conf_eval"`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._loader import load_runs


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    conf_df = df[df["split"] == "conf_eval"].copy()
    if conf_df.empty:
        return conf_df
    metric_cols = ["auroc", "aupr", "fpr95", "accuracy", "nll", "ece", "aurc"]
    group = conf_df.groupby(["method", "dataset", "score_type"])
    agg = group[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="results/table_conf.csv")
    args = parser.parse_args()

    df = load_runs(Path(args.runs))
    table = summarize(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {out} with {len(table)} rows")


if __name__ == "__main__":
    main()
