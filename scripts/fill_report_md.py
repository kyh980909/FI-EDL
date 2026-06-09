"""Populate results/baseline_comparison_report.md with table values from CSVs.

Reads:
    results/table1_extended.csv  (calibration: ece/nll/accuracy/aurc per method+dataset)
    results/table2_extended.csv  (OOD: auroc/aupr/fpr95 per method+dataset+score)
    results/table3_extended.csv  (conf_eval: misclassification metrics per method+dataset+score)

Replaces `_TBD_` cells in-place and writes back to the same md file.
Selects a single score_type per method following project conventions
(`maxalpha` for evidential methods on CIFAR-10 OOD, `maxp` otherwise, but
this default can be overridden via --score).
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


METHOD_ALIASES = {
    "EDL": ["edl_fixed_l1", "edl_l1", "edl"],
    "I-EDL": ["i_edl"],
    "R-EDL": ["r_edl"],
    "Re-EDL": ["re_edl"],
    "Re-EDL (λ=0.8)": ["re_edl"],
    "DAEDL": ["daedl"],
    "F-EDL": ["f_edl"],
    "FI-EDL": ["fi_edl"],
}


def _fmt(mean: float | None, std: float | None, pct: bool = False, places: int = 2) -> str:
    if mean is None or pd.isna(mean):
        return "n/a"
    scale = 100.0 if pct else 1.0
    if std is None or pd.isna(std):
        return f"{mean*scale:.{places}f}"
    return f"{mean*scale:.{places}f} ± {std*scale:.{places}f}"


def _row_for_method(df: pd.DataFrame, label: str, dataset: str,
                    extra_filter=None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    aliases = METHOD_ALIASES.get(label, [label.lower().replace("-", "_")])
    subset = df[(df["method"].isin(aliases)) & (df["dataset"] == dataset)]
    if extra_filter is not None:
        subset = extra_filter(subset)
    if subset.empty:
        return None
    return subset.iloc[0]


def fill_calibration(md: str, df: pd.DataFrame, dataset: str, methods: list[str]) -> str:
    """Fill ECE/NLL/Acc/AURC rows for one dataset table."""
    for label in methods:
        row = _row_for_method(df, label, dataset)
        if row is None:
            continue
        ece = _fmt(row.get("ece_mean"), row.get("ece_std"), pct=True)
        nll = _fmt(row.get("nll_mean"), row.get("nll_std"), pct=False, places=4)
        acc = _fmt(row.get("accuracy_mean"), row.get("accuracy_std"), pct=True)
        aurc = _fmt(row.get("aurc_mean"), row.get("aurc_std"), pct=False, places=4)
        pattern = rf"\| {re.escape(label)} \| _TBD_ \| _TBD_ \| _TBD_ \| _TBD_ \|"
        md = re.sub(pattern, f"| {label} | {ece} | {nll} | {acc} | {aurc} |",
                    md, count=1)
        # Bolded FI-EDL variant
        pattern_b = rf"\| \*\*{re.escape(label)}\*\* \| _TBD_ \| _TBD_ \| _TBD_ \| _TBD_ \|"
        md = re.sub(pattern_b, f"| **{label}** | {ece} | {nll} | {acc} | {aurc} |",
                    md, count=1)
    return md


def fill_ood(md: str, df: pd.DataFrame, dataset: str, methods: list[str],
             score_type: str) -> str:
    for label in methods:
        row = _row_for_method(
            df, label, dataset,
            extra_filter=lambda s, sc=score_type: s[s["score_type"] == sc],
        )
        if row is None:
            continue
        au = _fmt(row.get("auroc_mean"), row.get("auroc_std"), pct=True)
        ap = _fmt(row.get("aupr_mean"), row.get("aupr_std"), pct=True)
        fpr = _fmt(row.get("fpr95_mean"), row.get("fpr95_std"), pct=True)
        pattern = rf"\| {re.escape(label)} \| _TBD_ \| _TBD_ \| _TBD_ \|"
        md = re.sub(pattern, f"| {label} | {au} | {ap} | {fpr} |",
                    md, count=1)
        pattern_b = rf"\| \*\*{re.escape(label)}\*\* \| _TBD_ \| _TBD_ \| _TBD_ \|"
        md = re.sub(pattern_b, f"| **{label}** | {au} | {ap} | {fpr} |",
                    md, count=1)
    return md


def fill_conf(md: str, df: pd.DataFrame, dataset: str, methods: list[str],
              score_type: str) -> str:
    for label in methods:
        row = _row_for_method(
            df, label, dataset,
            extra_filter=lambda s, sc=score_type: s[s["score_type"] == sc],
        )
        if row is None:
            continue
        acc = _fmt(row.get("accuracy_mean"), row.get("accuracy_std"), pct=True)
        au = _fmt(row.get("auroc_mean"), row.get("auroc_std"), pct=True)
        ap = _fmt(row.get("aupr_mean"), row.get("aupr_std"), pct=True)
        fpr = _fmt(row.get("fpr95_mean"), row.get("fpr95_std"), pct=True)
        pattern = rf"\| {re.escape(label)} \| _TBD_ \| _TBD_ \| _TBD_ \| _TBD_ \|"
        md = re.sub(pattern, f"| {label} | {acc} | {au} | {ap} | {fpr} |",
                    md, count=1)
        pattern_b = rf"\| \*\*{re.escape(label)}\*\* \| _TBD_ \| _TBD_ \| _TBD_ \| _TBD_ \|"
        md = re.sub(pattern_b, f"| **{label}** | {acc} | {au} | {ap} | {fpr} |",
                    md, count=1)
    return md


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--md", default="results/baseline_comparison_report.md")
    p.add_argument("--ece-csv", default="results/table1_extended.csv")
    p.add_argument("--ood-csv", default="results/table2_extended.csv")
    p.add_argument("--conf-csv", default="results/table3_extended.csv")
    p.add_argument("--score-mnist", default="maxp",
                   help="OOD score_type to use for MNIST tables")
    p.add_argument("--score-cifar10", default="maxp",
                   help="OOD score_type to use for CIFAR-10 tables")
    args = p.parse_args()

    md_path = Path(args.md)
    md = md_path.read_text(encoding="utf-8")

    ece = pd.read_csv(args.ece_csv) if Path(args.ece_csv).exists() else None
    ood = pd.read_csv(args.ood_csv) if Path(args.ood_csv).exists() else None
    conf = pd.read_csv(args.conf_csv) if Path(args.conf_csv).exists() else None

    methods_mnist = ["EDL", "I-EDL", "R-EDL", "Re-EDL", "DAEDL", "F-EDL", "FI-EDL"]
    methods_cifar = ["EDL", "I-EDL", "R-EDL", "Re-EDL (λ=0.8)", "DAEDL", "F-EDL", "FI-EDL"]

    if ece is not None:
        md = fill_calibration(md, ece, "mnist", methods_mnist)
        md = fill_calibration(md, ece, "cifar10", methods_cifar)
    if ood is not None:
        md = fill_ood(md, ood, "kmnist", methods_mnist, args.score_mnist)
        md = fill_ood(md, ood, "fmnist", methods_mnist, args.score_mnist)
        md = fill_ood(md, ood, "svhn", methods_cifar, args.score_cifar10)
        md = fill_ood(md, ood, "cifar100", methods_cifar, args.score_cifar10)
    if conf is not None:
        md = fill_conf(md, conf, "cifar10", methods_cifar, args.score_cifar10)

    md = md.replace("_filled at aggregation time_",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"))

    md_path.write_text(md, encoding="utf-8")
    print(f"Updated {md_path}")


if __name__ == "__main__":
    main()
