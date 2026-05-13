"""FI-EDL vs R-EDL same-pipeline comparison table.

Loads run summaries from `runs/`, filters to the matched-protocol pair
(method=fi_edl & method_variant=fi_edl  vs  method=r_edl & method_variant=r_edl),
aggregates mean ± std across seeds, and emits both a wide CSV and a
reviewer-ready Markdown table.

Usage::

    uv run python scripts/build_redl_comparison.py
    uv run python scripts/build_redl_comparison.py --runs runs --out-dir results/redl_rebuttal
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from scripts._loader import load_runs


FI_EDL_LABEL = "FI-EDL"
R_EDL_LABEL = "R-EDL"


@dataclass(frozen=True)
class ComparisonSpec:
    task: str
    score_label: str
    metric_label: str
    dataset: str
    split: str
    score_type: str
    metric: str


_SPECS: List[ComparisonSpec] = [
    ComparisonSpec("OOD CIFAR10->SVHN", "alpha0", "AUPR", "svhn", "eval", "alpha0", "aupr"),
    ComparisonSpec("OOD CIFAR10->SVHN", "alpha0", "AUROC", "svhn", "eval", "alpha0", "auroc"),
    ComparisonSpec("OOD CIFAR10->SVHN", "alpha0", "FPR95", "svhn", "eval", "alpha0", "fpr95"),
    ComparisonSpec("OOD CIFAR10->CIFAR100", "alpha0", "AUPR", "cifar100", "eval", "alpha0", "aupr"),
    ComparisonSpec("OOD CIFAR10->CIFAR100", "alpha0", "AUROC", "cifar100", "eval", "alpha0", "auroc"),
    ComparisonSpec("OOD CIFAR10->CIFAR100", "alpha0", "FPR95", "cifar100", "eval", "alpha0", "fpr95"),
    ComparisonSpec("Misclf CIFAR10", "Max.P", "AUPR", "cifar10", "conf_eval", "maxp", "aupr"),
    ComparisonSpec("Misclf CIFAR10", "Max.P", "AUROC", "cifar10", "conf_eval", "maxp", "auroc"),
    ComparisonSpec("Misclf CIFAR10", "Max.P", "FPR95", "cifar10", "conf_eval", "maxp", "fpr95"),
    ComparisonSpec("Misclf CIFAR10", "Max.alpha", "AUPR", "cifar10", "conf_eval", "maxalpha", "aupr"),
    ComparisonSpec("Calib CIFAR10", "Max.P", "ECE", "cifar10", "conf_eval", "maxp", "ece"),
    ComparisonSpec("OOD MNIST->KMNIST", "alpha0", "AUPR", "kmnist", "eval", "alpha0", "aupr"),
    ComparisonSpec("OOD MNIST->KMNIST", "alpha0", "AUROC", "kmnist", "eval", "alpha0", "auroc"),
    ComparisonSpec("OOD MNIST->KMNIST", "alpha0", "FPR95", "kmnist", "eval", "alpha0", "fpr95"),
    ComparisonSpec("OOD MNIST->FMNIST", "alpha0", "AUPR", "fmnist", "eval", "alpha0", "aupr"),
    ComparisonSpec("OOD MNIST->FMNIST", "alpha0", "AUROC", "fmnist", "eval", "alpha0", "auroc"),
    ComparisonSpec("OOD MNIST->FMNIST", "alpha0", "FPR95", "fmnist", "eval", "alpha0", "fpr95"),
    ComparisonSpec("Misclf MNIST", "Max.P", "AUPR", "mnist", "conf_eval", "maxp", "aupr"),
    ComparisonSpec("Misclf MNIST", "Max.P", "AUROC", "mnist", "conf_eval", "maxp", "auroc"),
    ComparisonSpec("Calib MNIST", "Max.P", "ECE", "mnist", "conf_eval", "maxp", "ece"),
]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "method_variant", "dataset", "split", "score_type"]
    metric_cols = [c for c in ["aupr", "auroc", "fpr95", "ece", "accuracy", "nll", "aurc"] if c in df.columns]
    grouped = df.groupby(keys)[metric_cols].agg(["mean", "std", "count"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()


def _row_for(agg: pd.DataFrame, method: str, variant: str, spec: ComparisonSpec) -> Optional[pd.Series]:
    mask = (
        (agg["method"] == method)
        & (agg["method_variant"] == variant)
        & (agg["dataset"] == spec.dataset)
        & (agg["split"] == spec.split)
        & (agg["score_type"] == spec.score_type)
    )
    sub = agg[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def _format(row: Optional[pd.Series], metric: str) -> str:
    if row is None:
        return "N/A"
    mean_col = f"{metric}_mean"
    if mean_col not in row or pd.isna(row[mean_col]):
        return "N/A"
    return f"{row[mean_col]:.4f} ± {row[f'{metric}_std']:.4f}"


def _seed_count(row: Optional[pd.Series], metric: str) -> int:
    if row is None:
        return 0
    col = f"{metric}_count"
    if col not in row or pd.isna(row[col]):
        return 0
    return int(row[col])


def build_comparison(agg: pd.DataFrame, fi_variant: str, r_variant: str) -> List[dict]:
    out: List[dict] = []
    for spec in _SPECS:
        fi = _row_for(agg, "fi_edl", fi_variant, spec)
        rd = _row_for(agg, "r_edl", r_variant, spec)
        out.append(
            {
                "task": spec.task,
                "score": spec.score_label,
                "metric": spec.metric_label,
                "fi_edl": _format(fi, spec.metric),
                "r_edl": _format(rd, spec.metric),
                "fi_n": _seed_count(fi, spec.metric),
                "r_n": _seed_count(rd, spec.metric),
            }
        )
    return out


def _write_markdown(rows: List[dict], path: Path) -> None:
    lines = [
        "# FI-EDL vs R-EDL Same-Pipeline Comparison",
        "",
        "R-EDL: Chen et al., ICLR 2024. Matched protocol — identical backbone,",
        "epochs (200 MNIST / 100 CIFAR), batch (64), lr (1e-3 MNIST / 5e-4 CIFAR),",
        "augmentation, and seed set {0,1,2,3,4}. Only the loss function differs.",
        "",
        f"| Task | Score | Metric | {FI_EDL_LABEL} | {R_EDL_LABEL} |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['task']} | {r['score']} | {r['metric']} | {r['fi_edl']} | {r['r_edl']} |")
    lines += [
        "",
        "## Notes",
        "",
        "- OOD rows: positive class = OOD.",
        "- Misclassification rows: positive class = correct prediction.",
        "- R-EDL uses `lambda_prior = 0.1` (paper default) and no KL term.",
        "- `n` columns omitted from this table; see CSV for per-cell seed count.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs", help="Runs directory (default: runs)")
    parser.add_argument("--out-dir", default="results/redl_rebuttal")
    parser.add_argument("--fi-edl-variant", default="fi_edl",
                        help="method_variant tag for the main FI-EDL rows")
    parser.add_argument("--r-edl-variant", default="r_edl",
                        help="method_variant tag for the R-EDL baseline rows")
    args = parser.parse_args()

    df = load_runs(Path(args.runs))
    if df.empty:
        raise SystemExit(f"No metrics.jsonl rows under {args.runs}")

    agg = _aggregate(df)
    rows = build_comparison(agg, args.fi_edl_variant, args.r_edl_variant)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "redl_compact_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    md_path = out_dir / "redl_compact_comparison.md"
    _write_markdown(rows, md_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    missing = [r["task"] + "/" + r["metric"] for r in rows if r["fi_edl"] == "N/A" or r["r_edl"] == "N/A"]
    if missing:
        print(f"WARNING: {len(missing)} cell(s) missing data:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
