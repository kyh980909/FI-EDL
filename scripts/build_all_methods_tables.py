"""All-methods aggregated tables — every trained model in one place.

Loads `runs/`, aggregates mean ± std across seeds per
(method, method_variant, dataset, split, score_type), and emits:

  - all_methods_long.csv        — every (method, dataset, metric) cell
  - all_methods_<dataset>.md    — reviewer-ready wide tables (rows = method)

By default the beta/gamma sensitivity sweep (variants matching
``cifar_bg_b*_g*``) is excluded from the headline tables; pass
``--include-bg`` to keep them.

Usage::

    uv run python scripts/build_all_methods_tables.py
    uv run python scripts/build_all_methods_tables.py --runs runs --out-dir results/all_methods
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from scripts._loader import load_runs


_BG_VARIANT_RE = re.compile(r"^cifar_bg_b[\d.]+_g[\d.]+$")

# (method, method_variant) -> display label, sort priority.
_METHOD_LABELS: Dict[Tuple[str, str], Tuple[str, int]] = {
    ("edl_fixed", "edl_fixed_l1"):       ("EDL (λ=1.0)",          10),
    ("edl_fixed", "edl_fixed_l01"):      ("EDL (λ=0.1)",          11),
    ("edl_fixed", "edl_fixed_l0001"):    ("EDL (λ=0.001)",        12),
    ("i_edl", "i_edl"):                  ("I-EDL",                20),
    ("r_edl", "r_edl"):                  ("R-EDL",                30),
    ("fi_edl", "fi_edl"):                ("FI-EDL (ours)",        40),
    ("fi_edl", "fi_edl_constant"):       ("FI-EDL — const gate",  50),
    ("fi_edl", "fi_edl_alpha0_gate"):    ("FI-EDL — α₀ info",     51),
    ("fi_edl", "fi_edl_fim_nodetach"):   ("FI-EDL — no detach",   52),
}


def _label_for(method: str, variant: str) -> Tuple[str, int]:
    if (method, variant) in _METHOD_LABELS:
        return _METHOD_LABELS[(method, variant)]
    if variant and variant != method:
        return (f"{method}/{variant}", 90)
    return (method, 90)


@dataclass(frozen=True)
class TableSpec:
    title: str
    dataset: str
    split: str
    columns: List[Tuple[str, str, str]]  # (col_label, score_type, metric)


def _table_specs() -> List[TableSpec]:
    return [
        TableSpec(
            "OOD Detection — CIFAR10 → SVHN", "svhn", "eval",
            [("AUPR (α₀)", "alpha0", "aupr"),
             ("AUROC (α₀)", "alpha0", "auroc"),
             ("FPR95 (α₀)", "alpha0", "fpr95")],
        ),
        TableSpec(
            "OOD Detection — CIFAR10 → CIFAR100", "cifar100", "eval",
            [("AUPR (α₀)", "alpha0", "aupr"),
             ("AUROC (α₀)", "alpha0", "auroc"),
             ("FPR95 (α₀)", "alpha0", "fpr95")],
        ),
        TableSpec(
            "Misclassification — CIFAR10", "cifar10", "conf_eval",
            [("AUPR (Max.P)", "maxp", "aupr"),
             ("AUROC (Max.P)", "maxp", "auroc"),
             ("FPR95 (Max.P)", "maxp", "fpr95"),
             ("AUPR (Max.α)", "maxalpha", "aupr"),
             ("AUROC (Max.α)", "maxalpha", "auroc")],
        ),
        TableSpec(
            "Calibration — CIFAR10", "cifar10", "conf_eval",
            [("ECE (Max.P)", "maxp", "ece"),
             ("Acc (Max.P)", "maxp", "accuracy"),
             ("NLL (Max.P)", "maxp", "nll")],
        ),
        TableSpec(
            "OOD Detection — MNIST → KMNIST", "kmnist", "eval",
            [("AUPR (α₀)", "alpha0", "aupr"),
             ("AUROC (α₀)", "alpha0", "auroc"),
             ("FPR95 (α₀)", "alpha0", "fpr95")],
        ),
        TableSpec(
            "OOD Detection — MNIST → FMNIST", "fmnist", "eval",
            [("AUPR (α₀)", "alpha0", "aupr"),
             ("AUROC (α₀)", "alpha0", "auroc"),
             ("FPR95 (α₀)", "alpha0", "fpr95")],
        ),
        TableSpec(
            "Misclassification — MNIST", "mnist", "conf_eval",
            [("AUPR (Max.P)", "maxp", "aupr"),
             ("AUROC (Max.P)", "maxp", "auroc"),
             ("FPR95 (Max.P)", "maxp", "fpr95"),
             ("AUPR (Max.α)", "maxalpha", "aupr")],
        ),
        TableSpec(
            "Calibration — MNIST", "mnist", "conf_eval",
            [("ECE (Max.P)", "maxp", "ece"),
             ("Acc (Max.P)", "maxp", "accuracy"),
             ("NLL (Max.P)", "maxp", "nll")],
        ),
    ]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "method_variant", "dataset", "split", "score_type"]
    metric_cols = [c for c in ["aupr", "auroc", "fpr95", "ece", "accuracy", "nll", "aurc"] if c in df.columns]
    grouped = df.groupby(keys)[metric_cols].agg(["mean", "std", "count"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()


def _format_cell(row: Optional[pd.Series], metric: str) -> str:
    if row is None:
        return "—"
    mean = row.get(f"{metric}_mean")
    if mean is None or pd.isna(mean):
        return "—"
    std = row.get(f"{metric}_std")
    std_txt = f"{std:.4f}" if pd.notna(std) else "—"
    n = row.get(f"{metric}_count")
    n_txt = f" (n={int(n)})" if pd.notna(n) else ""
    return f"{mean:.4f} ± {std_txt}{n_txt}"


def _row_for(agg: pd.DataFrame, method: str, variant: str,
             dataset: str, split: str, score_type: str) -> Optional[pd.Series]:
    mask = (
        (agg["method"] == method)
        & (agg["method_variant"] == variant)
        & (agg["dataset"] == dataset)
        & (agg["split"] == split)
        & (agg["score_type"] == score_type)
    )
    sub = agg[mask]
    return None if sub.empty else sub.iloc[0]


def _build_table(agg: pd.DataFrame, spec: TableSpec,
                 method_keys: List[Tuple[str, str]]) -> List[str]:
    header = "| Method | " + " | ".join(c[0] for c in spec.columns) + " |"
    sep = "|---" + "|---:" * len(spec.columns) + "|"
    lines = [f"### {spec.title}", "", header, sep]
    for method, variant in method_keys:
        label, _ = _label_for(method, variant)
        cells = [
            _format_cell(_row_for(agg, method, variant, spec.dataset, spec.split, st), m)
            for (_, st, m) in spec.columns
        ]
        if all(c == "—" for c in cells):
            continue
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _present_method_keys(agg: pd.DataFrame, include_bg: bool) -> List[Tuple[str, str]]:
    pairs = agg[["method", "method_variant"]].drop_duplicates()
    keys: List[Tuple[str, str]] = []
    for _, r in pairs.iterrows():
        m, v = r["method"], r["method_variant"] or ""
        if not include_bg and _BG_VARIANT_RE.match(v):
            continue
        keys.append((m, v))
    keys.sort(key=lambda mv: (_label_for(*mv)[1], _label_for(*mv)[0]))
    return keys


def _write_long_csv(agg: pd.DataFrame, out: Path) -> None:
    df = agg.copy()
    df["method_label"] = [_label_for(m, v)[0] for m, v in zip(df["method"], df["method_variant"])]
    cols = ["method_label"] + [c for c in df.columns if c != "method_label"]
    df = df[cols].sort_values(["dataset", "split", "score_type", "method", "method_variant"])
    df.to_csv(out, index=False)


def _write_dataset_md(agg: pd.DataFrame, dataset_root: str,
                      method_keys: List[Tuple[str, str]], out: Path) -> None:
    lines = [f"# All-Methods Comparison — {dataset_root.upper()}", ""]
    for spec in _table_specs():
        spec_root = "mnist" if spec.dataset in {"mnist", "kmnist", "fmnist"} else "cifar10"
        if spec_root != dataset_root:
            continue
        lines += _build_table(agg, spec, method_keys)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out-dir", default="results/all_methods")
    parser.add_argument("--include-bg", action="store_true",
                        help="Include the cifar_bg_b*_g* sensitivity-sweep variants in the tables.")
    args = parser.parse_args()

    df = load_runs(Path(args.runs))
    if df.empty:
        raise SystemExit(f"No metrics.jsonl rows under {args.runs}")
    df["method_variant"] = df["method_variant"].fillna("")

    agg = _aggregate(df)
    method_keys = _present_method_keys(agg, include_bg=args.include_bg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_csv = out_dir / "all_methods_long.csv"
    _write_long_csv(agg, long_csv)
    print(f"Wrote {long_csv}")

    for root in ("cifar10", "mnist"):
        md = out_dir / f"all_methods_{root}.md"
        _write_dataset_md(agg, root, method_keys, md)
        print(f"Wrote {md}")

    print("\nMethods detected:")
    for m, v in method_keys:
        label, _ = _label_for(m, v)
        print(f"  - {label}   ({m}/{v or '∅'})")


if __name__ == "__main__":
    main()
