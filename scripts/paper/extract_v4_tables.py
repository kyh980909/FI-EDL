"""Extract v4 (old-cycle) Tables 1-5 from runs/oldcycle_{cifar10,mnist}/.

Reads ``metrics.jsonl`` files produced by ``src.eval`` and aggregates them
to mean ± sample std (n-1) per (method, dataset, metric). Then renders
LaTeX rows for each table, including bold (1st) and underline (2nd) markup.

Usage::

    python scripts/paper/extract_v4_tables.py \\
        --runs-cifar runs/oldcycle_cifar10 \\
        --runs-mnist runs/oldcycle_mnist \\
        --out results/v4_oldcycle/tables.txt

"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# ----- Method labels and ordering ------------------------------------------
METHOD_ORDER_T1_T2_T3 = [
    ("edl_l1", r"EDL~\citep{Sensoy2018}"),
    ("i_edl",  r"I-EDL~\citep{Deng2023}"),
    ("r_edl",  r"R-EDL~\citep{Chen2024a}"),
    ("re_edl", r"Re-EDL~\citep{Chen2024b}"),
    ("fi_edl", r"\textbf{FI-EDL (Ours)}"),
]

# Controller variants live under method=fi_edl with a method_variant suffix.
CONTROLLER_VARIANTS = [
    ("fi_edl_constant",     "Constant ($\\lambda=\\beta$)"),
    ("fi_edl_alpha0_gate",  "$\\alpha_0$ gate"),
    ("fi_edl_fim_nodetach", "FIM gate (no detach)"),
    ("fi_edl",              "\\textbf{FIM gate / FI-EDL (detach)}"),  # main run row
]

SENSITIVITY_CELLS = [(b, g) for b in (0.5, 1.0, 2.0) for g in (0.5, 1.0, 2.0)]


# ----- Metric specs (dataset, split, score_type, metric_key) ---------------
SPECS_CIFAR = {
    "ECE_cifar":     ("cifar10",  "conf_eval", "maxp",     "ece"),
    "Acc_cifar":     ("cifar10",  "conf_eval", "maxp",     "accuracy"),
    "MaxP_cifar":    ("cifar10",  "conf_eval", "maxp",     "aupr"),
    "MaxA_cifar":    ("cifar10",  "conf_eval", "maxalpha", "aupr"),
    "SVHN":          ("svhn",     "eval",      "alpha0",   "aupr"),
    "C100":          ("cifar100", "eval",      "alpha0",   "aupr"),
}
SPECS_MNIST = {
    "ECE_mnist":     ("mnist",    "conf_eval", "maxp",     "ece"),
    "Acc_mnist":     ("mnist",    "conf_eval", "maxp",     "accuracy"),
    "KMNIST":        ("kmnist",   "eval",      "alpha0",   "aupr"),
    "FMNIST":        ("fmnist",   "eval",      "alpha0",   "aupr"),
}


def collect(paths: Iterable[str], specs: Dict[str, Tuple[str, str, str, str]]
            ) -> Dict[str, Dict[int, float]]:
    """Map metric_key -> {seed: value}. Latest-eval-wins per (seed, key)."""
    out: Dict[str, Dict[int, float]] = {k: {} for k in specs}
    for path in sorted(paths):
        try:
            with open(path) as f:
                for line in f:
                    obj = json.loads(line)
                    seed = obj.get("seed")
                    ds, sp, st = obj.get("dataset"), obj.get("split"), obj.get("score_type")
                    metrics = obj.get("metrics", {}) or {}
                    for key, (d2, sp2, st2, m2) in specs.items():
                        if (ds, sp, st) == (d2, sp2, st2) and m2 in metrics:
                            out[key][seed] = float(metrics[m2])
        except Exception:
            pass
    return out


def fmt(values: Dict[int, float], scale: float = 100.0) -> Tuple[str, float, float]:
    """Return ('m.mm$\\pm$s.ss', mean*scale, std*scale) using sample std (n-1)."""
    if not values:
        return "TBD$\\pm$TBD", float("nan"), float("nan")
    vs = list(values.values())
    m = statistics.mean(vs) * scale
    s = (statistics.stdev(vs) if len(vs) > 1 else 0.0) * scale
    return f"{m:.2f}$\\pm${s:.2f}", m, s


# ----- Bold/underline by ranking -------------------------------------------
def rank_markup(rows: List[Tuple[str, float, float]],
                *, lower_is_better: bool) -> List[str]:
    """Given list of (cell_text, mean, std), return per-row markup ('bold',
    'underline', 'plain') based on mean ranking.

    Returns a list of cell_text with ``\\textbf{...}`` / ``\\underline{...}``
    wrappers applied where appropriate. NaN means are left untouched.
    """
    sortable = [(i, m) for i, (_, m, _) in enumerate(rows) if m == m]  # filter NaN
    sortable.sort(key=lambda t: t[1], reverse=not lower_is_better)
    out = [r[0] for r in rows]
    if sortable:
        first = sortable[0][0]
        out[first] = r"\textbf{" + out[first] + r"}"
        if len(sortable) > 1:
            second = sortable[1][0]
            out[second] = r"\underline{" + out[second] + r"}"
    return out


# ----- Table renderers -----------------------------------------------------
def render_table_main(runs_cifar: Path, runs_mnist: Path) -> Dict[str, str]:
    """Returns {'table1': str, 'table2': str, 'table3': str}."""
    rows_t1: List[Tuple[str, List[Tuple[str, float, float]]]] = []
    rows_t2: List[Tuple[str, List[Tuple[str, float, float]]]] = []
    rows_t3: List[Tuple[str, List[Tuple[str, float, float]]]] = []

    for method, label in METHOD_ORDER_T1_T2_T3:
        cifar_paths = glob.glob(str(runs_cifar / method / "seed_*" / "*" / "metrics.jsonl"))
        mnist_paths = glob.glob(str(runs_mnist / method / "seed_*" / "*" / "metrics.jsonl"))
        rc = collect(cifar_paths, SPECS_CIFAR)
        rm = collect(mnist_paths, SPECS_MNIST)
        rows_t1.append((label, [
            fmt(rm["ECE_mnist"]),
            fmt(rc["ECE_cifar"]),
            fmt(rc["Acc_cifar"]),
        ]))
        rows_t2.append((label, [
            fmt(rm["KMNIST"]),
            fmt(rm["FMNIST"]),
            fmt(rc["SVHN"]),
            fmt(rc["C100"]),
        ]))
        rows_t3.append((label, [
            fmt(rc["MaxP_cifar"]),
            fmt(rc["MaxA_cifar"]),
            fmt(rc["Acc_cifar"]),
        ]))

    def render(rows, lowers):
        # transpose, mark up, transpose back
        ncols = len(rows[0][1])
        cell_texts = [[c[0] for c in row[1]] for row in rows]
        means = [[c[1] for c in row[1]] for row in rows]
        stds  = [[c[2] for c in row[1]] for row in rows]
        marked = [list(r) for r in cell_texts]
        for j in range(ncols):
            col = [(cell_texts[i][j], means[i][j], stds[i][j]) for i in range(len(rows))]
            new = rank_markup(col, lower_is_better=lowers[j])
            for i in range(len(rows)):
                marked[i][j] = new[i]
        # build LaTeX rows; insert \midrule before FI-EDL row
        out_lines = []
        for i, (label, _) in enumerate(rows):
            if "FI-EDL" in label and i == len(rows) - 1:
                out_lines.append("\\midrule")
            out_lines.append(label + " & " + " & ".join(marked[i]) + r" \\")
        return "\n".join(out_lines)

    return {
        "table1": render(rows_t1, lowers=[True,  True,  False]),       # ECE↓ ECE↓ Acc↑
        "table2": render(rows_t2, lowers=[False, False, False, False]),# all AUPR↑
        "table3": render(rows_t3, lowers=[False, False, False]),       # all ↑
    }


def render_table_controller(runs_cifar: Path) -> str:
    rows: List[Tuple[str, List[Tuple[str, float, float]]]] = []
    for variant, label in CONTROLLER_VARIANTS:
        # method_variant filtering: eval dirs end with `_<variant>` for ablations,
        # and the main fi_edl run has no suffix (`fi_edl/seed_*/eval_*_cifar10_<bb>/`).
        if variant == "fi_edl":
            paths = [p for p in glob.glob(str(runs_cifar / "fi_edl" / "seed_*" / "*" / "metrics.jsonl"))
                     if re.search(r"eval_\d+T\d+_cifar10_[a-z0-9]+/metrics\.jsonl$", p)]
        else:
            paths = [p for p in glob.glob(str(runs_cifar / "fi_edl" / "seed_*" / "*" / "metrics.jsonl"))
                     if p.endswith(f"_{variant}/metrics.jsonl")]
        r = collect(paths, SPECS_CIFAR)
        rows.append((label, [
            fmt(r["ECE_cifar"]),
            fmt(r["Acc_cifar"]),
            fmt(r["SVHN"]),
            fmt(r["C100"]),
        ]))
    # ranking
    cell_texts = [[c[0] for c in row[1]] for row in rows]
    means = [[c[1] for c in row[1]] for row in rows]
    stds  = [[c[2] for c in row[1]] for row in rows]
    marked = [list(r) for r in cell_texts]
    lowers = [True, False, False, False]  # ECE↓, Acc↑, SVHN↑, C100↑
    for j in range(4):
        col = [(cell_texts[i][j], means[i][j], stds[i][j]) for i in range(len(rows))]
        new = rank_markup(col, lower_is_better=lowers[j])
        for i in range(len(rows)):
            marked[i][j] = new[i]
    out_lines = []
    for label, _ in rows:
        i = next(idx for idx, (lab, _) in enumerate(rows) if lab == label)
        out_lines.append(label + " & " + " & ".join(marked[i]) + r" \\")
    return "\n".join(out_lines)


def render_table_sensitivity(runs_cifar: Path) -> str:
    rows: List[Tuple[Tuple[float, float], List[Tuple[str, float, float]]]] = []
    for b, g in SENSITIVITY_CELLS:
        suffix = f"cifar_bg_b{b}_g{g}"
        paths = [p for p in glob.glob(str(runs_cifar / "fi_edl" / "seed_*" / "*" / "metrics.jsonl"))
                 if p.endswith(f"_{suffix}/metrics.jsonl")]
        r = collect(paths, SPECS_CIFAR)
        rows.append(((b, g), [
            fmt(r["ECE_cifar"]),
            fmt(r["SVHN"]),
            fmt(r["C100"]),
        ]))
    cell_texts = [[c[0] for c in row[1]] for row in rows]
    means = [[c[1] for c in row[1]] for row in rows]
    stds  = [[c[2] for c in row[1]] for row in rows]
    marked = [list(r) for r in cell_texts]
    lowers = [True, False, False]  # ECE↓, SVHN↑, C100↑
    for j in range(3):
        col = [(cell_texts[i][j], means[i][j], stds[i][j]) for i in range(len(rows))]
        new = rank_markup(col, lower_is_better=lowers[j])
        for i in range(len(rows)):
            marked[i][j] = new[i]
    out_lines = []
    for i, ((b, g), _) in enumerate(rows):
        beta_str = f"\\textbf{{{b}}}" if (b, g) == (1.0, 1.0) else f"{b}"
        gamma_str = f"\\textbf{{{g}}}" if (b, g) == (1.0, 1.0) else f"{g}"
        out_lines.append(f"{beta_str} & {gamma_str} & " + " & ".join(marked[i]) + r" \\")
    return "\n".join(out_lines)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-cifar", type=Path,
                        default=Path("runs/oldcycle_cifar10"))
    parser.add_argument("--runs-mnist", type=Path,
                        default=Path("runs/oldcycle_mnist"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/v4_oldcycle/tables.txt"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    main_tables = render_table_main(args.runs_cifar, args.runs_mnist)
    controller = render_table_controller(args.runs_cifar)
    sensitivity = render_table_sensitivity(args.runs_cifar)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("=== Table 1 — Calibration (ECE) ===\n")
        f.write(main_tables["table1"] + "\n\n")
        f.write("=== Table 2 — OOD AUPR ===\n")
        f.write(main_tables["table2"] + "\n\n")
        f.write("=== Table 3 — Misclassification AUPR + Acc ===\n")
        f.write(main_tables["table3"] + "\n\n")
        f.write("=== Table 4 — Controller ablation ===\n")
        f.write(controller + "\n\n")
        f.write("=== Table 5 — (β,γ) sensitivity ===\n")
        f.write(sensitivity + "\n")

    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
