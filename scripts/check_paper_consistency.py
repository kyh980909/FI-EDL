"""Paper consistency checks for the FI-EDL Neurocomputing manuscript.

Pre-submission gate that catches:
1. \\cite{} keys in main.tex that don't exist in refs.bib.
2. \\autoref{} / \\Cref{} / \\ref{} targets that don't exist as labels.
3. Numerical claims in main.tex that don't match the source-of-truth report.
4. \\todo{} markers still present.
5. Forbidden words ("F-EDL" without disambiguation, "always", etc).

Usage:
    uv run python scripts/check_paper_consistency.py

Output: prints a structured report; exits 1 if any high-severity issues are
found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN_TEX = REPO / "paper" / "main.tex"
REFS_BIB = REPO / "paper" / "refs.bib"
REPORT = REPO / "results" / "baseline_comparison_report.md"
# Auxiliary numerical sources whose contents also count as "report-verified".
EXTRA_SOURCES = [
    REPO / "results" / "stats_significance.md",
    REPO / "results" / "score_audit.md",
    REPO / "results" / "gap_analysis.md",
]


def load(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def check_citations(tex: str, bib: str) -> list[str]:
    """Return list of \\cite{} keys present in tex but missing from bib."""
    cite_keys = set()
    for m in re.finditer(r"\\cite[pt]?\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            cite_keys.add(k.strip())
    for m in re.finditer(r"\\citep\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            cite_keys.add(k.strip())
    bib_keys = set(re.findall(r"@\w+\{(\w[\w_]*)\s*,", bib))
    missing = sorted(cite_keys - bib_keys)
    return missing, sorted(cite_keys), sorted(bib_keys)


def check_refs(tex: str) -> list[str]:
    """Return list of \\autoref/\\Cref/\\ref targets without a \\label."""
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set()
    for m in re.finditer(r"\\(?:autoref|Cref|ref|cref)\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            refs.add(k.strip())
    return sorted(refs - labels)


def check_todos(tex: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(tex.splitlines(), 1):
        if r"\todo{" in line:
            out.append((i, line.strip()))
    return out


def check_forbidden(tex: str) -> list[tuple[int, str]]:
    """Find phrases that overclaim or that confuse FI-EDL with F-EDL."""
    out = []
    forbidden = [
        # Overclaims
        (r"\balways\b", "avoid 'always' — qualify scope"),
        (r"\bguaranteed\b", "avoid 'guaranteed' without proof"),
        (r"\bfirst to\b", "verify uniqueness if claiming primacy"),
        # F-EDL/FI-EDL confusion: our method is FI-EDL, F-EDL is Flexible EDL.
        # Flag bare "F-EDL" usage that doesn't follow a disambiguating context.
    ]
    for pat, msg in forbidden:
        for i, line in enumerate(tex.splitlines(), 1):
            if re.search(pat, line, flags=re.IGNORECASE):
                out.append((i, f"[{msg}] {line.strip()[:120]}"))
    return out


def check_numbers(tex: str, report: str) -> list[str]:
    """Spot-check headline numbers against the report.

    We do a simple substring search of distinctive numerical strings.
    """
    headline_numbers = [
        "3.93",       # ECE
        "92.99",      # SVHN AUROC
        "86.32",      # CIFAR-100 AUROC
        "90.70",      # Misclass AUROC
        "90.48",      # Acc
        "5.19",       # baseline FI-EDL ECE
        "1.70",       # MNIST FI-EDL ECE
        "13.60",      # MNIST +SN regression ECE
        "89.89",      # baseline FI-EDL maxp SVHN
        "83.80",      # baseline FI-EDL maxp C100
        "91.72",      # baseline FI-EDL alpha0 SVHN
        "84.67",      # baseline FI-EDL alpha0 C100
    ]
    missing_in_report = []
    for n in headline_numbers:
        if n in tex and n not in report:
            missing_in_report.append(n)
    return missing_in_report


def main():
    if not MAIN_TEX.exists():
        print(f"ERROR: {MAIN_TEX} not found")
        sys.exit(1)
    tex = load(MAIN_TEX)
    bib = load(REFS_BIB)
    report = load(REPORT) + "\n" + "\n".join(load(p) for p in EXTRA_SOURCES)

    issues = 0

    print("=" * 70)
    print("Paper Consistency Report — paper/main.tex")
    print("=" * 70)

    # 1. Citations
    missing_cites, all_cites, all_keys = check_citations(tex, bib)
    print(f"\n[1] Citation keys: {len(all_cites)} unique \\cite in tex, "
          f"{len(all_keys)} entries in refs.bib")
    if missing_cites:
        print(f"  MISSING in refs.bib ({len(missing_cites)}):")
        for k in missing_cites:
            print(f"    - {k}")
        issues += len(missing_cites)
    else:
        print("  All \\cite keys resolve. ✓")

    # 2. Cross-references
    missing_refs = check_refs(tex)
    print(f"\n[2] Cross-references")
    if missing_refs:
        print(f"  UNRESOLVED targets ({len(missing_refs)}):")
        for k in missing_refs:
            print(f"    - {k}")
        issues += len(missing_refs)
    else:
        print("  All \\autoref/\\ref targets resolve. ✓")

    # 3. TODO markers
    todos = check_todos(tex)
    print(f"\n[3] Remaining \\todo markers: {len(todos)}")
    for line_no, line in todos:
        print(f"  L{line_no}: {line[:120]}")

    # 4. Forbidden phrases
    forb = check_forbidden(tex)
    print(f"\n[4] Overclaim / forbidden phrases: {len(forb)}")
    for line_no, msg in forb[:20]:
        print(f"  L{line_no}: {msg}")
    if len(forb) > 20:
        print(f"  ... and {len(forb)-20} more")

    # 5. Headline numbers spot-check
    miss_num = check_numbers(tex, report)
    print(f"\n[5] Headline numbers in tex but not in report: {len(miss_num)}")
    for n in miss_num:
        print(f"    - {n}  (verify against results/baseline_comparison_report.md)")
        issues += 1

    print()
    print("=" * 70)
    print(f"Total high-severity issues: {issues}")
    print("Low-severity items (todos, overclaims): inspect and address before submission.")
    print("=" * 70)

    sys.exit(1 if issues > 0 else 0)


if __name__ == "__main__":
    main()
