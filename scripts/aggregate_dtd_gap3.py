"""Gap 3 DTD (Textures) OOD result aggregation.

Pulls per-seed AUROC@best-score from runs_reeval/cifar10_dtd/ for FI-EDL+SN,
DAEDL, F-EDL, Re-EDL on the DTD OOD task, computes paired stats, and writes
results/dtd_summary.md.

Usage:
    uv run python scripts/aggregate_dtd_gap3.py

Run after scripts/eval_gap3_dtd.sh completes.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent

# Per-method best-OOD-score convention (same as VGG-16 analysis)
BEST_SCORE = {
    "FI-EDL+SN": ("fi_edl", "alpha0"),
    "DAEDL":     ("daedl",  "alpha0"),
    "F-EDL":     ("f_edl",  "maxp"),
    "Re-EDL":    ("re_edl", "alpha0"),
}


def collect_auroc(method_dir: str, score: str, dataset: str = "dtd"):
    """Return {seed: AUROC} from runs_reeval/cifar10_dtd/."""
    out = {}
    pat = f"runs_reeval/cifar10_dtd/{method_dir}/seed_*/eval_*/metrics.jsonl"
    for f in glob.glob(str(REPO / pat)):
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        seed = rows[0].get("seed")
        for r in rows:
            if r.get("split") != "eval":
                continue
            if r.get("dataset") != dataset:
                continue
            if r.get("score_type") != score:
                continue
            v = r["metrics"].get("auroc")
            if v is not None:
                out.setdefault(seed, v)
                break
    return out


def paired(ours, base):
    seeds = sorted(set(ours.keys()) & set(base.keys()))
    if len(seeds) < 3:
        return None
    a = np.array([ours[s] for s in seeds])
    b = np.array([base[s] for s in seeds])
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    rng = np.random.default_rng(0)
    boots = np.array(
        [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)]
    )
    ci = np.percentile(boots, [2.5, 97.5])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    verdict = "WIN" if diff.mean() > 0 and p < 0.05 else (
        "loss" if diff.mean() < 0 and p < 0.05 else "tie")
    return dict(n=len(seeds), ours_mean=a.mean(), base_mean=b.mean(),
                diff_mean=diff.mean(), diff_std=diff.std(ddof=1),
                t=t, p=p, ci=tuple(ci), sig=sig, verdict=verdict)


def main():
    values = {label: collect_auroc(mdir, score) for label, (mdir, score) in BEST_SCORE.items()}

    md = ["# Gap 3 — DTD (Textures) OOD AUROC for CIFAR-10", ""]
    md.append("Source: `runs_reeval/cifar10_dtd/*/seed_*/eval_*/metrics.jsonl`")
    md.append("")
    md.append("## Headline numbers (5-seed mean ± std @ method's best score)")
    md.append("")
    md.append("| Method | Score | AUROC | n |")
    md.append("|---|---|---|---|")
    for label, (_, sc) in BEST_SCORE.items():
        v = values[label]
        if not v:
            md.append(f"| {label} | {sc} | — | 0 |")
            continue
        a = np.array(list(v.values())) * 100
        m = a.mean()
        s = a.std(ddof=1) if len(a) > 1 else 0.0
        md.append(f"| {label} | {sc} | {m:.2f} ± {s:.2f} | {len(a)} |")
    md.append("")

    md.append("## Paired tests: FI-EDL+SN vs each baseline")
    md.append("")
    md.append("| baseline | base score | n | Δ (ours − base) | t | p | 95% CI | sig | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    ours = values["FI-EDL+SN"]
    for bname in ["DAEDL", "F-EDL", "Re-EDL"]:
        base = values[bname]
        r = paired(ours, base)
        if r is None:
            md.append(f"| {bname} | {BEST_SCORE[bname][1]} | <3 | — | — | — | — | — | — |")
            continue
        md.append(
            f"| {bname} | {BEST_SCORE[bname][1]} | {r['n']} | "
            f"{r['diff_mean']*100:+.2f} ± {r['diff_std']*100:.2f} | "
            f"{r['t']:+.2f} | {r['p']:.4f} | "
            f"[{r['ci'][0]*100:+.2f}, {r['ci'][1]*100:+.2f}] | "
            f"{r['sig']} | **{r['verdict']}** |"
        )
    md.append("")
    md.append("DTD is a Hendrycks-standard texture OOD set; high AUROC indicates "
              "the recipe transfers to a new OOD distribution beyond SVHN/CIFAR-100.")

    out_path = REPO / "results" / "dtd_summary.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
