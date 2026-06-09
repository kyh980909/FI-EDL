"""Gap 1 ResNet-18 CIFAR-10 result aggregation.

Pulls per-seed metrics for FI-EDL+SN, DAEDL, F-EDL, Re-EDL with ResNet-18
backbone, computes paired t-test + bootstrap 95% CI of FI-EDL+SN vs each
baseline, and writes results/resnet_summary.md.

Usage:
    uv run python scripts/aggregate_resnet_gap1.py

Run after scripts/run_resnet_cifar10.sh completes (20 train + 20 eval).
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent


def collect(method_dir: str, variant_re_pat: str, score: str, split: str,
            dataset: str, metric_key: str) -> dict[int, float]:
    """Return {seed: value} from eval JSONL for runs whose dir matches the regex."""
    out: dict[int, float] = {}
    pat = re.compile(variant_re_pat)
    glob_pat = f"runs/cifar10/{method_dir}/seed_*/eval_*/metrics.jsonl"
    for f in glob.glob(str(REPO / glob_pat)):
        if not pat.search(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        seed = rows[0].get("seed")
        for r in rows:
            if r.get("split") != split:
                continue
            if r.get("dataset") != dataset:
                continue
            if r.get("score_type") != score:
                continue
            v = r["metrics"].get(metric_key)
            if v is not None:
                out.setdefault(seed, v)
                break
    return out


def paired_stats(ours: dict[int, float], base: dict[int, float],
                 lower_better: bool = False):
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
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    if lower_better:
        verdict = "WIN" if diff.mean() < 0 and p < 0.05 else (
            "loss" if diff.mean() > 0 and p < 0.05 else "tie")
    else:
        verdict = "WIN" if diff.mean() > 0 and p < 0.05 else (
            "loss" if diff.mean() < 0 and p < 0.05 else "tie")
    return dict(
        n=len(seeds), ours_mean=a.mean(), base_mean=b.mean(),
        diff_mean=diff.mean(), diff_std=diff.std(ddof=1),
        t=t, p=p, ci=(ci_lo, ci_hi), sig=sig, verdict=verdict,
    )


def main():
    # Variants:
    # FI-EDL+SN-resnet: method_variant=fi_edl_sn_resnet (Phase 1 of sweep)
    # DAEDL/F-EDL/Re-EDL on resnet: default variant, but backbone=resnet18 means
    # train_*_cifar10_resnet18 path
    METHOD_VAR = {
        "FI-EDL+SN": ("fi_edl", r"_fi_edl_sn_resnet/"),
        "DAEDL":     ("daedl",   r"_cifar10_resnet18/"),
        "F-EDL":     ("f_edl",   r"_cifar10_resnet18/"),
        "Re-EDL":    ("re_edl",  r"_cifar10_resnet18/"),
    }

    # Best-score per method (carried over from VGG-16 analysis; verify in summary)
    BEST_OOD_SCORE = {
        "FI-EDL+SN": "alpha0",
        "DAEDL":     "alpha0",
        "F-EDL":     "maxp",
        "Re-EDL":    "alpha0",
    }

    # Pull values
    values = {}
    for label, (mdir, vre) in METHOD_VAR.items():
        ece = collect(mdir, vre, "maxp", "conf_eval", "cifar10", "ece")
        acc = collect(mdir, vre, "maxp", "conf_eval", "cifar10", "accuracy")
        misc = collect(mdir, vre, "maxp", "conf_eval", "cifar10", "auroc")
        svhn_score = BEST_OOD_SCORE[label]
        c100_score = BEST_OOD_SCORE[label]
        svhn = collect(mdir, vre, svhn_score, "eval", "svhn", "auroc")
        c100 = collect(mdir, vre, c100_score, "eval", "cifar100", "auroc")
        values[label] = dict(acc=acc, ece=ece, svhn=svhn, c100=c100, misc=misc,
                              svhn_score=svhn_score, c100_score=c100_score)

    def fmt_seeds(d, pct=True, places=2):
        if not d:
            return "—"
        a = np.array(list(d.values())) * (100 if pct else 1)
        if len(a) < 2:
            return f"{a[0]:.{places}f} (n={len(a)})"
        return f"{a.mean():.{places}f} ± {a.std(ddof=1):.{places}f} (n={len(a)})"

    md = ["# Gap 1 — ResNet-18 CIFAR-10 Results", ""]
    md.append("Source: `runs/cifar10/{fi_edl,daedl,f_edl,re_edl}/seed_*/eval_*/metrics.jsonl`")
    md.append("")
    md.append("## Headline numbers (5-seed mean ± std)")
    md.append("")
    md.append("| Method | Acc ↑ | ECE↓ | SVHN AUROC ↑ | CIFAR-100 AUROC ↑ | Misclass AUROC ↑ |")
    md.append("|---|---|---|---|---|---|")
    for label in METHOD_VAR.keys():
        v = values[label]
        md.append(
            f"| {label} | {fmt_seeds(v['acc'])} | {fmt_seeds(v['ece'])} | "
            f"{fmt_seeds(v['svhn'])} ({v['svhn_score']}) | "
            f"{fmt_seeds(v['c100'])} ({v['c100_score']}) | "
            f"{fmt_seeds(v['misc'])} |"
        )
    md.append("")

    md.append("## Paired tests: FI-EDL+SN-resnet vs each baseline-resnet")
    md.append("")
    md.append("Significance: \\* p<0.05  \\*\\* p<0.01  \\*\\*\\* p<0.001  ns = not significant.")
    md.append("")
    ours = values["FI-EDL+SN"]
    for metric_label, key, lower_better in [
        ("CIFAR-10 ECE↓", "ece", True),
        ("SVHN AUROC↑", "svhn", False),
        ("CIFAR-100 AUROC↑", "c100", False),
        ("Misclass AUROC↑", "misc", False),
        ("Accuracy↑", "acc", False),
    ]:
        md.append(f"### {metric_label}")
        md.append("")
        md.append("| baseline | base score | n | Δ (ours − base) | t | p | 95% CI | sig | verdict |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for bname in ["DAEDL", "F-EDL", "Re-EDL"]:
            b = values[bname]
            sc = b["svhn_score"] if key == "svhn" else (
                b["c100_score"] if key == "c100" else "maxp")
            r = paired_stats(ours[key], b[key], lower_better=lower_better)
            if r is None:
                md.append(f"| {bname} | {sc} | <3 | — | — | — | — | — | — |")
                continue
            md.append(
                f"| {bname} | {sc} | {r['n']} | "
                f"{r['diff_mean']*100:+.2f} ± {r['diff_std']*100:.2f} | "
                f"{r['t']:+.2f} | {r['p']:.4f} | "
                f"[{r['ci'][0]*100:+.2f}, {r['ci'][1]*100:+.2f}] | "
                f"{r['sig']} | **{r['verdict']}** |"
            )
        md.append("")

    md.append("## Comparison vs VGG-16 (already in report §6.6.5/§6.6.7)")
    md.append("")
    md.append(
        "If recipe generalizes, FI-EDL+SN on ResNet-18 should still match or "
        "beat DAEDL/F-EDL/Re-EDL on ResNet-18 across CIFAR-100 AUROC (the "
        "uniquely-significant SOTA from VGG-16) and stay tied with DAEDL on "
        "ECE / Re-EDL on SVHN. Bigger gap or smaller gap matters for §5 main "
        "results and §7 Discussion."
    )
    md.append("")

    out_path = REPO / "results" / "resnet_summary.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
