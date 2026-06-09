"""CIFAR-100 ResNet-18 sweep result aggregation.

Run after scripts/run_cifar100.sh completes (15 train + 15 eval).
Writes results/cifar100_summary.md with per-method headline numbers and
paired t-tests of FI-EDL+SN vs each baseline.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent


def collect(method_dir: str, variant_re_pat: str, score: str, split: str,
            dataset: str, metric_key: str) -> dict[int, float]:
    out: dict[int, float] = {}
    pat = re.compile(variant_re_pat)
    for f in glob.glob(str(REPO / f"runs/cifar100/{method_dir}/seed_*/eval_*/metrics.jsonl")):
        if not pat.search(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        seed = rows[0].get("seed")
        for r in rows:
            if r.get("split") != split: continue
            if r.get("dataset") != dataset: continue
            if r.get("score_type") != score: continue
            v = r["metrics"].get(metric_key)
            if v is not None:
                out.setdefault(seed, v); break
    return out


def paired(ours, base, lower_better=False):
    seeds = sorted(set(ours.keys()) & set(base.keys()))
    if len(seeds) < 3: return None
    a = np.array([ours[s] for s in seeds])
    b = np.array([base[s] for s in seeds])
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)])
    ci = np.percentile(boots, [2.5, 97.5])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    if lower_better:
        verdict = "WIN" if diff.mean() < 0 and p < 0.05 else ("loss" if diff.mean() > 0 and p < 0.05 else "tie")
    else:
        verdict = "WIN" if diff.mean() > 0 and p < 0.05 else ("loss" if diff.mean() < 0 and p < 0.05 else "tie")
    return dict(n=len(seeds), ours_mean=a.mean(), base_mean=b.mean(),
                diff_mean=diff.mean(), diff_std=diff.std(ddof=1),
                t=t, p=p, ci=tuple(ci), sig=sig, verdict=verdict)


def main():
    METHOD_VAR = {
        "FI-EDL+SN": ("fi_edl", r"_fi_edl_sn/"),
        "DAEDL":     ("daedl",   r"_cifar100_resnet18/"),
        "F-EDL":     ("f_edl",   r"_cifar100_resnet18/"),
        "Re-EDL":    ("re_edl",  r"_cifar100_resnet18/"),
    }
    # Best-OOD-score conventions carried from CIFAR-10 analysis;
    # may differ on CIFAR-100 — aggregation script reports both maxp and alpha0.
    OOD_DATASETS = ["svhn", "cifar10"]  # +tinyimagenet via forward-only later

    values = {}
    for label, (mdir, vre) in METHOD_VAR.items():
        ece  = collect(mdir, vre, "maxp", "conf_eval", "cifar100", "ece")
        acc  = collect(mdir, vre, "maxp", "conf_eval", "cifar100", "accuracy")
        misc = collect(mdir, vre, "maxp", "conf_eval", "cifar100", "auroc")
        ood = {}
        for od in OOD_DATASETS:
            ood[(od, "maxp")]   = collect(mdir, vre, "maxp",   "eval", od, "auroc")
            ood[(od, "alpha0")] = collect(mdir, vre, "alpha0", "eval", od, "auroc")
        values[label] = dict(acc=acc, ece=ece, misc=misc, ood=ood)

    def fmt_seeds(d, pct=True, places=2):
        if not d: return "—"
        a = np.array(list(d.values())) * (100 if pct else 1)
        if len(a) < 2: return f"{a[0]:.{places}f} (n={len(a)})"
        return f"{a.mean():.{places}f} ± {a.std(ddof=1):.{places}f} (n={len(a)})"

    md = ["# CIFAR-100 — ResNet-18 Results", ""]
    md.append("Source: `runs/cifar100/{fi_edl,daedl,f_edl,re_edl}/seed_*/eval_*/metrics.jsonl`")
    md.append("Backbone: ResNet-18 (matching F-EDL Appendix E.2).")
    md.append("")
    md.append("## Headline numbers (5-seed mean ± std)")
    md.append("")
    md.append("| Method | Acc ↑ | ECE↓ | SVHN AUROC (best score) | CIFAR-10 AUROC (best score) | Misclass AUROC@maxp ↑ |")
    md.append("|---|---|---|---|---|---|")
    for label in METHOD_VAR:
        v = values[label]
        # pick best of maxp/alpha0 per OOD set
        def best_ood(od_name):
            mp = v["ood"].get((od_name, "maxp"), {})
            a0 = v["ood"].get((od_name, "alpha0"), {})
            mp_mean = (np.array(list(mp.values())).mean() if mp else 0)
            a0_mean = (np.array(list(a0.values())).mean() if a0 else 0)
            if a0_mean >= mp_mean: return a0, "alpha0"
            return mp, "maxp"
        svhn_d, svhn_sc = best_ood("svhn")
        c10_d, c10_sc = best_ood("cifar10")
        md.append(
            f"| {label} | {fmt_seeds(v['acc'])} | {fmt_seeds(v['ece'])} | "
            f"{fmt_seeds(svhn_d)} ({svhn_sc}) | "
            f"{fmt_seeds(c10_d)} ({c10_sc}) | "
            f"{fmt_seeds(v['misc'])} |"
        )
    md.append("")

    md.append("## Paired tests: FI-EDL+SN vs each baseline (each at their headline-best score)")
    md.append("")
    md.append("Significance: \\* p<0.05  \\*\\* p<0.01  \\*\\*\\* p<0.001  ns = not significant.")
    md.append("")
    ours = values["FI-EDL+SN"]

    def best_for(label, od):
        v = values[label]
        mp = v["ood"].get((od, "maxp"), {})
        a0 = v["ood"].get((od, "alpha0"), {})
        if not mp and not a0: return {}, "—"
        mp_mean = (np.array(list(mp.values())).mean() if mp else -1)
        a0_mean = (np.array(list(a0.values())).mean() if a0 else -1)
        if a0_mean >= mp_mean: return a0, "alpha0"
        return mp, "maxp"

    for title, key, lower_better in [
        ("CIFAR-100 ECE↓",    "ece",  True),
        ("SVHN AUROC↑",       "svhn", False),
        ("CIFAR-10 AUROC↑",   "cifar10", False),
        ("Misclass AUROC↑",   "misc", False),
    ]:
        md.append(f"### {title}")
        md.append("")
        md.append("| baseline | base score | n | Δ (ours − base) | t | p | 95% CI | sig | verdict |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        ours_d = ours[key] if key in ("ece","acc","misc") else best_for("FI-EDL+SN", key)[0]
        for bname in ["DAEDL", "F-EDL", "Re-EDL"]:
            if key in ("ece","acc","misc"):
                base_d, sc = values[bname][key], "maxp"
            else:
                base_d, sc = best_for(bname, key)
            r = paired(ours_d, base_d, lower_better=lower_better)
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

    out_path = REPO / "results" / "cifar100_summary.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
