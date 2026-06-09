"""Aggregate CIFAR-10 → GTSRB forward-only OOD eval (Re-EDL setup extension).

Reads runs_reeval/cifar10_extended/<method>/seed_*/eval_*/metrics.jsonl
and writes results/gtsrb_summary.md.
"""
from __future__ import annotations
import glob, json, re
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent


def collect(method_dir: str, variant_re_pat: str, score: str):
    out = {}
    pat = re.compile(variant_re_pat)
    for f in glob.glob(str(REPO / f"runs_reeval/cifar10_extended/{method_dir}/seed_*/eval_*/metrics.jsonl")):
        if not pat.search(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        if not rows: continue
        seed = rows[0].get("seed")
        for r in rows:
            if r.get("split") != "eval": continue
            if r.get("dataset") != "gtsrb": continue
            if r.get("score_type") != score: continue
            v = r["metrics"].get("auroc")
            if v is not None:
                out.setdefault(seed, v); break
    return out


def paired(ours, base):
    seeds = sorted(set(ours.keys()) & set(base.keys()))
    if len(seeds) < 3: return None
    a = np.array([ours[s] for s in seeds]); b = np.array([base[s] for s in seeds])
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)])
    ci = np.percentile(boots, [2.5, 97.5])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    verdict = "WIN" if diff.mean() > 0 and p < 0.05 else ("loss" if diff.mean() < 0 and p < 0.05 else "tie")
    return dict(n=len(seeds), diff_mean=diff.mean(), diff_std=diff.std(ddof=1),
                p=p, ci=tuple(ci), sig=sig, verdict=verdict,
                ours_mean=a.mean(), base_mean=b.mean())


METHOD_VAR = {
    "FI-EDL+SN": ("fi_edl", r"_fi_edl_sn/"),
    "DAEDL":     ("daedl",  r"_cifar10_vgg16/"),
    "F-EDL":     ("f_edl",  r"_cifar10_vgg16/"),
    "Re-EDL":    ("re_edl", r"_cifar10_vgg16/"),
}
BEST_SCORE = {"FI-EDL+SN": "alpha0", "DAEDL": "alpha0", "F-EDL": "maxp", "Re-EDL": "alpha0"}


def fmt(d):
    if not d: return "—"
    a = np.array(list(d.values())) * 100
    if len(a) < 2: return f"{a[0]:.2f} (n=1)"
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"


def main():
    md = ["# CIFAR-10 → GTSRB OOD (Re-EDL Table 2 extension)", ""]
    md.append("Forward-only eval on CIFAR-10/VGG-16 checkpoints (5 seeds).")
    md.append("GTSRB (German Traffic Sign Recognition Benchmark): 12,630 test traffic-sign images,")
    md.append("resize-then-crop to 32x32 to match CIFAR-10 input. Standard Re-EDL CIFAR-10 OOD.")
    md.append("")
    md.append("## AUROC headline")
    md.append("")
    md.append("| Method | Score | AUROC | n |")
    md.append("|---|---|---|---|")
    values = {}
    for label, (mdir, vre) in METHOD_VAR.items():
        score = BEST_SCORE[label]
        d = collect(mdir, vre, score)
        values[label] = d
        md.append(f"| {label} | {score} | {fmt(d)} | {len(d)} |")
    md.append("")
    md.append("## Paired tests vs FI-EDL+SN")
    md.append("")
    md.append("| baseline | n | Δ (ours − base) | p | 95% CI | sig | verdict |")
    md.append("|---|---|---|---|---|---|---|")
    ours = values["FI-EDL+SN"]
    for bn in ["DAEDL", "F-EDL", "Re-EDL"]:
        r = paired(ours, values[bn])
        if r is None: md.append(f"| {bn} | <3 | — | — | — | — | — |"); continue
        md.append(
            f"| {bn} | {r['n']} | {r['diff_mean']*100:+.2f} ± {r['diff_std']*100:.2f} | "
            f"{r['p']:.4f} | [{r['ci'][0]*100:+.2f}, {r['ci'][1]*100:+.2f}] | "
            f"{r['sig']} | **{r['verdict']}** |"
        )
    out_path = REPO / "results" / "gtsrb_summary.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
