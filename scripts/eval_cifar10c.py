"""CIFAR-10-C robustness eval — DAEDL Table 4 / F-EDL Table 2 protocol.

For each (method, seed) checkpoint, runs forward-only inference over the
19 corruption types × 5 severity levels of CIFAR-10-C and reports, per
severity averaged over 19 corruption types:
  - classification accuracy under corruption
  - ECE (calibration) under corruption
  - distribution-shift AUPR (clean test set = ID label 0, corrupted = label 1)

Run after sweeping CIFAR-10/VGG-16 main results. Writes
`results/cifar10c_summary.md`.

Usage:
    uv run python scripts/eval_cifar10c.py \
        [--ckpt-glob 'runs/cifar10/...']
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score
from torchvision import transforms

import src.registry  # noqa: F401 init registries
from src.data.cifar10_c import CIFAR10CDataset, CORRUPTION_TYPES
from src.models.lit_module import FIEDLLightningModule
from src.metrics.ood_metrics import multiclass_ece
from torchvision import datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def _forward(model, loader) -> Dict[str, np.ndarray]:
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            probs_all.append(out["probs"].detach().cpu().numpy())
            labels_all.append(y.numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    return {"probs": probs, "labels": labels, "maxp": probs.max(axis=1)}


def _load_model(ckpt_path: str, sn: bool):
    cfg_path = Path(ckpt_path).parent.parent / "config_resolved.yaml"
    cfg = OmegaConf.load(cfg_path)
    # ensure backbone_spectral_norm matches the training-time setting for FI-EDL+SN
    if sn:
        cfg.model.backbone_spectral_norm = True
    model = FIEDLLightningModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.to(DEVICE)
    return model


def _clean_loader(batch_size: int = 256) -> DataLoader:
    ds = datasets.CIFAR10(str(REPO / "data"), train=False, download=True,
                          transform=_eval_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def _corrupted_loader(corruption: str, severity: int, batch_size: int = 256) -> DataLoader:
    ds = CIFAR10CDataset(
        root=str(REPO / "data"),
        corruption=corruption,
        severity=severity,
        transform=_eval_transform(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def _eval_one_ckpt(ckpt_path: str, sn: bool, method: str, seed: int) -> Dict:
    model = _load_model(ckpt_path, sn=sn)
    clean = _forward(model, _clean_loader())
    out: Dict = {"method": method, "seed": seed, "clean_acc": float((clean["probs"].argmax(1) == clean["labels"]).mean())}
    per_sev: Dict[int, Dict[str, List[float]]] = {s: defaultdict(list) for s in range(1, 6)}
    for sev in range(1, 6):
        for corr in CORRUPTION_TYPES:
            r = _forward(model, _corrupted_loader(corr, sev))
            acc = float((r["probs"].argmax(1) == r["labels"]).mean())
            ece = float(multiclass_ece(r["probs"], r["labels"]))
            # distribution-shift AUPR: ID (label 0) = clean test, OOD (label 1) = corrupted
            scores = np.concatenate([-clean["maxp"], -r["maxp"]])  # higher = more OOD
            labels = np.concatenate([np.zeros(len(clean["maxp"])), np.ones(len(r["maxp"]))])
            aupr = float(average_precision_score(labels, scores))
            per_sev[sev]["acc"].append(acc)
            per_sev[sev]["ece"].append(ece)
            per_sev[sev]["aupr"].append(aupr)
        print(f"  [{method} seed={seed}] severity {sev} done, mean acc={np.mean(per_sev[sev]['acc']):.4f} ece={np.mean(per_sev[sev]['ece']):.4f} aupr={np.mean(per_sev[sev]['aupr']):.4f}")
    out["per_severity"] = {
        s: {k: float(np.mean(v)) for k, v in d.items()} for s, d in per_sev.items()
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/cifar10c_summary.md")
    args = parser.parse_args()

    # Map method label → (ckpt-glob pattern, needs_sn)
    METHODS = {
        "FI-EDL+SN": (str(REPO / "runs/cifar10/fi_edl/seed_*/train_*_fi_edl_sn/checkpoints/best.ckpt"), True),
        "DAEDL":     (str(REPO / "runs/cifar10/daedl/seed_*/train_*_cifar10_vgg16/checkpoints/best.ckpt"), False),
        "F-EDL":     (str(REPO / "runs/cifar10/f_edl/seed_*/train_*_cifar10_vgg16/checkpoints/best.ckpt"), False),
        "Re-EDL":    (str(REPO / "runs/cifar10/re_edl/seed_*/train_*_cifar10_vgg16/checkpoints/best.ckpt"), False),
    }
    all_results = []
    for method, (pat, sn) in METHODS.items():
        for ckpt in sorted(glob.glob(pat)):
            seed_m = re.search(r"seed_(\d+)", ckpt)
            seed = int(seed_m.group(1)) if seed_m else -1
            print(f"\n=== {method} seed={seed} ===")
            try:
                r = _eval_one_ckpt(ckpt, sn=sn, method=method, seed=seed)
                all_results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")

    # aggregate per (method, severity) across seeds
    md = ["# CIFAR-10-C Distribution Shift Detection (DAEDL/F-EDL Setup)", ""]
    md.append("Forward-only eval on existing CIFAR-10/VGG-16 checkpoints (5 seeds each).")
    md.append("Per severity: mean across 19 corruption types; then mean ± std across 5 seeds.")
    md.append("Distribution-shift AUPR: clean CIFAR-10 test = ID (label 0), corrupted = OOD (label 1); score = −maxp.")
    md.append("")
    md.append("## AUPR (distribution-shift detection) per severity")
    md.append("")
    md.append("| Method | C=1 | C=2 | C=3 | C=4 | C=5 |")
    md.append("|---|---|---|---|---|---|")
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        for sev, m in r["per_severity"].items():
            grouped[r["method"]][int(sev)].append(m["aupr"] * 100)
    for method in METHODS:
        row = [method]
        for sev in range(1, 6):
            vals = grouped[method].get(sev, [])
            if len(vals) >= 2:
                row.append(f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}")
            elif vals:
                row.append(f"{vals[0]:.2f}")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Acc (classification under corruption) per severity")
    md.append("")
    md.append("| Method | C=1 | C=2 | C=3 | C=4 | C=5 |")
    md.append("|---|---|---|---|---|---|")
    grouped_acc: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        for sev, m in r["per_severity"].items():
            grouped_acc[r["method"]][int(sev)].append(m["acc"] * 100)
    for method in METHODS:
        row = [method]
        for sev in range(1, 6):
            vals = grouped_acc[method].get(sev, [])
            if len(vals) >= 2:
                row.append(f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}")
            elif vals:
                row.append(f"{vals[0]:.2f}")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## ECE (calibration under corruption) per severity")
    md.append("")
    md.append("| Method | C=1 | C=2 | C=3 | C=4 | C=5 |")
    md.append("|---|---|---|---|---|---|")
    grouped_ece: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        for sev, m in r["per_severity"].items():
            grouped_ece[r["method"]][int(sev)].append(m["ece"] * 100)
    for method in METHODS:
        row = [method]
        for sev in range(1, 6):
            vals = grouped_ece[method].get(sev, [])
            if len(vals) >= 2:
                row.append(f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}")
            elif vals:
                row.append(f"{vals[0]:.2f}")
            else:
                row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # save raw too
    raw = REPO / "results" / "cifar10c_raw.json"
    raw.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    out_path = REPO / args.out
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out_path} and {raw}")


if __name__ == "__main__":
    main()
