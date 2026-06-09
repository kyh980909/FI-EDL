"""Forward-only Brier re-eval from saved checkpoints.

For each method x seed, loads the SAME eval config_resolved.yaml that produced
the calibration table, re-runs the ID test forward pass, and computes the
multiclass Brier score. Cross-checks the regenerated ECE against the value
recorded in the eval's metrics.jsonl to guarantee the probs are identical to
those that produced the published ECE/NLL.

Probs were NOT persisted during eval (only summary metrics), so this is a
forward-only re-eval (no training).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datamodule import FIEDLDataModule  # noqa: E402
from src.eval import _collect_outputs, _enable_checkpoint_safe_globals  # noqa: E402
from src.metrics.ood_metrics import (  # noqa: E402
    multiclass_brier,
    multiclass_ece,
    multiclass_nll,
)
from src.models.lit_module import FIEDLLightningModule  # noqa: E402


# (dataset, method) -> eval-dir suffix selector. The suffix is what follows the
# backbone tag in the eval dir name. For FI-EDL we pick the configuration that
# the published calibration table reports (MNIST=SN-off plain; CIFAR-10=+SN).
TASKS = {
    "mnist": {
        "edl_l1": "_mnist_convnet_edl_fixed_l1",
        "i_edl": "_mnist_convnet",
        "r_edl": "_mnist_convnet",
        "re_edl": "_mnist_convnet",
        "daedl": "_mnist_convnet",
        "f_edl": "_mnist_convnet",
        "fi_edl": "_mnist_convnet",  # SN-off, plain (exclude *_fi_edl_sn)
    },
    "cifar10": {
        "edl_l1": "_cifar10_vgg16_edl_fixed_l1",
        "i_edl": "_cifar10_vgg16",
        "r_edl": "_cifar10_vgg16",
        "re_edl": "_cifar10_vgg16",
        "daedl": "_cifar10_vgg16",
        "f_edl": "_cifar10_vgg16",
        "fi_edl": "_cifar10_vgg16_fi_edl_sn",  # +SN per table
    },
    "cifar100": {
        "re_edl": "_cifar100_resnet18",
        "daedl": "_cifar100_resnet18",
        "f_edl": "_cifar100_resnet18",
        "fi_edl": "_cifar100_resnet18_fi_edl_sn",
    },
}

DISPLAY = {
    "edl_l1": "EDL",
    "i_edl": "I-EDL",
    "r_edl": "R-EDL",
    "re_edl": "Re-EDL",
    "daedl": "DAEDL",
    "f_edl": "F-EDL",
    "fi_edl": "FI-EDL",
}


def _pick_eval_dir(dataset: str, method: str, seed: int, suffix: str) -> str | None:
    base = f"runs/{dataset}/{method}/seed_{seed}"
    cands = glob.glob(f"{base}/eval_*{suffix}")
    out = []
    for c in cands:
        name = os.path.basename(c.rstrip("/"))
        # everything after the timestamp+tag prefix must equal suffix exactly,
        # so we don't accidentally match longer variant suffixes.
        # name looks like eval_<ts>_<dataset>_<backbone>[suffix-extra]
        # We require the name to END with suffix AND not contain extra variant
        # tokens beyond it.
        if not name.endswith(suffix):
            continue
        # Reconstruct the part after eval_<ts>_ ; if suffix is the bare backbone
        # tag we must reject names that have additional tokens (variants).
        # Strip "eval_<ts>_" prefix:
        parts = name.split("_", 2)  # eval, <ts>, rest
        rest = parts[2] if len(parts) > 2 else ""
        if rest != suffix.lstrip("_"):
            continue
        out.append(c)
    if not out:
        return None
    return sorted(out)[-1]


def _recorded_ece(eval_dir: str) -> float | None:
    mj = os.path.join(eval_dir, "metrics.jsonl")
    if not os.path.exists(mj):
        return None
    for line in open(mj):
        d = json.loads(line)
        if d.get("split") == "conf_eval" and d.get("score_type") == "maxp":
            return float(d["metrics"]["ece"])
    return None


def main() -> None:
    _enable_checkpoint_safe_globals()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict = {}
    only_ds = sys.argv[1] if len(sys.argv) > 1 else None

    for dataset, methods in TASKS.items():
        if only_ds and dataset != only_ds:
            continue
        results[dataset] = {}
        for method, suffix in methods.items():
            per_seed = []
            for seed in range(5):
                ed = _pick_eval_dir(dataset, method, seed, suffix)
                if ed is None:
                    print(f"[skip] {dataset}/{method}/seed_{seed}: no eval dir", flush=True)
                    continue
                cfg = OmegaConf.load(os.path.join(ed, "config_resolved.yaml"))
                ckpt = cfg.checkpoint
                if not os.path.exists(ckpt):
                    print(f"[skip] {dataset}/{method}/seed_{seed}: missing ckpt {ckpt}", flush=True)
                    continue
                dm = FIEDLDataModule(cfg)
                dm.setup()
                model = FIEDLLightningModule.load_from_checkpoint(ckpt, cfg=cfg)
                model.to(device)
                temperature = float(
                    cfg.eval.temperature if cfg.eval.calibration == "temperature" else 1.0
                )
                out = _collect_outputs(model, dm.test_dataloader(), temperature=temperature)
                probs, labels = out["probs"], out["labels"]
                brier = multiclass_brier(probs, labels)
                ece = multiclass_ece(probs, labels)
                nll = multiclass_nll(probs, labels)
                rec = _recorded_ece(ed)
                flag = ""
                if rec is not None and abs(rec - ece) > 1e-4:
                    flag = f"  !! ECE mismatch recorded={rec:.4f} recomputed={ece:.4f}"
                print(
                    f"{dataset:8s} {method:7s} seed{seed} brier={brier:.4f} "
                    f"ece={ece*100:.2f} (rec {rec*100 if rec else float('nan'):.2f}) "
                    f"nll={nll:.4f}{flag}",
                    flush=True,
                )
                per_seed.append({"seed": seed, "brier": brier, "ece": ece, "nll": nll})
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            results[dataset][method] = per_seed

    os.makedirs("results", exist_ok=True)
    with open("results/brier_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n===== SUMMARY (Brier x100, mean±std ddof=1) =====")
    for dataset, md in results.items():
        print(f"\n## {dataset}")
        for method, ps in md.items():
            if not ps:
                continue
            b = np.array([p["brier"] for p in ps]) * 100.0
            print(
                f"{DISPLAY[method]:8s} n={len(b)} Brier={b.mean():.2f}±{b.std(ddof=1):.2f}"
            )


if __name__ == "__main__":
    main()
