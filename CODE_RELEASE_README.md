# FI-EDL+SN — Code Release for Neurocomputing submission

Reproducibility companion for the paper *"Spectral-Normalized, KL-Free
Evidential Deep Learning: A Simpler Recipe for Calibrated Uncertainty
and OOD Detection."* The full manuscript is in `paper/main.tex`.

## What's here

```
src/                # PyTorch Lightning + Hydra training/eval code
configs/            # Hydra configs for all baselines + the FI-EDL+SN recipe
configs/paper/      # Paper-specific multi-seed/multi-method presets
scripts/            # Reproduction scripts (one per experiment)
results/            # Aggregated CSVs and the source-of-truth report
                    # baseline_comparison_report.md.
runs/               # Per-(method, seed) training and eval artifacts.
                    # Not committed; recreated by running the scripts.
.claude/            # Internal AI-assistant configuration; not required to
                    # reproduce, but documents the paper-writing pipeline.
```

## Quick reproduction

```bash
# 0. Environment
curl -fsSL https://astral.sh/uv/install.sh | sh
uv sync --dev          # installs PyTorch, Lightning, Hydra, scipy, scikit-learn

# 1. Recommended recipe — FI-EDL+SN on CIFAR-10 (5 seeds)
uv run python run.py preset main_cifar10 \
    "methods=[fi_edl]" \
    "model.backbone_spectral_norm=true"

# 2. Baseline FI-EDL on MNIST (5 seeds; SN off because of §6.3)
uv run python run.py preset main_mnist \
    "methods=[fi_edl]"

# 3. All competitor baselines on CIFAR-10 (5 seeds each)
uv run python run.py preset main_cifar10 \
    "methods=[edl_l1, i_edl, r_edl, re_edl, daedl, f_edl]"

# 4. Aggregation + statistical tests
uv run python scripts/aggregate_cifar10.py
uv run python scripts/aggregate_resnet_gap1.py   # if Gap 1 sweep done
```

## What each method is

| Method (paper label) | Loss / head | Config |
|---|---|---|
| EDL (Sensoy 2018)        | Brier + KL-to-uniform (annealed) | `experiment=edl_l1` |
| I-EDL (Deng 2023 ICML)   | Fisher-info-based KL             | `experiment=i_edl` |
| R-EDL (Chen 2024 ICLR)   | Relaxed prior weight             | `experiment=r_edl` |
| Re-EDL (Chen 2024 arXiv) | Tunable prior, KL-free           | `experiment=re_edl` |
| DAEDL (Yoon 2024 ICML)   | Standard EDL + GMM density       | `experiment=daedl` |
| F-EDL = Flexible EDL (Yoon 2025 NeurIPS) | Flexible Dirichlet (α, p, τ) + Brier on p | `experiment=f_edl` |
| **FI-EDL+SN (this paper)** | Brier-only fit (no KL); backbone SN | `experiment=fi_edl` + `model.backbone_spectral_norm=true` |

Naming note: the prior FI-EDL paper used the abbreviation "F-EDL" for an
internal Fisher-information variant. Following the published name of
Yoon & Kim 2025 NeurIPS, **F-EDL** in this repository means "Flexible
EDL" and our method is consistently "FI-EDL+SN" (Fisher-Information EDL
with spectral normalisation). This matches the manuscript.

## Hardware notes

All experiments were run on a single GPU. Approximate wall-times per run
(200-epoch budget with early stopping on `val/acc`, patience 20):

| Backbone | Dataset | Per-seed time |
|---|---|---|
| ConvNet (3 conv layers) | MNIST | ~15–20 min |
| VGG-16                  | CIFAR-10 | ~1–1.5 h |
| ResNet-18               | CIFAR-10 | ~50 min |

5-seed totals: ~1.5h (MNIST), ~5–7h per method (CIFAR-10 VGG-16),
~4h per method (CIFAR-10 ResNet-18).

## Logged outputs

Each run writes to `runs/<dataset>/<method>/seed_<N>/train_<ts>_<...>/`
with `checkpoints/best.ckpt`, `config_resolved.yaml`, `env.json`, and a
`summary.json`. Eval runs (forward-only, per seed) write to
`eval_<ts>_<...>/metrics.jsonl`. The aggregated CSVs in `results/` are
generated from these JSONL files by `scripts/aggregate_*.py`.

## Statistical tests

Paired t-test and 10k-resample bootstrap 95% CI between FI-EDL+SN and
each baseline (DAEDL, F-EDL, Re-EDL) on each headline metric, per-seed
paired:

```bash
uv run python scripts/aggregate_resnet_gap1.py
```

The CIFAR-10 stats (VGG-16) are in `results/stats_significance.md` and
the corresponding report subsection is §6.6.7 of
`results/baseline_comparison_report.md`.

## Citing

[BIBTEX TODO — fill once Neurocomputing assigns DOI.]

## License

[USER: fill in — MIT or Apache-2.0 typical for ML papers.]
