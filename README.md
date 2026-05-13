# Fisher Information–based Evidential Deep Learning (FI-EDL)

**English** | [한국어](README.ko.md)

Reproduction package for the FI-EDL (Fisher Information–based Evidential Deep
Learning) paper. Trains and evaluates five loss variants on MNIST and CIFAR-10,
then reproduces the paper's main tables (OOD detection, confidence, ECE).

| Method | Hydra `experiment` key | Loss name |
|---|---|---|
| EDL (λ = 1.0) | `edl_l1` | `edl_fixed` |
| I-EDL (Deng et al., 2023) | `i_edl` | `i_edl` |
| R-EDL (Chen et al., 2024a) | `r_edl` | `r_edl` |
| Re-EDL (Chen et al., 2024b) | `re_edl` | `re_edl` |
| FI-EDL (this paper) | `fi_edl` | `fi_edl` |

## Setup

```bash
cd FI-EDL
uv sync --dev
```

## GPU requirements

Any CUDA-capable GPU with **4 GB VRAM or more** can run the full experiment suite.

| Experiment | Backbone | Minimum VRAM |
|---|---|---|
| MNIST (ConvNet) | 4-layer ConvNet (~2.4 M params) | 2 GB |
| CIFAR-10 (VGG-16) | CIFAR-style VGG-16 BN (~9.5 M params) | 4 GB |

Both datasets use batch size 64 and full float32 precision.
A **6 GB GPU** (e.g. RTX 3060, RTX 2060, GTX 1080) comfortably covers both.
Training times in this README are measured on a single NVIDIA H100 80 GB GPU;
expect 4–10× longer on a mid-range consumer GPU.

## Single run

```bash
# Train
uv run python -m src.train experiment=fi_edl dataset=cifar10 seed=0

# Evaluate (checkpoint path: runs/<exp>/seed_<n>/<ts>/checkpoints/best.ckpt)
uv run python -m src.eval experiment=fi_edl dataset=cifar10 seed=0 checkpoint=<PATH>
```

## Reproducing paper results

`run.py preset <name>` handles train → eval automatically and writes results
under `runs/`. Estimated time is per seed on a single H100 80GB GPU.

### Main results (Tables 1–2)

Pick the method and dataset you want to reproduce:

```bash
# FI-EDL, I-EDL, R-EDL, EDL — MNIST (ConvNet, ~30–52 min/seed × 5 seeds)
uv run python run.py preset main_mnist

# FI-EDL, I-EDL, R-EDL, EDL — CIFAR-10 (VGG-16, ~73–76 min/seed × 5 seeds)
uv run python run.py preset main_cifar10

# Re-EDL — MNIST  (lambda_prior=0.1, ~31 min/seed × 5 seeds)
uv run python run.py preset baseline_re_edl_mnist

# Re-EDL — CIFAR-10  (lambda_prior=0.8, ~158 min/seed × 5 seeds)
uv run python run.py preset baseline_re_edl_cifar10
```

> Re-EDL uses a different `lambda_prior` on CIFAR-10 (0.8) vs MNIST (0.1),
> so it runs as a separate preset instead of being bundled with the others.

### Controller ablation (Table 3)

```bash
uv run python run.py preset controller_constant       # constant gate
uv run python run.py preset controller_alpha0_gate    # α₀ gate
uv run python run.py preset controller_fim_nodetach   # FIM gate, no detach
# FI-EDL default row reuses main_cifar10 results
```

### β·γ sensitivity (Appendix)

```bash
# Default configuration (β=1.0, γ=1.0) — already covered by main_cifar10
uv run python run.py preset bg_b10_g10

# Full 3×3 grid
for p in bg_b05_g05 bg_b05_g10 bg_b05_g20 \
         bg_b10_g05 bg_b10_g10 bg_b10_g20 \
         bg_b20_g05 bg_b20_g10 bg_b20_g20; do
  uv run python run.py preset $p
done
```

### Single-seed quick test (~30–75 min)

To verify a method works before committing to 5 seeds:

```bash
uv run python run.py preset main_cifar10 seed=0   # FI-EDL + baselines, 1 seed
```

### Run all experiments (~132 GPU-hours total)

```bash
bash scripts/reproduce_paper.sh
```

## Tables and figures

After runs accumulate under `runs/`, aggregate with:

```bash
# Paper tables (NeurIPS format)
uv run python scripts/paper/extract_v4_tables.py

uv run python scripts/build_table_ood.py   --runs runs --out results/table_ood.csv
uv run python scripts/build_table_conf.py  --runs runs --out results/table_conf.csv
uv run python scripts/build_table_ece.py   --runs runs --out results/table_ece.csv

uv run python scripts/plot_reliability.py       --runs runs --out results/reliability.pdf
uv run python scripts/plot_training_dynamics.py --runs runs --out results/dynamics.pdf
```

## Layout

```
configs/
  config.yaml          # shared defaults
  experiment/*.yaml    # 5 method selectors
  dataset/*.yaml       # MNIST / CIFAR-10
  backbone/*.yaml      # convnet / vgg16 / resnet18
  paper/*.yaml         # presets (methods × seeds × overrides)
src/
  contracts/           # protocols (Backbone, Head, Loss, Score)
  registry/            # plugin registry + side-effect registration imports
  data/                # LightningDataModule + MNIST / CIFAR-10 adapters
  models/              # LightningModule + backbones + heads
  losses/              # edl_fixed, i_edl, r_edl, re_edl, fi_edl
  scores/              # maxp, alpha0, vacuity
  metrics/             # OOD / calibration metrics (numpy)
  callbacks/           # NaN detector
  reporting/           # metric JSONL writer
  train.py
  eval.py
scripts/               # table / figure builders
tests/                 # pytest smoke
run.py                 # preset driver
```

## Extending

To add a new loss / backbone / head / score, create the corresponding file and
register it exactly once with `@..._REGISTRY.register("name")`. Registration
happens via the side-effect imports in `src/registry/__init__.py` — do not
remove those imports, or the registry keys will disappear.

## License

MIT. See [LICENSE](LICENSE).
