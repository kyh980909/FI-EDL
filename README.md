# FI-EDL

**English** | [한국어](README.ko.md)

Reproduction package for the FI-EDL (Fisher Information–based Evidential Deep
Learning) paper. Trains and evaluates five loss variants on MNIST and CIFAR-10,
then reproduces the paper's main tables (OOD detection, confidence, ECE).

| Method | Hydra `experiment` key | Loss name |
|---|---|---|
| EDL (λ = 1.0) | `edl_l1` | `edl_fixed` |
| EDL (λ = 0.1) | `edl_l01` | `edl_fixed` |
| EDL (λ = 0.001) | `edl_l0001` | `edl_fixed` |
| I-EDL (baseline) | `i_edl` | `i_edl` |
| FI-EDL (this paper) | `fi_edl` | `fi_edl` |

## Setup

```bash
cd FI-EDL
uv sync --dev
```

## Single run

```bash
# Train
uv run python -m src.train experiment=fi_edl dataset=cifar10 seed=0

# Evaluate (checkpoint path: runs/<exp>/seed_<n>/<ts>/checkpoints/best.ckpt)
uv run python -m src.eval experiment=fi_edl dataset=cifar10 seed=0 checkpoint=<PATH>
```

## Paper reproduction presets

```bash
# MNIST main results (5 methods × 5 seeds, MNIST rows of Tables 2, 3, 4)
uv run python run.py preset main_mnist

# CIFAR-10 main results (5 methods × 5 seeds, CIFAR-10 rows of Tables 2, 3, 4)
uv run python run.py preset main_cifar10

# FI-EDL controller ablation (CIFAR-10)
uv run python run.py preset controller_ablation
```

Each preset runs train → eval → metric-row JSONL write end to end.

## Tables and figures

After runs accumulate under `runs/`, aggregate with:

```bash
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
  backbone/*.yaml      # convnet / resnet18
  paper/*.yaml         # presets (methods × seeds × overrides)
src/
  contracts/           # protocols (Backbone, Head, Loss, Score)
  registry/            # plugin registry + side-effect registration imports
  data/                # LightningDataModule + MNIST / CIFAR-10 adapters
  models/              # LightningModule + backbones + heads
  losses/              # edl_fixed, fi_edl, i_edl
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
