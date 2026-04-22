# FI-EDL

[English](README.md) | **한국어**

Fisher Information–based Evidential Deep Learning (FI-EDL) 논문 실험 재현
패키지. 다섯 가지 손실 방법을 MNIST·CIFAR-10에서 학습·평가하고 논문의 주요
표(OOD 탐지, 신뢰도, ECE)를 재현합니다.

| 메서드 | Hydra `experiment` 키 | 손실 이름 |
|---|---|---|
| EDL (λ = 1.0) | `edl_l1` | `edl_fixed` |
| EDL (λ = 0.1) | `edl_l01` | `edl_fixed` |
| EDL (λ = 0.001) | `edl_l0001` | `edl_fixed` |
| I-EDL (기준선) | `i_edl` | `i_edl` |
| FI-EDL (본 연구) | `fi_edl` | `fi_edl` |

## 설치

```bash
cd FI-EDL
uv sync --dev
```

## 단일 실행

```bash
# 학습
uv run python -m src.train experiment=fi_edl dataset=cifar10 seed=0

# 평가 (체크포인트 경로는 runs/<exp>/seed_<n>/<ts>/checkpoints/best.ckpt)
uv run python -m src.eval experiment=fi_edl dataset=cifar10 seed=0 checkpoint=<PATH>
```

## 논문 재현 프리셋

```bash
# MNIST 메인 결과 (5 methods × 5 seeds, 표 2·3·4의 MNIST 행)
uv run python run.py preset main_mnist

# CIFAR-10 메인 결과 (5 methods × 5 seeds, 표 2·3·4의 CIFAR-10 행)
uv run python run.py preset main_cifar10

# FI-EDL 컨트롤러 ablation (CIFAR-10)
uv run python run.py preset controller_ablation
```

프리셋은 학습 → 평가 → 메트릭 JSONL 기록까지 한 번에 수행합니다.

## 표·그림 생성

학습·평가가 완료되어 `runs/`에 결과가 쌓이면 아래로 집계합니다.

```bash
uv run python scripts/build_table_ood.py   --runs runs --out results/table_ood.csv
uv run python scripts/build_table_conf.py  --runs runs --out results/table_conf.csv
uv run python scripts/build_table_ece.py   --runs runs --out results/table_ece.csv

uv run python scripts/plot_reliability.py       --runs runs --out results/reliability.pdf
uv run python scripts/plot_training_dynamics.py --runs runs --out results/dynamics.pdf
```

## 디렉터리

```
configs/
  config.yaml          # 공통 기본값
  experiment/*.yaml    # 5개 메서드 선택자
  dataset/*.yaml       # MNIST / CIFAR-10
  backbone/*.yaml      # convnet / resnet18
  paper/*.yaml         # 프리셋 (methods × seeds × overrides)
src/
  contracts/           # 프로토콜 (Backbone, Head, Loss, Score)
  registry/            # 플러그인 레지스트리 + 등록 side-effect import
  data/                # LightningDataModule + MNIST / CIFAR-10 어댑터
  models/              # LightningModule + backbones + heads
  losses/              # edl_fixed, fi_edl, i_edl
  scores/              # maxp, alpha0, vacuity
  metrics/             # OOD · calibration 메트릭 (numpy)
  callbacks/           # NaN 감지
  reporting/           # 메트릭 JSONL 기록
  train.py
  eval.py
scripts/               # 표·그림 빌더
tests/                 # pytest smoke
run.py                 # preset 드라이버
```

## 확장

새 손실 / 백본 / 헤드 / 스코어를 추가할 때는 대응하는 파일을 만들고
`@..._REGISTRY.register("name")` 데코레이터로 한 번만 등록합니다. 등록은
`src/registry/__init__.py` 안의 side-effect import를 통해 자동으로
이루어지므로, import를 삭제하면 레지스트리 키가 사라지는 점에 유의하세요.

## 라이선스

MIT. [LICENSE](LICENSE) 참고.
