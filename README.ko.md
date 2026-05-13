# Fisher Information–based Evidential Deep Learning (FI-EDL)

[English](README.md) | **한국어**

Fisher Information–based Evidential Deep Learning (FI-EDL) 논문 실험 재현
패키지. **일곱 가지** EDL 방법을 MNIST·CIFAR-10에서 학습·평가하고 논문의 주요
표(OOD 탐지, 신뢰도, ECE)를 재현합니다. 최근 발표된 비교 베이스라인인
DAEDL·F-EDL도 포함되어 있습니다(Neurocomputing 저널 확장 대응).

| 메서드 | Hydra `experiment` 키 | 손실 이름 | 헤드 |
|---|---|---|---|
| EDL (λ = 1.0) | `edl_l1` | `edl_fixed` | `edl` |
| I-EDL (Deng et al., 2023) | `i_edl` | `i_edl` | `edl` |
| R-EDL (Chen et al., 2024a) | `r_edl` | `r_edl` | `edl` |
| Re-EDL (Chen et al., 2024b) | `re_edl` | `re_edl` | `edl` |
| **DAEDL** (Yoon et al., 2024) | `daedl` | `daedl` | `daedl` |
| **F-EDL** (Yoon et al., 2025) | `f_edl` | `f_edl` | `f_edl` |
| FI-EDL (본 연구) | `fi_edl` | `fi_edl` | `edl` |

## 설치

```bash
cd FI-EDL
uv sync --dev
```

## GPU 요구사항

**VRAM 4 GB 이상**의 CUDA GPU라면 전체 실험을 실행할 수 있습니다.

| 실험 | 백본 | 최소 VRAM |
|---|---|---|
| MNIST (ConvNet) | 4-layer ConvNet (약 2.4M 파라미터) | 2 GB |
| CIFAR-10 (VGG-16) | CIFAR용 VGG-16 BN (약 9.5M 파라미터) | 4 GB |

배치 크기 64, float32 기준입니다.
**6 GB GPU** (RTX 3060, RTX 2060, GTX 1080 등)라면 두 데이터셋 모두 여유롭게 실행 가능합니다.
README에 표기된 학습 시간은 NVIDIA H100 80 GB 기준이며,
일반 소비자용 GPU에서는 4–10배 더 걸릴 수 있습니다.

## 단일 실행

```bash
# 학습
uv run python -m src.train experiment=fi_edl dataset=cifar10 seed=0

# 평가 (체크포인트 경로는 runs/<exp>/seed_<n>/<ts>/checkpoints/best.ckpt)
uv run python -m src.eval experiment=fi_edl dataset=cifar10 seed=0 checkpoint=<PATH>
```

## 논문 결과 재현

`run.py preset <name>`은 학습 → 평가를 자동으로 수행하고 결과를 `runs/` 아래에 기록합니다.
소요 시간은 H100 80GB 기준 seed당 추정치입니다.

### 메인 결과 (표 1–2)

재현하고 싶은 기법과 데이터셋을 선택하세요:

```bash
# FI-EDL, I-EDL, R-EDL, EDL — MNIST (ConvNet, ~30–52분/seed × 5 seeds)
uv run python run.py preset main_mnist

# FI-EDL, I-EDL, R-EDL, EDL — CIFAR-10 (VGG-16, ~73–76분/seed × 5 seeds)
uv run python run.py preset main_cifar10

# Re-EDL — MNIST  (lambda_prior=0.1, ~31분/seed × 5 seeds)
uv run python run.py preset baseline_re_edl_mnist

# Re-EDL — CIFAR-10  (lambda_prior=0.8, ~158분/seed × 5 seeds)
uv run python run.py preset baseline_re_edl_cifar10
```

> Re-EDL은 CIFAR-10(0.8)과 MNIST(0.1)에서 서로 다른 `lambda_prior`를 사용하므로
> 별도 프리셋으로 분리되어 있습니다.

### DAEDL·F-EDL 베이스라인 (Neurocomputing 저널 확장)

개별 추가 또는 7-method 전체 비교로 실행합니다.

```bash
# 개별 베이스라인
uv run python run.py preset baseline_fedl_mnist        # F-EDL MNIST  (~30분/seed × 5)
uv run python run.py preset baseline_fedl_cifar10      # F-EDL CIFAR-10  (~80분/seed × 5)
uv run python run.py preset baseline_daedl_mnist       # DAEDL MNIST  (~35분/seed × 5)
uv run python run.py preset baseline_daedl_cifar10     # DAEDL CIFAR-10  (~110분/seed × 5)

# 7-method 전체 비교 (저널 메인 표)
uv run python run.py preset comparison_all_mnist       # 7-method × 5 seeds, MNIST
uv run python run.py preset comparison_all_cifar10     # 6-method × 5 seeds, CIFAR-10
uv run python run.py preset baseline_re_edl_cifar10    # Re-EDL CIFAR-10 (lambda_prior=0.8)
```

> **DAEDL 구현 주의**: 이 저장소의 DAEDL은 공식 normalizing-flow 밀도 추정 대신
> spectral norm + 학습 가능 class prototype 밀도를 사용하는 근사 버전입니다.
> 공식 코드 확보 시 `src/models/heads/daedl_head.py`만 교체하면 나머지 파이프라인은
> 그대로 사용할 수 있습니다.

### 컨트롤러 ablation (표 3)

```bash
uv run python run.py preset controller_constant       # 상수 게이트
uv run python run.py preset controller_alpha0_gate    # α₀ 게이트
uv run python run.py preset controller_fim_nodetach   # FIM 게이트, detach 없음
# FI-EDL 기본 행은 main_cifar10 결과를 그대로 사용
```

### β·γ 민감도 분석 (부록)

```bash
# 기본 설정 (β=1.0, γ=1.0) — main_cifar10 결과와 동일
uv run python run.py preset bg_b10_g10

# 3×3 전체 그리드
for p in bg_b05_g05 bg_b05_g10 bg_b05_g20 \
         bg_b10_g05 bg_b10_g10 bg_b10_g20 \
         bg_b20_g05 bg_b20_g10 bg_b20_g20; do
  uv run python run.py preset $p
done
```

### 단일 seed 빠른 테스트 (~30–75분)

5 seed 전체 실행 전에 동작 확인:

```bash
uv run python run.py preset main_cifar10 seed=0   # FI-EDL + 베이스라인, 1 seed
```

### 전체 실험 일괄 실행 (약 132 GPU-시간)

```bash
bash scripts/reproduce_paper.sh
```

## 표·그림 생성

학습·평가가 완료되어 `runs/`에 결과가 쌓이면 아래로 집계합니다.

```bash
# 논문 표 생성 (NeurIPS 형식)
uv run python scripts/paper/extract_v4_tables.py

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
  experiment/*.yaml    # 7개 메서드 선택자 (daedl, f_edl 포함)
  dataset/*.yaml       # MNIST / CIFAR-10
  backbone/*.yaml      # convnet / vgg16 / resnet18
  paper/*.yaml         # 프리셋 (methods × seeds × overrides)
src/
  contracts/           # 프로토콜 (Backbone, Head, Loss, Score)
  registry/            # 플러그인 레지스트리 + 등록 side-effect import
  data/                # LightningDataModule + MNIST / CIFAR-10 어댑터
  models/              # LightningModule + backbones + heads (edl, daedl, f_edl)
  losses/              # edl_fixed, i_edl, r_edl, re_edl, fi_edl, daedl, f_edl_flex
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
