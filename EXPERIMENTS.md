# 논문 실험 ↔ FI-EDL 매핑

논문(`Neurocomputing/FI-EDL_eng.tex`, `NeurIPS_2026/FI-EDL_eng_v4_oldcycle.tex`)의 실험 섹션과
이 저장소 구성 요소 간의 매핑을 한 곳에 정리한 문서입니다. 표/그림 단위로, 어떤 프리셋·
experiment·스크립트로 재현되는지 추적할 수 있게 했습니다.

---

## 0. 메서드 ↔ Hydra 설정 매핑

| 논문 표기 | Hydra `experiment` | `loss.name` | 설명 |
|---|---|---|---|
| EDL (λ = 1.0) | `edl_l1` | `edl_fixed` | Sensoy et al. 2018 베이스라인, KL 가중치 고정 |
| **I-EDL** (Deng et al., 2023) | `i_edl` | `i_edl` | Fisher-weighted MSE + log-det Fisher + linear KL anneal |
| **R-EDL** (Chen et al., 2024a) | `r_edl` | `r_edl` | KL 정규화 제거, λ_prior=0.1 (MNIST) / 0.8 (CIFAR-10) |
| **Re-EDL** (Chen et al., 2024b) | `re_edl` | `re_edl` | KL + variance 항 모두 제거, λ_prior=0.1 (MNIST) / 0.8 (CIFAR-10) |
| **FI-EDL (Ours)** | `fi_edl` | `fi_edl` | Fisher 정보 기반 적응형 KL 게이트 λ(v) = β·exp(−γ·v) |

데이터셋·백본:

| 데이터셋 | `dataset` | `backbone` | OOD 대상 |
|---|---|---|---|
| MNIST | `mnist` | `convnet` | KMNIST, FashionMNIST |
| CIFAR-10 | `cifar10` | `vgg16` | SVHN, CIFAR-100 |

---

## 1. 표 · 그림별 재현 매핑

### Table 2 — OOD Detection Summary (`tab:iedl_style_ood`)

- **논문 내용**: MNIST→KMNIST, MNIST→FMNIST, CIFAR10→SVHN, CIFAR10→CIFAR100의 AUPR
  (α₀ score), 5 seed 평균±표준편차. EDL·I-EDL·R-EDL·Re-EDL·FI-EDL 5개 메서드를 동일
  파이프라인에서 측정.
- **재현 프리셋**: `main_mnist` + `main_cifar10` + Re-EDL 별도 프리셋을 모두 실행하면
  4개 OOD 쌍이 한번에 채워집니다.
- **실행**:
  ```bash
  uv run python run.py preset main_mnist       # MNIST 행 (KMNIST, FMNIST) — 4개 메서드
  uv run python run.py preset main_cifar10     # CIFAR-10 행 (SVHN, CIFAR-100) — 4개 메서드
  uv run python run.py preset baseline_re_edl_mnist    # Re-EDL MNIST
  uv run python run.py preset baseline_re_edl_cifar10  # Re-EDL CIFAR-10 (lambda_prior=0.8)
  ```
  `main_mnist` / `main_cifar10` 프리셋은 4개 메서드(`edl_l1/i_edl/r_edl/fi_edl`) × 5 seed ×
  (train → eval)을 순차 수행합니다. Re-EDL은 CIFAR-10에서 lambda_prior=0.8이 필요하여
  별도 프리셋(`baseline_re_edl_cifar10`)으로 분리되어 있습니다.
  결과는 `runs/<exp>/seed_*/`에 OOD 메트릭으로 기록됩니다.
- **집계 스크립트**: `scripts/build_table_ood.py`
  ```bash
  uv run python scripts/build_table_ood.py --runs runs --out results/table_ood.csv
  ```

### Table 3 — CIFAR-10 Confidence Evaluation (`tab:iedl_style_conf`)

- **논문 내용**: CIFAR-10 오분류 검출 AUPR (Max.P, Max.α) + 분류 정확도.
  EDL·I-EDL·R-EDL·Re-EDL·FI-EDL 5개 메서드를 동일 파이프라인에서 재측정.
- **재현 프리셋**: `main_cifar10` 5개 메서드 × 5 seed eval 로그.
- **집계 스크립트**: `scripts/build_table_conf.py`
  ```bash
  uv run python scripts/build_table_conf.py --runs runs --out results/table_conf.csv
  ```
- **참고**: 문헌 기반 baseline은 재현 대상이 아니므로 표에는 그대로 옮겨 씁니다.

### Table 4 — AUCE Summary (`tab:auce_support`)

- **논문 내용**: MNIST·CIFAR-10에서 I-EDL Ref vs FI-EDL AUCE (낮을수록 좋음), 5 seed.
- **재현 프리셋**: `main_mnist`, `main_cifar10` (eval 단계에서 reliability bin이 기록됨).
- **집계 스크립트**: **현재 AUCE 집계는 자동화되어 있지 않습니다.** `scripts/build_table_ece.py`는
  ECE / accuracy / NLL / AURC만 출력합니다 (`src/metrics/ood_metrics.py`에도 AUCE 함수 없음).
  AUCE가 필요하면 `reliability_bins()` 출력으로부터 직접 계산하거나, 집계 스크립트를 확장해야
  합니다.
- **TODO**: `src/metrics/ood_metrics.py`에 `auce_from_reliability()` 추가 + `build_table_ece.py`가
  이를 집계하도록 수정.

### Figure — Reliability Diagrams (`fig:reliability`)

- **논문 내용**: MNIST/CIFAR-10에서 I-EDL Ref vs FI-EDL 신뢰도-정확도 reliability 다이어그램.
- **재현**:
  ```bash
  uv run python scripts/plot_reliability.py --runs runs --out results/reliability.pdf
  ```
- **전제**: `main_mnist`, `main_cifar10` 프리셋이 완료되어 `runs/**/metrics.json`에
  reliability bin이 쌓여 있어야 합니다.

### Figure — Training Dynamics (`fig:training_dynamics`)

- **논문 내용**: CIFAR-10 FI-EDL 학습 중 λ 평균/표준편차, Fisher trace 대체값, 손실 분해의
  epoch별 궤적.
- **재현**: 위와 동일한 `main_cifar10` 러닝 후
  ```bash
  uv run python scripts/plot_training_dynamics.py --runs runs --out results/dynamics.pdf
  ```
- **전제**: 스크립트는 Lightning CSVLogger 산출물 `runs/**/metrics.csv`를 직접 읽습니다.
  `fi_edl` 실행에서 λ 평균/표준편차, Fisher trace, 손실 구성이 `self.log(...)`로 기록되고
  있어야 합니다 (`src/models/lit_module.py` 참고).

### Table 5 — ECE Summary (`tab:ece_support`)

- **논문 내용**: MNIST·CIFAR-10 ECE 비교 (I-EDL Ref vs FI-EDL), 5 seed.
- **재현 프리셋**: `main_mnist`, `main_cifar10`.
- **집계 스크립트**: `scripts/build_table_ece.py`.

### Table 6 — Controller Ablation (`tab:controller_ablation`)

- **논문 내용**: CIFAR-10에서 게이트 설계를 비교 — constant / α₀ gate / FIM gate (no detach) /
  FIM gate (detach) / FI-EDL default.
- **재현 프리셋**: 변형별 3개 프리셋 + `main_cifar10` (default FI-EDL 행 겸용).
  ```bash
  uv run python run.py preset controller_constant
  uv run python run.py preset controller_alpha0_gate
  uv run python run.py preset controller_fim_nodetach
  # FIM gate (detach) / FI-EDL default 행은 main_cifar10 결과를 그대로 사용
  ```
- **변형 ↔ 프리셋 매핑**:
  | 논문 행 | 프리셋 |
  |---|---|
  | Constant gate | `controller_constant` |
  | α₀ gate | `controller_alpha0_gate` |
  | FIM gate (no detach) | `controller_fim_nodetach` |
  | **FI-EDL (detach / default)** | `main_cifar10` |

### Table 7 — β·γ Sensitivity (`tab:beta_gamma_sensitivity`)

- **논문 내용**: CIFAR-10에서 β, γ ∈ {0.5, 1.0, 2.0}의 3×3 sweep (총 9 설정), 5 seed.
- **재현 프리셋**: `bg_b{β}_g{γ}.yaml` (9개, β·γ ∈ {0.5, 1.0, 2.0}):
  ```bash
  for beta in 0.5 1.0 2.0; do
    for gamma in 0.5 1.0 2.0; do
      uv run python run.py preset bg_b${beta/./}_g${gamma/./}
    done
  done
  ```
  또는 개별 실행:
  ```bash
  uv run python run.py preset bg_b05_g05   # β=0.5, γ=0.5
  uv run python run.py preset bg_b10_g10   # β=1.0, γ=1.0 (default)
  # ... (나머지 7개 동일 패턴)
  ```
  프로토콜: VGG-16, 200 epochs, early stopping, val_split=5%, 5 seeds.

---

## 2. 프리셋 요약

| 프리셋 (`configs/paper/*.yaml`) | 대응 표/그림 | 실행 시간 규모 |
|---|---|---|
| `main_mnist` | Table 2 (MNIST 열), Table 4 (MNIST), Table 5 (MNIST), reliability (MNIST) | 4 메서드 × 5 seed × 200 epochs (convnet) |
| `main_cifar10` | Table 2 (CIFAR-10 열), Table 3, Table 4 (CIFAR-10), Table 5 (CIFAR-10), reliability (CIFAR-10), training dynamics, Table 6 default 행 | 4 메서드 × 5 seed × 200 epochs (vgg16) |
| `baseline_re_edl_mnist` | Table 2 (MNIST Re-EDL 행) | 1 메서드 × 5 seed × 200 epochs (convnet) |
| `baseline_re_edl_cifar10` | Table 2 (CIFAR-10 Re-EDL 행), Table 3 (Re-EDL 행) | 1 메서드 × 5 seed × 200 epochs (vgg16) |
| `controller_constant` | Table 6 — Constant gate 행 | 5 seed × 200 epochs (vgg16) |
| `controller_alpha0_gate` | Table 6 — α₀ gate 행 | 5 seed × 200 epochs (vgg16) |
| `controller_fim_nodetach` | Table 6 — FIM no-detach 행 | 5 seed × 200 epochs (vgg16) |
| `bg_b*_g*` (9개) | Table 7 β·γ sensitivity | 1 설정 × 5 seed × 200 epochs (vgg16), 9개 순차 실행 |

---

## 3. 집계 스크립트 ↔ 표 매핑

| 스크립트 | 대응 논문 표/그림 | 출력 |
|---|---|---|
| `scripts/build_table_ood.py` | Table 2 | `results/table_ood.csv` — (method, OOD target, score, AUPR mean±std) |
| `scripts/build_table_conf.py` | Table 3 | `results/table_conf.csv` — (method, Max.P AUPR, Max.α AUPR, Acc) |
| `scripts/build_table_ece.py` | Table 5 (ECE) | `results/table_ece.csv` — (method, dataset, ECE, accuracy, NLL, AURC) |
| `scripts/plot_reliability.py` | Fig. Reliability | `results/reliability.pdf` |
| `scripts/plot_training_dynamics.py` | Fig. Training Dynamics | `results/dynamics.pdf` |

입력 소스는 스크립트마다 다릅니다.

- `build_table_*.py` / `plot_reliability.py`: eval 시 `src/reporting/collector.py`가 기록한
  `runs/**/metrics.json`(+JSONL). 공통 로딩 로직은 `scripts/_loader.py`에 있습니다.
- `plot_training_dynamics.py`: 학습 중 Lightning CSVLogger가 쓴 `runs/**/metrics.csv`.

---

## 4. 현재 저장소에서 재현되지 않는 것

논문에 등장하지만 FI-EDL 재현 패키지에 포함하지 않은 항목:

- **Table 3의 문헌 baseline** (MC Dropout/KL-PN/RKL-PN/PostN/EDL): 외부 논문 인용값이라
  재실행 대상이 아닙니다.
- **Few-shot transfer 보조 분석**: 본문에서 "auxiliary analyses"로 분류된 실험으로,
  `FIM-EDL/external/iedl_official/` 기반 러너가 필요합니다. 필요 시 FIM-EDL 원본 저장소를
  사용하세요.
- **Controller ablation의 일괄 실행**: 변형별 프리셋 3개(`controller_constant`, `controller_alpha0_gate`, `controller_fim_nodetach`)를 개별 실행해야 합니다. 일괄 실행 스크립트는 없습니다.

---

## 5. 논문 표 재현 체크리스트

- [ ] `uv sync --dev`
- [ ] `uv run python run.py preset main_mnist`  →  MNIST 관련 행/그림 (4개 메서드)
- [ ] `uv run python run.py preset main_cifar10`  →  CIFAR-10 관련 행/그림 (4개 메서드)
- [ ] `uv run python run.py preset baseline_re_edl_mnist`  →  Re-EDL MNIST 행
- [ ] `uv run python run.py preset baseline_re_edl_cifar10`  →  Re-EDL CIFAR-10 행 (lambda_prior=0.8)
- [ ] `uv run python run.py preset controller_constant` / `controller_alpha0_gate` / `controller_fim_nodetach`  →  Table 6
- [ ] `bg_b*_g*` 9개 프리셋 실행  →  Table 7
- [ ] `build_table_ood.py` / `build_table_conf.py` / `build_table_ece.py` 순차 실행
- [ ] `plot_reliability.py` / `plot_training_dynamics.py` 실행
- [ ] 생성된 CSV/PDF를 논문 본문 수치와 대조
