# 논문 실험 ↔ FI-EDL 매핑

논문(`Neurocomputing/FI-EDL_eng.tex`, `NeurIPS_2026/neurips_2026.tex`)의 실험 섹션과
이 저장소 구성 요소 간의 매핑을 한 곳에 정리한 문서입니다. 표/그림 단위로, 어떤 프리셋·
experiment·스크립트로 재현되는지 추적할 수 있게 했습니다.

---

## 0. 메서드 ↔ Hydra 설정 매핑

| 논문 표기 | Hydra `experiment` | `loss.name` | 설명 |
|---|---|---|---|
| EDL (λ = 1.0) | `edl_l1` | `edl_fixed` | Sensoy et al. 2018 베이스라인, KL 가중치 고정 |
| EDL (λ = 0.1) | `edl_l01` | `edl_fixed` | 중간 수준 KL 가중치 |
| EDL (λ = 0.001) | `edl_l0001` | `edl_fixed` | 약한 KL 가중치 |
| **I-EDL Ref** (Deng 2023 재구현) | `i_edl` | `i_edl` | Fisher-weighted MSE + log-det Fisher + linear KL anneal |
| **FI-EDL (Ours)** | `fi_edl` | `fi_edl` | Fisher 정보 기반 적응형 KL 게이트 λ(v) = β·exp(−γ·v) |

데이터셋·백본:

| 데이터셋 | `dataset` | `backbone` | OOD 대상 |
|---|---|---|---|
| MNIST | `mnist` | `convnet` | KMNIST, FashionMNIST |
| CIFAR-10 | `cifar10` | `resnet18` | SVHN, CIFAR-100 |

---

## 1. 표 · 그림별 재현 매핑

### Table 2 — OOD Detection Summary (`tab:iedl_style_ood`)

- **논문 내용**: MNIST→KMNIST, MNIST→FMNIST, CIFAR10→SVHN, CIFAR10→CIFAR100의 AUPR
  (Max.P, α₀ 각 2칸), 5 seed 평균±표준편차. I-EDL Ref와 FI-EDL 두 행을 동일 파이프라인에서
  측정.
- **재현 프리셋**: `main_mnist` + `main_cifar10` 두 개를 모두 실행하면 4개 OOD 쌍이 한번에
  채워집니다.
- **실행**:
  ```bash
  uv run python run.py preset main_mnist       # MNIST 행 (KMNIST, FMNIST)
  uv run python run.py preset main_cifar10     # CIFAR-10 행 (SVHN, CIFAR-100)
  ```
  각 프리셋은 5개 메서드(`edl_l1/edl_l01/edl_l0001/fi_edl/i_edl`) × 5 seed × (train → eval)을
  순차 수행하고 `runs/<exp>/seed_*/`에 OOD 메트릭을 기록합니다.
- **집계 스크립트**: `scripts/build_table_ood.py`
  ```bash
  uv run python scripts/build_table_ood.py --runs runs --out results/table_ood.csv
  ```
- **참고**: 논문 본문은 I-EDL Ref / FI-EDL 두 행만 리포트하지만, 프리셋은 EDL λ-sweep 3종도
  함께 돌려 보조 비교(Section 5.5 대응 등)가 가능하도록 구성돼 있습니다.

### Table 3 — CIFAR-10 Confidence Evaluation (`tab:iedl_style_conf`)

- **논문 내용**: CIFAR-10 오분류 검출 AUPR (Max.P, Max.α) + 분류 정확도. 외부 문헌 행
  (MC Dropout/KL-PN/RKL-PN/PostN/EDL/I-EDL)은 Deng 2023·Charpentier 2020 값을 그대로
  인용. 우리는 I-EDL Ref와 FI-EDL 2행만 재측정.
- **재현 프리셋**: `main_cifar10`의 `i_edl`, `fi_edl` 시드 5개 eval 로그.
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
- **재현 프리셋**: `controller_ablation`.
  ```bash
  uv run python run.py preset controller_ablation
  ```
  프리셋은 3 seed 기준이며, 논문 5 seed와 정렬하려면 `overrides`에 `seeds: [0,1,2,3,4]`를
  직접 추가하거나 프리셋을 복사해 수정하세요.
- **변형 ↔ `loss.*` 매핑** (모두 `experiment=fi_edl` 위에서 override):
  | 논문 행 | override |
  |---|---|
  | Constant gate | `loss.info_type=fisher loss.gate_type=constant` |
  | α₀ gate | `loss.info_type=alpha0 loss.gate_type=exp` |
  | FIM gate (no detach) | `loss.info_type=fisher loss.gate_type=exp loss.detach_weight=false` |
  | FIM gate (detach) | `loss.info_type=fisher loss.gate_type=exp loss.detach_weight=true` |
  | **FI-EDL (default)** | 기본값 (`info_type=fisher, gate_type=exp, detach_weight=true`) |
- **현재 상태**: `configs/paper/controller_ablation.yaml`은 FI-EDL 기본 설정 1행만 실행하도록
  정의되어 있습니다. 위 4개 변형을 함께 돌리려면 `methods:`를 확장하거나 프리셋을 여러 번
  다른 override와 호출해야 합니다 (`run.py preset controller_ablation --override '...'`).
  **TODO**: 4개 변형을 한꺼번에 돌리는 확장 프리셋 추가 여부 결정 필요.

### Table 7 — β·γ Sensitivity (`tab:beta_gamma_sensitivity`)

- **논문 내용**: CIFAR-10에서 β, γ ∈ {0.5, 1.0, 2.0}의 3×3 sweep (총 9 설정), 5 seed.
- **재현 프리셋**: **아직 프리셋 없음.** 수동 실행 권장:
  ```bash
  for beta in 0.5 1.0 2.0; do
    for gamma in 0.5 1.0 2.0; do
      for seed in 0 1 2 3 4; do
        uv run python -m src.train experiment=fi_edl dataset=cifar10 backbone=resnet18 \
            trainer.max_epochs=100 data.batch_size=64 optimizer.lr=0.0005 \
            loss.beta=$beta loss.gamma=$gamma seed=$seed \
            experiment.method_variant=fi_edl_b${beta}_g${gamma}
      done
    done
  done
  ```
  `method_variant`를 분리해야 집계 스크립트가 행을 구분해 묶습니다.
- **TODO**: `configs/paper/beta_gamma_sensitivity.yaml` 프리셋 추가 검토.

---

## 2. 프리셋 요약

| 프리셋 (`configs/paper/*.yaml`) | 대응 표/그림 | 실행 시간 규모 |
|---|---|---|
| `main_mnist` | Table 2 (MNIST 열), Table 3 부재, Table 4 (MNIST), Table 5 (MNIST), reliability (MNIST) | 5 메서드 × 5 seed × ~200 epochs (convnet) |
| `main_cifar10` | Table 2 (CIFAR-10 열), Table 3, Table 4 (CIFAR-10), Table 5 (CIFAR-10), reliability (CIFAR-10), training dynamics | 5 메서드 × 5 seed × 100 epochs (resnet18) |
| `controller_ablation` | Table 6 (일부) | 1 설정 × 3 seed × 100 epochs, 변형 직접 추가 필요 |
| (없음) | Table 7 β·γ sensitivity | 9 설정 × 5 seed, 수동 sweep |

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
- **β·γ sensitivity (Table 7)**: 위 2절처럼 sweep 수동 실행. 프리셋 추가는 TODO.
- **Controller ablation의 5개 변형 일괄 실행**: 현 프리셋은 기본 설정 1행만 포함.

---

## 5. 논문 표 재현 체크리스트

- [ ] `uv sync --dev`
- [ ] `uv run python run.py preset main_mnist`  →  MNIST 관련 행/그림
- [ ] `uv run python run.py preset main_cifar10`  →  CIFAR-10 관련 행/그림
- [ ] `uv run python run.py preset controller_ablation` (+ 4개 변형 override)  →  Table 6
- [ ] β·γ sweep 수동 실행  →  Table 7
- [ ] `build_table_ood.py` / `build_table_conf.py` / `build_table_ece.py` 순차 실행
- [ ] `plot_reliability.py` / `plot_training_dynamics.py` 실행
- [ ] 생성된 CSV/PDF를 논문 본문 수치와 대조
