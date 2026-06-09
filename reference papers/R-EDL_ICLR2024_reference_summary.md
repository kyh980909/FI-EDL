# R-EDL: Relaxing Nonessential Settings of Evidential Deep Learning

**출처**: ICLR 2024 Spotlight  
**저자**: Mengyuan Chen, Junyu Gao, Changsheng Xu (Institute of Automation, Chinese Academy of Sciences)  
**BibTeX 키**: `Chen2024a`  
**GitHub**: https://github.com/MengyuanChen21/ICLR2024-REDL  
**arXiv**: 논문 기준 전신 논문, Re-EDL (arXiv:2410.00393)에서 계승

---

## 핵심 주장

EDL은 subjective logic 이론에 기반하지 않는 **비필수적(nonessential) 설정**을 포함하고 있다.
이를 제거(relaxing)함으로써 더 단순하고 효과적인 방법을 만들 수 있다.

구체적으로 R-EDL이 제거한 두 가지 비필수 설정:
1. **Prior weight λ 고정**: EDL은 λ를 클래스 수 K로 고정하지만, 이는 evidence의 비율과 크기 사이 균형에 영향을 미침
2. **KL divergence 정규화**: KL 정규화 항의 최적화 방향이 intended purpose를 벗어나 evidence magnitude의 정보를 손상시킴

---

## 실험 결과 (Re-EDL 논문 TABLE XIII에서 확인, CIFAR-10)

| 방법 | ECE (15 bins) ↓ | Brier Score ↓ | Mis Detect AUPR | OOD Detect AUPR (Mean) |
|------|----------------|---------------|-----------------|------------------------|
| Temp Scale | 1.06 ± 0.10 | 18.44 ± 0.49 | 98.89 ± 0.05 | 82.07 ± 2.23 |
| EDL | 11.56 ± 0.93 | 27.34 ± 0.71 | 98.74 ± 0.07 | 82.32 ± 0.98 |
| I-EDL | **44.35 ± 1.27** | 59.73 ± 1.31 | 98.71 ± 0.11 | 82.01 ± 1.47 |
| **R-EDL** | **3.47 ± 0.31** | 18.15 ± 0.50 | 98.98 ± 0.05 | 83.73 ± 1.07 |
| Re-EDL | 5.72 ± 0.32 | 14.95 ± 0.47 | 98.81 ± 0.05 | 85.46 ± 1.41 |

> **FI-EDL과 비교**: FI-EDL ECE = 2.95 (R-EDL 3.47보다 우수)

---

## FI-EDL 논문에서의 활용

- Related Work에서 R-EDL을 KL 정규화를 수정한 관련 연구로 언급
- ECE 비교에서 FI-EDL (2.95) > R-EDL (3.47)임을 보임
- Re-EDL 논문이 I-EDL ECE ≈ 44%를 독립 측정 → FI-EDL 기준값(40.24) 검증
