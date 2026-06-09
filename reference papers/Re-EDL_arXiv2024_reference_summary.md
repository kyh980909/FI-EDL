# Re-EDL: Revisiting Essential and Nonessential Settings of Evidential Deep Learning

**출처**: arXiv:2410.00393 (2024년 10월, under review)  
**저자**: Mengyuan Chen, Junyu Gao, Changsheng Xu  
**BibTeX 키**: `Chen2024b`  
**GitHub**: https://github.com/MengyuanChen21/Re-EDL  
**R-EDL의 확장판**

---

## 핵심 주장 (R-EDL 대비 더 근본적인 분석)

EDL의 비필수 설정을 더 포괄적으로 분석:

1. **Prior weight λ**: 클래스 수 K로 고정이 아닌 조정 가능한 하이퍼파라미터로 처리
2. **분산 최소화 항(ℒ_var)**: Dirac delta로 수렴시켜 과신(overconfidence) 악화
3. **KL divergence 정규화(ℒ_kl)**: 의도된 목적 외 방향으로 최적화되어 evidence magnitude 정보 손상

Re-EDL: 두 항(ℒ_var, ℒ_kl) 모두 제거, Dirichlet PDF의 기대값만 직접 최적화

---

## 핵심 실험 데이터 (TABLE XIII, CIFAR-10 Classical Setting)

### ECE (보정 오차) 비교:
| 방법 | ECE (15 bins) ↓ |
|------|----------------|
| EDL | 11.56 ± 0.93 |
| I-EDL | **44.35 ± 1.27** ← 독립 측정값 |
| R-EDL | 3.47 ± 0.31 |
| Re-EDL | 5.72 ± 0.32 |
| **FI-EDL (우리 논문)** | **2.95** ← 비교 기준 |

### 핵심 관찰:
- I-EDL이 EDL보다 ECE가 **4배 더 나쁨** (11.56 → 44.35)
- FIM을 fitting term에 적용한 I-EDL은 보정이 악화됨
- FI-EDL은 FIM을 regularization term에 적용하여 보정 개선

---

## FI-EDL 논문에서의 활용

1. **독립 검증**: I-EDL ECE ≈ 44%를 독립적으로 측정 → 우리 기준값 40.24% 신뢰성 확인
2. **비교 우위**: FI-EDL (2.95) < R-EDL (3.47) < Re-EDL (5.72) — calibration 최고
3. **서사 강화**: I-EDL의 FIM 적용 위치(fitting term)가 보정을 악화시킴 → FI-EDL의 위치(regularization term)가 올바름을 실증

---

## 인용 추가 위치 (논문 내)

- Related Work EDL 단락 끝: R-EDL, Re-EDL 언급
- ECE table 각주 또는 본문: "Re-EDL independently reports I-EDL ECE=44.35..."
