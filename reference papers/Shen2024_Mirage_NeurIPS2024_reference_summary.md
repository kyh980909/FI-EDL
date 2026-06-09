# Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?

**출처**: NeurIPS 2024  
**저자**: Maohao Shen, J. Jon Ryu, Soumya Ghosh, Yuheng Bu, Prasanna Sattigeri, Subhro Das, Gregory W. Wornell  
**BibTeX 키**: `Shen2024`

---

## 핵심 주장 (EDL에 대한 이론적 비판)

EDL의 불확실성 정량화(UQ) 능력이 실제로 신뢰할 수 없음을 이론적으로 증명:

- 광범위한 EDL 방법들이 최적해(optimal solution)에서 **에너지 기반 OOD 탐지기(energy-based OOD detector)** 처럼 동작함
- 즉, EDL이 학습하는 메타-분포(meta-distribution)는 에너지 기반 OOD 탐지기의 그것과 동일한 형태
- "EDL의 UQ 능력은 환상(mirage)일 수 있다"

---

## FI-EDL 논문에서의 활용

FI-EDL의 보정 개선(ECE 36→0.4 on MNIST, 40→2.95 on CIFAR-10)이 이 비판에 대한 **경험적 반론**:

> "FI-EDL의 근본적 보정 개선은 EDL이 단순한 에너지 기반 OOD 탐지기로만 동작한다는 해석과 일치하지 않는다. 기하학적 정규화를 통해 EDL이 진정한 UQ 방법으로 기능할 수 있음을 처음으로 실증한다."

### 인용 위치:
- Introduction: EDL에 대한 최근 비판 소개
- Related Work: EDL 단락 끝
- Discussion: FI-EDL의 의의 설명
