### Reviewer Q1. ECE computation details

현재 실험 파이프라인에서 ECE는 top-1 confidence 기반 standard ECE로 계산했습니다. 구체적으로 각 샘플의 confidence는 max(prob)이고, correctness는 argmax(prob) == y로 정의했습니다. binning은 n_bins=15의 equal-width bin이며 [0,1] 구간을 균등 분할합니다. adaptive binning은 사용하지 않았고, label smoothing도 사용하지 않았습니다. calibration protocol은 기본적으로 none이며, 본 논문 표에 사용된 값들은 temperature scaling 없이 얻은 결과입니다. 구현은 FIM-EDL/src/metrics/ood_metrics.py, FIM-EDL/src/eval.py, FIM-EDL/configs/config.yaml에 있습니다.

CIFAR-10의 40.24 -> 2.95는 비율이 아니라 percent 표기라서, raw ECE로는 0.402361 -> 0.029490입니다. 이는 FIM-EDL/results/additional_experiments/summary_mean_std.csv의 iedl_ref,cifar10,conf_eval 및 info_edl,cifar10,conf_eval 행과 일치합니다. 다만 현재 FIM-EDL 파이프라인에는 reliability diagram/AUCE를 생성하는 코드가 없습니다. AUCE도 아직 계산하지 않았습니다. 따라서 이 질문에는 “현재 논문 수치는 15-bin top-1 ECE 기준이며, reliability diagram과 AUCE는 revision에서 추가하겠다”가 정확합니다.

### Reviewer Q2. Baseline tuning

고정-λ EDL baseline은 per-metric tuning으로 고른 것이 아니라 사전 정의된 실험 설정으로 돌렸습니다. 기본 experiment config는 λ ∈ {1.0, 0.1, 0.001}이고, 각 config는 FIM-EDL/configs/experiment/edl_l1.yaml, FIM-EDL/configs/experiment/edl_l01.yaml, FIM-EDL/configs/experiment/edl_l0001.yaml에 있습니다. 고정 EDL은 annealed KL을 사용하며, 실제 loss는 λ \* min(1, epoch/anneal_epochs)입니다. 구현은 FIM-EDL/src/losses/edl_fixed.py에 있습니다.

논문 메인 CIFAR 비교에서는 paper_iedl_cifar_ref preset이 loss.anneal_epochs=200으로 override되어 matched training protocol을 맞췄습니다. 이는 FIM-EDL/configs/preset/paper_iedl_cifar_ref.yaml에서 확인됩니다. 다만 reviewer 지적대로, “best-per-metric fixed-λ EDL multi-seed sweep”은 아직 없습니다. 현재 추가 결과는 CIFAR에서 λ=1.0 matched 5-seed는 있고, λ=0.1, 0.001은 일부가 n=2 또는 별도 파이프라인입니다. 따라서 hyperparameter artifact 우려를 완전히 닫으려면 multi-seed fixed-λ sweep을 더 해야 합니다.

### Reviewer Q3. Robustness to β, γ

현재 multi-seed sensitivity는 완료되지 않았습니다. 다만 β,γ ∈ {0.5,1.0,2.0} 3x3 grid를 돌리는 스크립트는 이미 있고, FIM-EDL/scripts/paper/run_additional_cifar_experiments.py에서 확인됩니다. 현재 저장된 결과는 기본 info_edl만 5-seed이고, sensitivity grid 행들은 대부분 n=1입니다. 즉 reviewer가 요구한 “multi-seed β/γ sensitivity”에는 아직 미달입니다.

다만 controller의 operating regime 자체는 로그로 추적 가능합니다. loss에서 lambda_mean, lambda_std, lambda_min, lambda_max, fim_trace_mean을 기록하고 있고, Lightning module도 epoch 단위로 이를 기록합니다. 관련 구현은 FIM-EDL/src/losses/edl_info_adaptive.py, FIM-EDL/src/models/lit_module.py에 있습니다. 따라서 답변은 “trajectory는 로깅되고 있으나, 논문에는 아직 요약 figure로 넣지 못했다”가 맞습니다.

### Reviewer Q4. Architectural breadth and overhead

현재 메인 실험 backbone은 MNIST에서는 convnet, CIFAR-10에서는 vgg16입니다. 관련 preset은 FIM-EDL/configs/preset/paper_iedl_mnist_ref.yaml, FIM-EDL/configs/preset/paper_iedl_cifar_ref.yaml에 있습니다. CIFAR-10 augmentation은 기본 파이프라인에서 RandomHorizontalFlip + RandomCrop(32, padding=4) + Normalize이고, MNIST는 Resize(32) + Grayscale(3ch) + RandomCrop(padding=2) + Normalize입니다. 구현은 FIM-EDL/src/data/adapters/cifar10_adapter.py, FIM-EDL/src/data/adapters/mnist_adapter.py입니다. 다만 I-EDL official-style CIFAR rerun은 별도 preset에서 normalize=false, random_rotation_degrees=15, val_use_train_transform=true를 사용했습니다.

WRN-28-10은 코드베이스에 존재하지만 few-shot protocol에만 연결되어 있고, 현재 OOD/main CIFAR 실험에는 넣지 않았습니다. 또한 wall-clock overhead는 아직 계측하지 않았습니다. 구현 관점에서 FI-EDL의 추가 계산은 batch마다 torch.polygamma(1, alpha)와 torch.polygamma(1, alpha.sum(...)), 그리고 elementwise exp 한 번이라 계산 복잡도는 대략 O(BK) 추가입니다. 하지만 reviewer 질문은 “실측 wall-clock”이므로, 현재 상태에서는 측정치를 제공하지 못했다고 답하는 것이 맞습니다.

### Reviewer Q5. Competing EDL variants

현재 코드와 결과에는 DAEDL이나 DIP-EDL이 없습니다. 따라서 공정한 동일 파이프라인 비교도 아직 없습니다. 이 질문에는 정직하게 “현재 비교는 EDL, I-EDL reference, 그리고 내부 ablation에 한정되며, DAEDL/DIP-EDL은 revision에서 추가할 필요가 있다”고 답해야 합니다.

### Reviewer Q6. AUROC and prevalence sensitivity

OOD에 대해서는 이미 AUROC와 FPR95를 계산하고 저장합니다. 구현은 FIM-EDL/src/metrics/ood_metrics.py, FIM-EDL/src/eval.py에 있고, 요약 결과는 FIM-EDL/results/additional_experiments/summary_mean_std.csv에 있습니다. 예를 들어 CIFAR10→SVHN alpha0 기준으로 I-EDL Ref는 AUROC 0.8812, FPR95 0.2972, FI-EDL은 AUROC 0.9290, FPR95 0.2296입니다. CIFAR10→CIFAR100 alpha0는 0.8239/0.4854 -> 0.8648/0.4504입니다.

AUPR positive-class convention도 코드로 명확합니다. OOD evaluation에서는 y_true=[0 for ID, 1 for OOD]이므로 positive class는 OOD입니다. 또 \_ood_score_from_raw에서 maxp와 alpha0는 부호를 뒤집어 “값이 클수록 OOD-like”가 되게 맞춥니다. 다만 misclassification/confidence evaluation은 현재 AUROC/FPR95를 계산하지 않으며, AUPR도 positive class가 misclassified가 아니라 correct입니다. \_confidence_aupr가 labels = correct를 쓰기 때문입니다. 따라서 reviewer가 misclassification-positive convention을 묻는다면, 현재 표의 conf_eval AUPR은 엄밀히 말해 “correctness ranking AUPR”입니다. 이 점은 rebuttal에서 반드시 명시하는 게 좋습니다.

### Reviewer Q7. Mechanistic analysis

현재 mechanistic evidence로는 alpha0 gate ablation과 FIM/no-detach, constant gate 비교까지는 있습니다. 5-seed 결과는 FIM-EDL/results/additional_experiments/summary_mean_std.csv에 있습니다. 예를 들어 CIFAR10→SVHN alpha0 AUPR은 alpha0 gate 0.8965, FI-EDL 0.9593이고, CIFAR10→CIFAR100 alpha0 AUPR은 0.8076 -> 0.8404입니다. 즉 full FIM trace가 단순 alpha0 monotone gate보다 강하다는 정성적 근거는 이미 있습니다.

다만 reviewer가 요구한 더 직접적인 정량화, 즉 vFIM-alpha0 상관 분석이나 alternative monotone gate λ = β/(1 + γ α0^{-1}) 실험은 현재 없습니다. 코드상 대안 gate는 FIM-EDL repo에는 exp와 constant만 있고, Fisher-EDL 쪽 별도 실험용 코드에는 inv와 sigmoid gate가 있습니다. 하지만 본 논문 파이프라인 결과로는 아직 연결되지 않았습니다. 따라서 이 질문에는 “현재는 α0-only gate ablation까지 제공 가능하고, correlation/alternative monotone gate는 추가 분석이 필요하다”고 답하는 것이 정확합니다.
