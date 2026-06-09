#!/usr/bin/env bash
# Forward-only TinyImageNet OOD eval on CIFAR-100 ResNet-18 checkpoints.
# F-EDL (Yoon NeurIPS 2025) uses CIFAR-100 → TinyImageNet as the standard
# OOD pair on this benchmark. Runs after scripts/run_cifar100.sh completes.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/cifar100_tinyimagenet.log
mkdir -p logs
echo "=== CIFAR-100 → TinyImageNet eval started $(date) ===" | tee "$LOG"

eval_one () {
  local method="$1" seed="$2" ckpt="$3" extra="${4:-}"
  echo ">>> $method seed=$seed" | tee -a "$LOG"
  uv run python -m src.eval experiment="$method" seed="$seed" \
    dataset=cifar100 backbone=resnet18 \
    checkpoint="$ckpt" \
    'data.ood_list=[tinyimagenet,dtd]' \
    'eval.scores=[maxp,alpha0,vacuity,fisher]' \
    'eval.confidence_scores=[maxp]' \
    logging.wandb.enabled=false \
    logging.local_dir="runs_reeval/cifar100_tin" \
    $extra 2>&1 | tee -a "$LOG"
}

# FI-EDL+SN (needs backbone_spectral_norm=true)
for s in 0 1 2 3 4; do
  ck=$(ls runs/cifar100/fi_edl/seed_$s/train_*_resnet18_fi_edl_sn/checkpoints/best.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && eval_one fi_edl "$s" "$ck" "model.backbone_spectral_norm=true experiment.method_variant=fi_edl_sn"
done

for m in daedl f_edl re_edl; do
  for s in 0 1 2 3 4; do
    ck=$(ls runs/cifar100/$m/seed_$s/train_*_cifar100_resnet18/checkpoints/best.ckpt 2>/dev/null | head -1)
    [ -n "$ck" ] && eval_one "$m" "$s" "$ck"
  done
done

echo "=== CIFAR-100 → TinyImageNet eval complete $(date) ===" | tee -a "$LOG"
