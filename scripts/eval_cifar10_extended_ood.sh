#!/usr/bin/env bash
# Forward-only eval on existing CIFAR-10/VGG-16 checkpoints over an extended
# OOD set: SVHN + CIFAR-100 + DTD (already in main results) + GTSRB (new,
# matches Re-EDL Table 2). LFWPeople/Food101/Places365 deferred (deprecated
# torchvision download / heavy size).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/cifar10_extended_ood.log
mkdir -p logs
echo "=== CIFAR-10 extended OOD eval started $(date) ===" | tee "$LOG"

eval_one () {
  local method="$1" seed="$2" ckpt="$3" extra="${4:-}"
  echo ">>> $method seed=$seed" | tee -a "$LOG"
  uv run python -m src.eval experiment="$method" seed="$seed" \
    dataset=cifar10 backbone=vgg16 \
    checkpoint="$ckpt" \
    'data.ood_list=[gtsrb]' \
    'eval.scores=[maxp,alpha0,vacuity,fisher]' \
    'eval.confidence_scores=[maxp]' \
    logging.wandb.enabled=false \
    logging.local_dir="runs_reeval/cifar10_extended" \
    $extra 2>&1 | tee -a "$LOG"
}

# FI-EDL+SN (VGG-16)
for s in 0 1 2 3 4; do
  ck=$(ls runs/cifar10/fi_edl/seed_$s/train_*_fi_edl_sn/checkpoints/best.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && eval_one fi_edl "$s" "$ck" "model.backbone_spectral_norm=true experiment.method_variant=fi_edl_sn"
done

# DAEDL, F-EDL, Re-EDL (default published settings)
for m in daedl f_edl re_edl; do
  for s in 0 1 2 3 4; do
    ck=$(ls runs/cifar10/$m/seed_$s/train_*_cifar10_vgg16/checkpoints/best.ckpt 2>/dev/null | head -1)
    [ -n "$ck" ] && eval_one "$m" "$s" "$ck"
  done
done

echo "=== CIFAR-10 extended OOD eval complete $(date) ===" | tee -a "$LOG"
