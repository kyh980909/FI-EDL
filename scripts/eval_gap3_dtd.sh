#!/usr/bin/env bash
# Gap 3 — Evaluate existing CIFAR-10 checkpoints on DTD (Textures) OOD.
# Forward-only, no retraining. Writes to runs_reeval/cifar10_dtd/.
# Targets the top 4 contenders: FI-EDL+SN (VGG-16), DAEDL, F-EDL, Re-EDL.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/gap3_dtd.log
mkdir -p logs
echo "=== Gap 3 DTD OOD eval started $(date) ===" | tee "$LOG"

eval_one () {
  local method="$1" bb="$2" seed="$3" ckpt="$4" variant_suffix="$5" extra="${6:-}"
  echo ">>> $method seed=$seed ($variant_suffix)" | tee -a "$LOG"
  uv run python -m src.eval experiment="$method" seed="$seed" \
    dataset=cifar10 backbone="$bb" \
    checkpoint="$ckpt" \
    'data.ood_list=[dtd]' \
    'eval.scores=[maxp,alpha0,vacuity,fisher]' \
    'eval.confidence_scores=[maxp]' \
    logging.wandb.enabled=false \
    logging.local_dir="runs_reeval/cifar10_dtd" \
    $extra 2>&1 | tee -a "$LOG"
}

# FI-EDL+SN (VGG-16) — variant suffix _fi_edl_sn; needs backbone_spectral_norm=true
for s in 0 1 2 3 4; do
  ck=$(ls runs/cifar10/fi_edl/seed_$s/train_*_fi_edl_sn/checkpoints/best.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && eval_one fi_edl vgg16 "$s" "$ck" "fi_edl_sn" "model.backbone_spectral_norm=true experiment.method_variant=fi_edl_sn"
done

# DAEDL — uses default daedl experiment
for s in 0 1 2 3 4; do
  ck=$(ls runs/cifar10/daedl/seed_$s/train_*_cifar10_vgg16/checkpoints/best.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && eval_one daedl vgg16 "$s" "$ck" "daedl"
done

# F-EDL
for s in 0 1 2 3 4; do
  ck=$(ls runs/cifar10/f_edl/seed_$s/train_*_cifar10_vgg16/checkpoints/best.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && eval_one f_edl vgg16 "$s" "$ck" "f_edl"
done

# Re-EDL
for s in 0 1 2 3 4; do
  ck=$(ls runs/cifar10/re_edl/seed_$s/train_*_cifar10_vgg16/checkpoints/best.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && eval_one re_edl vgg16 "$s" "$ck" "re_edl"
done

echo "=== Gap 3 DTD OOD eval complete $(date) ===" | tee -a "$LOG"
