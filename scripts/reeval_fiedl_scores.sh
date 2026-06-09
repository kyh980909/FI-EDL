#!/usr/bin/env bash
# Re-evaluate existing FI-EDL checkpoints with expanded scores (forward only,
# no training) to test whether alpha0 / fisher discriminate misclassification
# (conf_eval) and OOD better than the maxp/maxalpha used in the report.
# Writes to runs_reeval/ to avoid polluting the existing aggregation.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/reeval_misclass.log
mkdir -p logs
echo "=== FI-EDL score re-eval started $(date) ===" | tee "$LOG"

eval_one () {
  local ds="$1" bb="$2" seed="$3" ckpt="$4"
  echo ">>> $ds seed=$seed" | tee -a "$LOG"
  uv run python -m src.eval experiment=fi_edl seed="$seed" \
    dataset="$ds" backbone="$bb" \
    checkpoint="$ckpt" \
    eval.confidence_scores='[maxp,maxalpha,alpha0,fisher]' \
    eval.scores='[maxp,alpha0,vacuity,fisher]' \
    logging.wandb.enabled=false \
    logging.local_dir="runs_reeval/$ds" 2>&1 | tee -a "$LOG"
}

for s in 0 1 2 3 4; do
  ck=$(ls runs/mnist/fi_edl/seed_$s/train_*/checkpoints/best.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && eval_one mnist convnet "$s" "$ck"
done
for s in 0 1 2 3 4; do
  ck=$(ls runs/cifar10/fi_edl/seed_$s/train_*/checkpoints/best.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] && eval_one cifar10 vgg16 "$s" "$ck"
done

echo "=== FI-EDL score re-eval complete $(date) ===" | tee -a "$LOG"
