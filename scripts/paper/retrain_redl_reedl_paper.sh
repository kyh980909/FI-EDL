#!/bin/bash
# Re-train R-EDL and Re-EDL using their own paper protocols (Chen 2024 ICLR / arXiv).
#   Source: https://github.com/MengyuanChen21/ICLR2024-REDL
#           https://github.com/MengyuanChen21/Re-EDL
# Differences from old-cycle (I-EDL paper protocol):
#   - CIFAR-10 lr: 5e-4 → 1e-4
#   - Re-EDL CIFAR-10 lambda_prior: 0.1 → 0.8
#   - MNIST max_epochs: 200 → 60
#   - early_stopping: true (val/loss) → false (best ckpt by val/acc)
# Run from ~/FI-EDL/
set -euo pipefail
PYTHON=./.venv/bin/python

COMMON_CIFAR=(
    dataset=cifar10
    backbone=vgg16
    trainer.max_epochs=200
    data.batch_size=64
    data.num_workers=8
    data.val_from_train=true
    data.val_split=0.05
    optimizer.lr=0.0001
    trainer.early_stopping=false
    logging.wandb.enabled=true
    logging.wandb.project=FI-EDL-CIFAR10
    logging.local_dir=runs/cifar10
)

COMMON_MNIST=(
    dataset=mnist
    backbone=convnet
    trainer.max_epochs=60
    data.batch_size=64
    data.num_workers=8
    data.val_from_train=true
    data.val_split=0.2
    optimizer.lr=0.001
    trainer.early_stopping=false
    logging.wandb.enabled=true
    logging.wandb.project=FI-EDL-MNIST
    logging.local_dir=runs/mnist
)

RETRAIN_MARK="$(date -d '2026-05-05 13:00' +%s 2>/dev/null || echo 0)"
already_done() {
    local method=$1 seed=$2 dataset=$3 backbone=$4
    for s in runs/${dataset}/${method}/seed_${seed}/train_*_${dataset}_${backbone}*/summary.json; do
        [ -f "$s" ] || continue
        local mt; mt=$(stat -c %Y "$s")
        if [ "$mt" -ge "$RETRAIN_MARK" ]; then
            return 0
        fi
    done
    return 1
}

echo "=== Re-training R-EDL (paper-match: lambda_prior=0.1, lr=1e-4 CIFAR) ==="
for seed in 0 1 2 3 4; do
    if already_done r_edl $seed cifar10 vgg16; then
        echo "[CIFAR-10] R-EDL seed=$seed (skip — already retrained)"
    else
        echo "[CIFAR-10] R-EDL seed=$seed"
        $PYTHON -m src.train experiment=r_edl seed=$seed loss.lambda_prior=0.1 "${COMMON_CIFAR[@]}"
    fi
    if already_done r_edl $seed mnist convnet; then
        echo "[MNIST] R-EDL seed=$seed (skip — already retrained)"
    else
        echo "[MNIST] R-EDL seed=$seed"
        $PYTHON -m src.train experiment=r_edl seed=$seed loss.lambda_prior=0.1 "${COMMON_MNIST[@]}"
    fi
done

echo "=== Re-training Re-EDL (paper-match: CIFAR lambda_prior=0.8, MNIST=0.1, lr=1e-4 CIFAR) ==="
for seed in 0 1 2 3 4; do
    if already_done re_edl $seed cifar10 vgg16; then
        echo "[CIFAR-10] Re-EDL seed=$seed (skip — already retrained)"
    else
        echo "[CIFAR-10] Re-EDL seed=$seed"
        $PYTHON -m src.train experiment=re_edl seed=$seed loss.lambda_prior=0.8 "${COMMON_CIFAR[@]}"
    fi
    if already_done re_edl $seed mnist convnet; then
        echo "[MNIST] Re-EDL seed=$seed (skip — already retrained)"
    else
        echo "[MNIST] Re-EDL seed=$seed"
        $PYTHON -m src.train experiment=re_edl seed=$seed loss.lambda_prior=0.1 "${COMMON_MNIST[@]}"
    fi
done

echo "All R-EDL/Re-EDL paper-match training done."
