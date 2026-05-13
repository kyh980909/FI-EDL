#!/bin/bash
# Re-run EDL and I-EDL using the actual I-EDL paper protocol (Deng 2023):
#   200 epochs, early stopping on val/acc (max) with patience=20,
#   anneal_epochs=10 default (paper uses min(1, epoch/10)).
# Run from ~/FI-EDL/
set -euo pipefail
PYTHON=./.venv/bin/python

COMMON_CIFAR=(
    dataset=cifar10
    backbone=vgg16
    trainer.max_epochs=200
    data.batch_size=64
    data.val_from_train=true
    data.val_split=0.05
    optimizer.lr=0.0005
    trainer.early_stopping=true
    trainer.early_stopping_monitor=val/acc
    trainer.early_stopping_mode=max
    trainer.early_stopping_patience=20
    logging.wandb.enabled=true
    logging.wandb.project=FI-EDL-CIFAR10
    logging.local_dir=runs/cifar10
)

COMMON_MNIST=(
    dataset=mnist
    backbone=convnet
    trainer.max_epochs=200
    data.batch_size=64
    data.val_from_train=true
    data.val_split=0.2
    optimizer.lr=0.001
    trainer.early_stopping=true
    trainer.early_stopping_monitor=val/acc
    trainer.early_stopping_mode=max
    trainer.early_stopping_patience=20
    logging.wandb.enabled=true
    logging.wandb.project=FI-EDL-MNIST
    logging.local_dir=runs/mnist
)

# Skip a (method, seed, dataset, backbone) combo if a completed retrain run
# already exists. We treat any train_<ts>_<dataset>_<backbone>*/summary.json
# created on/after this script's first invocation as "already retrained".
RETRAIN_MARK="$(date -d '2026-05-04 22:00' +%s 2>/dev/null || echo 0)"
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

echo "=== Re-training EDL (paper-match: monitor=val/acc, anneal_epochs=10) ==="
for seed in 0 1 2 3 4; do
    if already_done edl_l1 $seed cifar10 vgg16; then
        echo "[CIFAR-10] EDL seed=$seed (skip — already retrained)"
    else
        echo "[CIFAR-10] EDL seed=$seed"
        $PYTHON -m src.train experiment=edl_l1 seed=$seed "${COMMON_CIFAR[@]}"
    fi
    if already_done edl_l1 $seed mnist convnet; then
        echo "[MNIST] EDL seed=$seed (skip — already retrained)"
    else
        echo "[MNIST] EDL seed=$seed"
        $PYTHON -m src.train experiment=edl_l1 seed=$seed "${COMMON_MNIST[@]}"
    fi
done

echo "=== Re-training I-EDL (paper-match: monitor=val/acc, anneal_epochs=10) ==="
for seed in 0 1 2 3 4; do
    if already_done i_edl $seed cifar10 vgg16; then
        echo "[CIFAR-10] I-EDL seed=$seed (skip — already retrained)"
    else
        echo "[CIFAR-10] I-EDL seed=$seed"
        $PYTHON -m src.train experiment=i_edl seed=$seed "${COMMON_CIFAR[@]}"
    fi
    if already_done i_edl $seed mnist convnet; then
        echo "[MNIST] I-EDL seed=$seed (skip — already retrained)"
    else
        echo "[MNIST] I-EDL seed=$seed"
        $PYTHON -m src.train experiment=i_edl seed=$seed "${COMMON_MNIST[@]}"
    fi
done

echo "All re-training done."
