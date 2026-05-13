#!/bin/bash
# Re-train FI-EDL on MNIST with paper-match protocol (val/acc max early-stop,
# anneal_epochs=10, ConvNet) to fix seed=0 collapse (89% acc) that produced
# the OOD AUPR std blow-up in Table 2.
# Run from ~/FI-EDL/
set -euo pipefail
PYTHON=./.venv/bin/python

COMMON_MNIST=(
    dataset=mnist
    backbone=convnet
    trainer.max_epochs=200
    data.batch_size=64
    data.num_workers=8
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

RETRAIN_MARK="$(date -d '2026-05-07 14:00' +%s 2>/dev/null || echo 0)"
already_done() {
    local seed=$1
    for s in runs/mnist/fi_edl/seed_${seed}/train_*_mnist_convnet/summary.json; do
        [ -f "$s" ] || continue
        local mt; mt=$(stat -c %Y "$s")
        if [ "$mt" -ge "$RETRAIN_MARK" ]; then
            return 0
        fi
    done
    return 1
}

echo "=== Re-training FI-EDL MNIST (paper-match: val/acc max, anneal_epochs=10) ==="
for seed in 0 1 2 3 4; do
    if already_done $seed; then
        echo "[MNIST] FI-EDL seed=$seed (skip — already retrained)"
    else
        echo "[MNIST] FI-EDL seed=$seed"
        $PYTHON -m src.train experiment=fi_edl seed=$seed "${COMMON_MNIST[@]}"
    fi
done

echo "All FI-EDL MNIST paper-match training done."
