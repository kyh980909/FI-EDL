#!/usr/bin/env bash
# (B) spectral-norm backbone FI-EDL — MNIST + CIFAR-10, 5 seeds each.
#
# Tests whether SN's bi-Lipschitz constraint on the backbone closes the
# MNIST OOD gap (FI-EDL 6-7th vs DAEDL 99.98 via density). Loss settings
# are gate-off baseline (signal_norm=none, kl_target_mask=false), so any
# improvement is attributable to SN itself, not the controller (which we
# already showed is empirically inert).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/fiedl_sn.log
mkdir -p logs
echo "=== FI-EDL +SN training started $(date) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Stage 1/2: MNIST (5 seeds, ConvNet, ~1.5h)  $(date)" | tee -a "$LOG"
uv run python run.py preset sn_fiedl_mnist 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Stage 2/2: CIFAR-10 (5 seeds, VGG16, ~5h)  $(date)" | tee -a "$LOG"
uv run python run.py preset sn_fiedl_cifar10 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== FI-EDL +SN training complete $(date) ===" | tee -a "$LOG"
