#!/usr/bin/env bash
# Gap 1 — ResNet-18 CIFAR-10 sweep.
# Phase 1: FI-EDL+SN (the extension's recipe) — 5 seeds.
# Phase 2: DAEDL, F-EDL, Re-EDL at their published-SN settings — 15 seeds total.
# Total: 20 train + 20 eval, ~15–25h GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/resnet_cifar10.log
mkdir -p logs
echo "=== Gap 1 ResNet-18 CIFAR-10 sweep started $(date) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Phase 1/2: FI-EDL+SN on ResNet-18 (5 seeds) $(date)" | tee -a "$LOG"
uv run python run.py preset resnet_cifar10_fiedl_sn 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Phase 2/2: DAEDL, F-EDL, Re-EDL on ResNet-18 (15 runs) $(date)" | tee -a "$LOG"
uv run python run.py preset resnet_cifar10 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Gap 1 ResNet-18 CIFAR-10 sweep complete $(date) ===" | tee -a "$LOG"
