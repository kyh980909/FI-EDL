#!/usr/bin/env bash
# CIFAR-100-as-ID sweep — main_cifar100 protocol.
# Phase 1: FI-EDL+SN (our recipe) — 5 seeds
# Phase 2: 6 baseline methods (EDL, I-EDL, R-EDL, Re-EDL, DAEDL, F-EDL) — 30 runs
# Estimated total: 35 train + 35 eval, ~50-60h GPU at VGG-16.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/cifar100.log
mkdir -p logs
echo "=== CIFAR-100 sweep started $(date) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Phase 1/2: FI-EDL+SN on CIFAR-100 (5 seeds) $(date)" | tee -a "$LOG"
uv run python run.py preset main_cifar100_fiedl_sn 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo ">>> Phase 2/2: 6 baselines on CIFAR-100 (30 runs) $(date)" | tee -a "$LOG"
uv run python run.py preset main_cifar100 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== CIFAR-100 sweep complete $(date) ===" | tee -a "$LOG"
