#!/usr/bin/env bash
# FI-EDL gate-normalization sweep, Stage 1 (CIFAR-10, 2 seeds per config).
#
# Each config restores KL regularisation that the default (signal_norm=none)
# gate suppresses to ~5e-7. We compare three hypotheses:
#   A  fisher  + div_k   gamma=1.0  -> near-constant lambda (~0.23): "just turn KL on"
#   C  fisher  + batch_z gamma=0.5  -> amplified Fisher per-sample adaptivity
#   D  alpha0  + batch_z gamma=0.5  -> genuine adaptivity (confident<-less, uncertain<-more KL)
#
# Run from repo root:  bash scripts/run_fiedl_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/fiedl_sweep_stage1.log
mkdir -p logs
echo "=== FI-EDL sweep Stage 1 started $(date) ===" | tee -a "$LOG"

run_cfg () {
  local variant="$1" info="$2" norm="$3" gamma="$4"
  echo "" | tee -a "$LOG"
  echo ">>> [$variant] info_type=$info signal_norm=$norm gamma=$gamma  $(date)" | tee -a "$LOG"
  uv run python run.py preset sweep_fiedl_cifar10 \
    "loss.info_type=${info}" \
    "loss.signal_norm=${norm}" \
    "loss.gamma=${gamma}" \
    "loss.beta=1.0" \
    "experiment.method_variant=${variant}" 2>&1 | tee -a "$LOG"
}

run_cfg fi_edl_fisher_divk_g1   fisher  div_k   1.0
run_cfg fi_edl_fisher_bz_g05    fisher  batch_z 0.5
run_cfg fi_edl_alpha0_bz_g05    alpha0  batch_z 0.5

echo "" | tee -a "$LOG"
echo "=== FI-EDL sweep Stage 1 complete $(date) ===" | tee -a "$LOG"
