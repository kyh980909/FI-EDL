#!/usr/bin/env bash
# FI-EDL gate-normalization sweep, Stage 2 — masked-KL (kl_target_mask=true).
#
# Stage 1 showed that turning the gate ON with the default full-alpha KL
# destroys calibration (ECE 5%→71%) because the regulariser pulls the target
# class evidence toward 1/K. Masking the target class (alpha_hat = alpha·(1-y)+y)
# is standard EDL practice; it penalises wrong-class evidence only.
#
# Tested configs (CIFAR-10, 2 seeds each):
#   A1  fisher  + div_k   gamma=1.0  masked=true  -> ECE recovery vs Stage-1 fisher_divk_g1
#   A2  alpha0  + batch_z gamma=0.5  masked=true  -> adaptive + masked: the real contender
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/fiedl_masked_sweep.log
mkdir -p logs
echo "=== FI-EDL masked-KL sweep started $(date) ===" | tee "$LOG"

run_cfg () {
  local variant="$1" info="$2" norm="$3" gamma="$4"
  echo "" | tee -a "$LOG"
  echo ">>> [$variant] info_type=$info signal_norm=$norm gamma=$gamma masked=true  $(date)" | tee -a "$LOG"
  uv run python run.py preset sweep_fiedl_cifar10 \
    "loss.info_type=${info}" \
    "loss.signal_norm=${norm}" \
    "loss.gamma=${gamma}" \
    "loss.beta=1.0" \
    "loss.kl_target_mask=true" \
    "experiment.method_variant=${variant}" 2>&1 | tee -a "$LOG"
}

run_cfg fi_edl_fisher_divk_g1_masked   fisher  div_k   1.0
run_cfg fi_edl_alpha0_bz_g05_masked    alpha0  batch_z 0.5

echo "" | tee -a "$LOG"
echo "=== FI-EDL masked-KL sweep complete $(date) ===" | tee -a "$LOG"
