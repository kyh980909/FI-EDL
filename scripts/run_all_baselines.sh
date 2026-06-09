#!/usr/bin/env bash
# Chained baseline runner: MNIST → CIFAR-10 → Re-EDL CIFAR-10 → aggregation.
# Designed to be invoked once and left running.
set -euo pipefail

cd "$(dirname "$0")/.."

LOG=logs/chain_full.log
mkdir -p logs results

ts() { date '+%Y-%m-%dT%H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOG"; }

log "=== Chain start ==="

# Step 0 — wait for any in-flight MNIST run.py process to finish.
while pgrep -f "run.py preset comparison_all_mnist" >/dev/null; do
  log "Waiting for in-flight MNIST run.py to finish..."
  sleep 60
done

log "Step 1 — comparison_all_mnist (resumes any incomplete seeds)"
uv run python run.py preset comparison_all_mnist >> logs/mnist_full.log 2>&1
log "Step 1 done"

log "Step 2 — comparison_all_cifar10"
uv run python run.py preset comparison_all_cifar10 > logs/cifar10_full.log 2>&1
log "Step 2 done"

log "Step 3 — baseline_re_edl_cifar10 (lambda_prior=0.8)"
uv run python run.py preset baseline_re_edl_cifar10 > logs/cifar10_re_edl.log 2>&1
log "Step 3 done"

log "Step 4 — aggregate tables"
uv run python scripts/build_table_ece.py  --runs runs --out results/table1_extended.csv
uv run python scripts/build_table_ood.py  --runs runs --out results/table2_extended.csv
uv run python scripts/build_table_conf.py --runs runs --out results/table3_extended.csv
log "Step 4 done"

log "Step 5 — fill markdown report from CSVs"
uv run python scripts/fill_report_md.py
log "Step 5 done"

log "=== Chain complete ==="
