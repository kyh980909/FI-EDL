#!/usr/bin/env bash
# Reproduces all paper experiments (NeurIPS 2026 submission).
# Run from the FI-EDL/ root directory with the venv activated.
#
# Total estimated time: ~132 GPU-hours on a single H100 80GB.
# Wall-clock time per single-seed:
#   EDL  ~35 min/seed (MNIST), ~76 min/seed (CIFAR-10)
#   FI-EDL ~55 min/seed (MNIST), ~73 min/seed (CIFAR-10)
#
# Usage:
#   bash scripts/reproduce_paper.sh
# or using uv:
#   uv run bash scripts/reproduce_paper.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"

run_preset() {
  local name=$1
  echo "============================================"
  echo "Running preset: $name"
  echo "============================================"
  $PYTHON run.py preset "$name"
}

# ── Main results (Tables 1, 2) ─────────────────────────────────────────────
run_preset main_mnist           # EDL, I-EDL, R-EDL, FI-EDL on MNIST
run_preset main_cifar10         # EDL, I-EDL, R-EDL, FI-EDL on CIFAR-10
run_preset baseline_re_edl_mnist    # Re-EDL on MNIST  (lambda_prior=0.1)
run_preset baseline_re_edl_cifar10  # Re-EDL on CIFAR-10 (lambda_prior=0.8)

# ── Controller ablation (Table 3 / tab:ablation) ──────────────────────────
run_preset controller_constant
run_preset controller_alpha0_gate
run_preset controller_fim_nodetach
# Note: FI-EDL default row reuses main_cifar10 results (already run above).

# ── β·γ sensitivity (Appendix / tab:sensitivity) ──────────────────────────
for preset in \
  bg_b05_g05 bg_b05_g10 bg_b05_g20 \
  bg_b10_g05 bg_b10_g10 bg_b10_g20 \
  bg_b20_g05 bg_b20_g10 bg_b20_g20; do
  run_preset "$preset"
done

echo ""
echo "All presets complete. Aggregate results with:"
echo "  python scripts/paper/extract_v4_tables.py"
