#!/bin/bash
# Eval script for all paper experiment checkpoints
# Run from ~/projects/FI-EDL/
set -euo pipefail
PYTHON=./.venv/bin/python

run_eval() {
    local exp=$1 variant=$2 backbone=$3 dataset=$4 local_dir=$5 seed=$6 ckpt=$7
    local extra=""
    [ "$variant" != "$exp" ] && extra="experiment.method_variant=$variant"
    $PYTHON -m src.eval \
        experiment="$exp" \
        dataset="$dataset" \
        backbone="$backbone" \
        seed="$seed" \
        checkpoint="$ckpt" \
        data.root=/home/user/FI-EDL/data \
        logging.local_dir="$local_dir" \
        logging.wandb.enabled=false \
        ${extra:+"$extra"}
    echo "[OK] $exp/$variant seed=$seed $dataset"
}

echo "Starting 110 eval jobs..."

echo '=== CIFAR10 ==='
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_0/train_20260428T222651_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_1/train_20260428T224355_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_2/train_20260428T230013_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_3/train_20260428T231639_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_4/train_20260428T233302_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260429T123149_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_constant' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260429T185334_cifar10_vgg16_fi_edl_constant/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_alpha0_gate' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260429T201642_cifar10_vgg16_fi_edl_alpha0_gate/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_fim_nodetach' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260430T024300_cifar10_vgg16_fi_edl_fim_nodetach/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260430T083929_cifar10_vgg16_cifar_bg_b0.5_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260430T145700_cifar10_vgg16_cifar_bg_b0.5_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260430T202606_cifar10_vgg16_cifar_bg_b0.5_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260501T014352_cifar10_vgg16_cifar_bg_b1.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260501T082400_cifar10_vgg16_cifar_bg_b1.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260501T144533_cifar10_vgg16_cifar_bg_b1.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260501T201458_cifar10_vgg16_cifar_bg_b2.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260502T031911_cifar10_vgg16_cifar_bg_b2.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_0/train_20260502T085000_cifar10_vgg16_cifar_bg_b2.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260429T135826_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_constant' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260429T191016_cifar10_vgg16_fi_edl_constant/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_alpha0_gate' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260429T212052_cifar10_vgg16_fi_edl_alpha0_gate/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_fim_nodetach' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260430T035205_cifar10_vgg16_fi_edl_fim_nodetach/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260430T093447_cifar10_vgg16_cifar_bg_b0.5_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260430T154849_cifar10_vgg16_cifar_bg_b0.5_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260430T212241_cifar10_vgg16_cifar_bg_b0.5_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260501T031043_cifar10_vgg16_cifar_bg_b1.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260501T095027_cifar10_vgg16_cifar_bg_b1.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260501T155701_cifar10_vgg16_cifar_bg_b1.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260501T211703_cifar10_vgg16_cifar_bg_b2.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260502T041953_cifar10_vgg16_cifar_bg_b2.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_1/train_20260502T100107_cifar10_vgg16_cifar_bg_b2.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260429T151430_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_constant' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260429T192700_cifar10_vgg16_fi_edl_constant/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_alpha0_gate' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260429T231726_cifar10_vgg16_fi_edl_alpha0_gate/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_fim_nodetach' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260430T050824_cifar10_vgg16_fi_edl_fim_nodetach/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260430T103816_cifar10_vgg16_cifar_bg_b0.5_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260430T173050_cifar10_vgg16_cifar_bg_b0.5_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260430T224106_cifar10_vgg16_cifar_bg_b0.5_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260501T042908_cifar10_vgg16_cifar_bg_b1.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260501T110726_cifar10_vgg16_cifar_bg_b1.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260501T170027_cifar10_vgg16_cifar_bg_b1.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260501T223012_cifar10_vgg16_cifar_bg_b2.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260502T055142_cifar10_vgg16_cifar_bg_b2.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_2/train_20260502T111804_cifar10_vgg16_cifar_bg_b2.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260429T161122_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_constant' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260429T194338_cifar10_vgg16_fi_edl_constant/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_alpha0_gate' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260430T002148_cifar10_vgg16_fi_edl_alpha0_gate/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_fim_nodetach' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260430T060752_cifar10_vgg16_fi_edl_fim_nodetach/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260430T121620_cifar10_vgg16_cifar_bg_b0.5_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260430T182813_cifar10_vgg16_cifar_bg_b0.5_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260430T233332_cifar10_vgg16_cifar_bg_b0.5_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260501T054653_cifar10_vgg16_cifar_bg_b1.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260501T120310_cifar10_vgg16_cifar_bg_b1.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260501T181008_cifar10_vgg16_cifar_bg_b1.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260502T000707_cifar10_vgg16_cifar_bg_b2.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260502T064629_cifar10_vgg16_cifar_bg_b2.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_3/train_20260502T122140_cifar10_vgg16_cifar_bg_b2.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260429T173916_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_constant' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260429T200007_cifar10_vgg16_fi_edl_constant/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_alpha0_gate' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260430T013003_cifar10_vgg16_fi_edl_alpha0_gate/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl_fim_nodetach' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260430T071456_cifar10_vgg16_fi_edl_fim_nodetach/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260430T131912_cifar10_vgg16_cifar_bg_b0.5_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260430T191915_cifar10_vgg16_cifar_bg_b0.5_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b0.5_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260501T004413_cifar10_vgg16_cifar_bg_b0.5_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260501T065804_cifar10_vgg16_cifar_bg_b1.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260501T133241_cifar10_vgg16_cifar_bg_b1.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b1.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260501T190544_cifar10_vgg16_cifar_bg_b1.0_g2.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g0.5' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260502T020135_cifar10_vgg16_cifar_bg_b2.0_g0.5/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g1.0' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260502T074554_cifar10_vgg16_cifar_bg_b2.0_g1.0/checkpoints/best.ckpt'
run_eval 'fi_edl' 'cifar_bg_b2.0_g2.0' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/fi_edl/seed_4/train_20260502T131613_cifar10_vgg16_cifar_bg_b2.0_g2.0/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_0/train_20260428T234934_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_1/train_20260429T000729_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_2/train_20260429T002553_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_3/train_20260429T004400_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_4/train_20260429T010208_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_0/train_20260429T012021_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_1/train_20260429T023028_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_2/train_20260429T032111_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_3/train_20260429T044300_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_4/train_20260429T053813_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_0/train_20260429T065136_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_1/train_20260429T075804_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_2/train_20260429T092457_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_3/train_20260429T102501_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_4/train_20260429T111919_cifar10_vgg16/checkpoints/best.ckpt'

echo '=== MNIST ==='
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_0/train_20260428T065024_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_1/train_20260428T070152_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_2/train_20260428T071322_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_3/train_20260428T072501_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_4/train_20260428T073656_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/fi_edl/seed_0/train_20260428T171026_mnist_convnet/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/fi_edl/seed_1/train_20260428T180020_mnist_convnet/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/fi_edl/seed_2/train_20260428T191508_mnist_convnet/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/fi_edl/seed_3/train_20260428T200746_mnist_convnet/checkpoints/best.ckpt'
run_eval 'fi_edl' 'fi_edl' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/fi_edl/seed_4/train_20260428T211420_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_0/train_20260428T074827_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_1/train_20260428T080254_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_2/train_20260428T081715_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_3/train_20260428T083201_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_4/train_20260428T084633_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_0/train_20260428T090030_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_1/train_20260428T094441_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_2/train_20260428T112051_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_3/train_20260428T120128_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'r_edl' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_4/train_20260428T123824_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_0/train_20260428T131942_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_1/train_20260428T135404_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_2/train_20260428T144126_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_3/train_20260428T153254_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 're_edl' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_4/train_20260428T163301_mnist_convnet/checkpoints/best.ckpt'

echo "All eval jobs completed."
