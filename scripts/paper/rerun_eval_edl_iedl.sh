#!/bin/bash
# Re-eval EDL/I-EDL with paper-match retrain ckpts (val/acc monitor, anneal=10).
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

echo "Starting 20 paper-match eval jobs..."

echo '=== CIFAR-10 EDL ==='
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_0/train_20260504T132030_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_1/train_20260504T142107_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_2/train_20260504T161933_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_3/train_20260504T183701_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/edl_l1/seed_4/train_20260504T210046_cifar10_vgg16_edl_fixed_l1/checkpoints/best.ckpt'
echo '=== CIFAR-10 I-EDL ==='
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_0/train_20260504T223513_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_1/train_20260505T002232_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_2/train_20260505T022448_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_3/train_20260505T042339_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/i_edl/seed_4/train_20260505T065539_cifar10_vgg16/checkpoints/best.ckpt'
echo '=== MNIST EDL ==='
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_0/train_20260504T134336_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_1/train_20260504T154806_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_2/train_20260504T181410_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_3/train_20260504T202742_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
run_eval 'edl_l1' 'edl_fixed_l1' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/edl_l1/seed_4/train_20260504T214540_mnist_convnet_edl_fixed_l1/checkpoints/best.ckpt'
echo '=== MNIST I-EDL ==='
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_0/train_20260504T233432_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_1/train_20260505T013250_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_2/train_20260505T033917_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_3/train_20260505T055918_mnist_convnet/checkpoints/best.ckpt'
run_eval 'i_edl' 'i_edl' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/i_edl/seed_4/train_20260505T080908_mnist_convnet/checkpoints/best.ckpt'

echo "All eval jobs done."
