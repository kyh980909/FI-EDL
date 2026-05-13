#!/bin/bash
# Re-eval R-EDL/Re-EDL with paper-match retrain ckpts.
set -euo pipefail
PYTHON=./.venv/bin/python
run_eval() {
    local exp=$1 backbone=$2 dataset=$3 local_dir=$4 seed=$5 ckpt=$6
    $PYTHON -m src.eval \
        experiment="$exp" \
        dataset="$dataset" \
        backbone="$backbone" \
        seed="$seed" \
        checkpoint="$ckpt" \
        data.root=/home/user/FI-EDL/data \
        logging.local_dir="$local_dir" \
        logging.wandb.enabled=false
    echo "[OK] $exp seed=$seed $dataset"
}
echo "Starting 20 R-EDL/Re-EDL paper-match eval jobs..."
echo '=== R-EDL CIFAR10 ==='
run_eval 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_0/train_20260505T112818_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_1/train_20260505T143551_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_2/train_20260505T174605_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_3/train_20260505T205607_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 'r_edl' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/r_edl/seed_4/train_20260506T000611_cifar10_vgg16/checkpoints/best.ckpt'
echo '=== R-EDL MNIST ==='
run_eval 'r_edl' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_0/train_20260505T140502_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_1/train_20260505T171328_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_2/train_20260505T202313_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_3/train_20260505T233425_mnist_convnet/checkpoints/best.ckpt'
run_eval 'r_edl' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/r_edl/seed_4/train_20260506T024226_mnist_convnet/checkpoints/best.ckpt'
echo '=== Re-EDL CIFAR10 ==='
run_eval 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 0 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_0/train_20260506T031621_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 1 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_1/train_20260506T062357_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 2 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_2/train_20260506T093326_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 3 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_3/train_20260506T124521_cifar10_vgg16/checkpoints/best.ckpt'
run_eval 're_edl' 'vgg16' 'cifar10' 'runs/cifar10' 4 '/home/user/FI-EDL/runs/oldcycle_cifar10/re_edl/seed_4/train_20260506T155554_cifar10_vgg16/checkpoints/best.ckpt'
echo '=== Re-EDL MNIST ==='
run_eval 're_edl' 'convnet' 'mnist' 'runs/mnist' 0 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_0/train_20260506T055247_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 'convnet' 'mnist' 'runs/mnist' 1 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_1/train_20260506T085949_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 'convnet' 'mnist' 'runs/mnist' 2 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_2/train_20260506T121308_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 'convnet' 'mnist' 'runs/mnist' 3 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_3/train_20260506T152400_mnist_convnet/checkpoints/best.ckpt'
run_eval 're_edl' 'convnet' 'mnist' 'runs/mnist' 4 '/home/user/FI-EDL/runs/oldcycle_mnist/re_edl/seed_4/train_20260506T183837_mnist_convnet/checkpoints/best.ckpt'
echo "All R-EDL/Re-EDL eval done."
