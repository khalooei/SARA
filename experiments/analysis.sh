#!/usr/bin/env bash
# Analysis experiments on MNIST with LeNet: isolated hyperparameter ablation with a
# local-Lipschitz estimate, multi-seed variance, hyperparameter search, computational
# overhead and robustness to common corruptions.
# Usage (from the repository root): bash experiments/analysis.sh
set -euo pipefail

# Isolated ablation of epsilon, xi and kappa (Standard+SARA, 10 epochs).
python scripts/run_ablation.py --sweep all

# Three seeds: Standard vs. Standard+SARA (15 epochs) and AT-PGD vs. AT-PGD+SARA (20 epochs).
python scripts/run_multiseed.py --epochs 15
python scripts/run_multiseed.py --epochs 20 --base-loss pgd --label-prefix atpgd_ \
    --out runs/multiseed/mnist_atpgd_multiseed.jsonl

# Two-round hyperparameter search for AT-PGD+SARA, then three-seed confirmation of the best setting.
python scripts/sara_hparam_search.py
python scripts/sara_hparam_search_round2.py
python scripts/confirm_best_sara.py

# Batch time and peak GPU memory as a function of kappa (requires CUDA), for an untrained model
# (worst case) and for a LeNet trained for three epochs (converged case).
python scripts/overhead_benchmark.py --dataset mnist --model lenet --base-loss ce
python train.py --dataset mnist --model lenet --epochs 3 --lr 0.1 --out-dir runs/mnist/pretrained_3ep
python scripts/overhead_benchmark.py --dataset mnist --model lenet --base-loss ce \
    --init-checkpoint runs/mnist/pretrained_3ep/model_final.pt
python scripts/overhead_benchmark.py --dataset cifar10 --model resnet18 --base-loss pgd

# Corruption robustness of trained checkpoints (train them first with experiments/controlled_mnist.sh).
for name in standard standard_sara; do
    python scripts/eval_corruptions.py --dataset mnist --model lenet --label "$name" \
        --checkpoint "runs/mnist/$name/model_final.pt"
done
