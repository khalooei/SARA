#!/usr/bin/env bash
# Stabilization mechanisms for AT-PGD+SARA: correction budget, warmup, confidence reweighting,
# success-rate damping and the confidence cap.
# Usage (from the repository root): bash experiments/stabilization.sh
set -euo pipefail

OUT="${OUT:-runs}"

python train.py --dataset cifar10 --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/cifar10_budget"
python train.py --dataset cifar10 --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --sara-warmup-epochs 3 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/cifar10_warmup"
python train.py --dataset cifar10 --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --gamma-reweight 2.0 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/cifar10_reweighting"
python train.py --dataset cifar10 --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --sara-warmup-epochs 3 --gamma-reweight 2.0 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/cifar10_all_three"
python train.py --dataset cifar10 --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --cg-confidence-cap --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/cifar10_budget_cap"
python train.py --dataset svhn --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/svhn_budget"
python train.py --dataset svhn --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --sara-warmup-epochs 3 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/svhn_warmup"
python train.py --dataset svhn --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --gamma-reweight 2.0 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/svhn_reweighting"
python train.py --dataset svhn --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --sara-warmup-epochs 3 --gamma-reweight 2.0 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/svhn_all_three"
python train.py --dataset svhn --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --cg-damping --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/svhn_damping"
python train.py --dataset svhn --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --cg-confidence-cap --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/svhn_budget_cap"
python train.py --dataset mnist --model lenet --base-loss pgd --sara --sara-max-correct 64 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/mnist_budget"
python train.py --dataset mnist --model lenet --base-loss pgd --sara --sara-max-correct 64 --sara-warmup-epochs 5 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/mnist_warmup"
python train.py --dataset mnist --model lenet --base-loss pgd --sara --sara-max-correct 64 --gamma-reweight 2.0 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/mnist_reweighting"
python train.py --dataset mnist --model lenet --base-loss pgd --sara --sara-max-correct 64 --sara-warmup-epochs 5 --gamma-reweight 2.0 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/stabilization/mnist_all_three"
