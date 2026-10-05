#!/usr/bin/env bash
# Controlled protocol on CIFAR100 with ResNet18, with the class-count-relative confidence threshold
# and, for comparison, the unscaled threshold.
# Usage (from the repository root): bash experiments/controlled_cifar100.sh
set -euo pipefail

OUT="${OUT:-runs}"

python train.py --dataset cifar100 --model resnet18 --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar100/standard"
python train.py --dataset cifar100 --model resnet18 --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar100/standard_sara"
python train.py --dataset cifar100 --model resnet18 --base-loss pgd --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar100/at_pgd"
python train.py --dataset cifar100 --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --sara-max-correct 64 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar100/at_pgd_sara"
python train.py --dataset cifar100 --model resnet18 --sara --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 --no-xi-relative \
    --out-dir "$OUT/cifar100/standard_sara_xi_unscaled"
python train.py --dataset cifar100 --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 --no-xi-relative \
    --out-dir "$OUT/cifar100/at_pgd_sara_xi_unscaled"
