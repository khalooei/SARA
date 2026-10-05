#!/usr/bin/env bash
# Controlled protocol on CIFAR10 with ResNet18.
# Usage (from the repository root): bash experiments/controlled_cifar10.sh
set -euo pipefail

OUT="${OUT:-runs}"

python train.py --dataset cifar10 --model resnet18 --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar10/standard"
python train.py --dataset cifar10 --model resnet18 --sara --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar10/standard_sara"
python train.py --dataset cifar10 --model resnet18 --base-loss pgd --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar10/at_pgd"
python train.py --dataset cifar10 --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar10/at_pgd_sara"
python train.py --dataset cifar10 --model resnet18 --base-loss mart --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar10/at_mart"
python train.py --dataset cifar10 --model resnet18 --base-loss mart --sara --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar10/at_mart_sara"
python train.py --dataset cifar10 --model resnet18 --method gairat --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/cifar10/gairat"
