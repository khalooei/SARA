#!/usr/bin/env bash
# Controlled protocol on SVHN with ResNet18.
# Usage (from the repository root): bash experiments/controlled_svhn.sh
set -euo pipefail

OUT="${OUT:-runs}"

python train.py --dataset svhn --model resnet18 --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/svhn/standard"
python train.py --dataset svhn --model resnet18 --sara --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/svhn/standard_sara"
python train.py --dataset svhn --model resnet18 --base-loss pgd --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/svhn/at_pgd"
python train.py --dataset svhn --model resnet18 --base-loss pgd --sara --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/svhn/at_pgd_sara"
python train.py --dataset svhn --model resnet18 --base-loss mart --epochs 15 --batch-size 256 --lr 0.02 --eval-every 15 --eval-apgd --eval-apgd-iters 30 --eval-max-batches 20 \
    --out-dir "$OUT/svhn/at_mart"
