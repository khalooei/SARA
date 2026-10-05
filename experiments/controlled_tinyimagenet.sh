#!/usr/bin/env bash
# Controlled protocol on Tiny-ImageNet-200 with ResNet18. Download
# https://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into data/ first.
# Usage (from the repository root): bash experiments/controlled_tinyimagenet.sh
set -euo pipefail

OUT="${OUT:-runs}"

python train.py --dataset tinyimagenet --model resnet18 --epochs 10 --batch-size 256 --lr 0.02 --eval-every 10 --eval-apgd --eval-apgd-iters 20 --eval-max-batches 15 \
    --out-dir "$OUT/tinyimagenet/standard"
python train.py --dataset tinyimagenet --model resnet18 --sara --epochs 10 --batch-size 256 --lr 0.02 --sara-max-correct 64 --eval-every 10 --eval-apgd --eval-apgd-iters 20 --eval-max-batches 15 \
    --out-dir "$OUT/tinyimagenet/standard_sara"
python train.py --dataset tinyimagenet --model resnet18 --sara --epochs 10 --batch-size 256 --lr 0.02 --sara-max-correct 64 --lambda-margin 0.1 --eval-every 10 --eval-apgd --eval-apgd-iters 20 --eval-max-batches 15 \
    --out-dir "$OUT/tinyimagenet/standard_sara_lambda0.1"
