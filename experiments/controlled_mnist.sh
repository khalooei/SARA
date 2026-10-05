#!/usr/bin/env bash
# Controlled protocol on MNIST with LeNet: SARA with each base loss and the OHEM, FAT and GAIRAT
# baselines.
# Usage (from the repository root): bash experiments/controlled_mnist.sh
set -euo pipefail

OUT="${OUT:-runs}"

python train.py --dataset mnist --model lenet --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/standard"
python train.py --dataset mnist --model lenet --sara --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/standard_sara"
python train.py --dataset mnist --model lenet --base-loss pgd --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/at_pgd"
python train.py --dataset mnist --model lenet --base-loss pgd --sara --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/at_pgd_sara"
python train.py --dataset mnist --model lenet --base-loss mart --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/at_mart"
python train.py --dataset mnist --model lenet --base-loss mart --sara --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/at_mart_sara"
python train.py --dataset mnist --model lenet --method ohem --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/ohem"
python train.py --dataset mnist --model lenet --method fat --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/fat"
python train.py --dataset mnist --model lenet --method gairat --eval-every 20 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/gairat"
python train.py --dataset mnist --model lenet --base-loss pgd --epochs 40 --eval-every 40 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/at_pgd_40ep"
python train.py --dataset mnist --model lenet --base-loss pgd --sara --epochs 40 --eval-every 40 --eval-apgd --eval-apgd-iters 50 \
    --out-dir "$OUT/mnist/at_pgd_sara_40ep"
