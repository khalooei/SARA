"""Phase A2 pilot: alignment of adversarial predictions to accepted corrections (one seed, validation split)."""
import sys

OUT = sys.argv[1]
DS = {
    "mnist":   "--dataset mnist --model lenet --epochs 20 --batch-size 128 --lr 0.01",
    "cifar10": "--dataset cifar10 --model resnet18 --epochs 15 --batch-size 256 --lr 0.02",
    "svhn":    "--dataset svhn --model resnet18 --epochs 15 --batch-size 256 --lr 0.02",
}
EV = "--val-size 5000 --eval-every 5 --eval-max-batches 4 --eval-apgd --eval-apgd-iters 30"
S = "--sara --sara-max-correct 64 --cg-confidence-cap"
V = {
    "at_pgd_sara_align1": f"--base-loss pgd {S} --sara-align 1",
    "at_pgd_sara_align3": f"--base-loss pgd {S} --sara-align 3",
    "at_pgd_sara_align1_rev": f"--base-loss pgd {S} --sara-align 1 --revisit-memory 4096",
    "at_pgd_sara_align1_xi03": f"--base-loss pgd {S} --sara-align 1 --xi 0.3",
    "at_pgd_sara_align3_xi03": f"--base-loss pgd {S} --sara-align 3 --xi 0.3",
    "at_mart_sara_align1": f"--base-loss mart {S} --sara-align 1",
}
for ds, dargs in DS.items():
    for name, vargs in V.items():
        print(f"python train.py {dargs} {vargs} {EV} --out-dir {OUT}/{ds}/{name}")
