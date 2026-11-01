"""Phase A pilot: SARA variants under the controlled protocol (one seed, validation split)."""
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/pilot"
DS = {
    "mnist":   "--dataset mnist --model lenet --epochs 20 --batch-size 128 --lr 0.01",
    "cifar10": "--dataset cifar10 --model resnet18 --epochs 15 --batch-size 256 --lr 0.02",
    "svhn":    "--dataset svhn --model resnet18 --epochs 15 --batch-size 256 --lr 0.02",
}
SMALL = {"mnist": 0.1, "cifar10": 0.01, "svhn": 0.01}
BIG = {"mnist": 0.2, "cifar10": 0.03, "svhn": 0.03}
EV = "--val-size 5000 --eval-every 5 --eval-max-batches 4 --eval-apgd --eval-apgd-iters 30"
S = "--sara --sara-max-correct 64"
V = {
    "standard": "",
    "standard_sara": f"{S}",
    "standard_sara_rev": f"{S} --cg-confidence-cap --revisit-memory 4096",
    "at_pgd": "--base-loss pgd",
    "at_pgd_sara": f"--base-loss pgd {S}",
    "at_pgd_sara_cap": f"--base-loss pgd {S} --cg-confidence-cap",
    "at_pgd_sara_robust": f"--base-loss pgd {S} --robust-correction",
    "at_pgd_sara_robust_cap": f"--base-loss pgd {S} --robust-correction --cg-confidence-cap",
    "at_pgd_sara_rev": f"--base-loss pgd {S} --cg-confidence-cap --revisit-memory 4096",
    "at_pgd_sara_robust_rev": f"--base-loss pgd {S} --robust-correction --cg-confidence-cap --revisit-memory 4096",
    "at_mart": "--base-loss mart",
    "at_mart_sara_robust_cap": f"--base-loss mart {S} --robust-correction --cg-confidence-cap",
}
for ds, dargs in DS.items():
    for name, vargs in V.items():
        print(f"python train.py {dargs} {vargs} {EV} --out-dir {OUT}/{ds}/{name}")
    # Tuned-protocol-like setting: adversarial training radius below the evaluation radius.
    small, big = SMALL[ds], BIG[ds]
    mm = f"--base-loss pgd --at-epsilon {small} --eval-epsilons {big}"
    print(f"python train.py {dargs} {mm} {EV} --out-dir {OUT}/{ds}/smallat_pgd")
    print(f"python train.py {dargs} {mm} {S} --sara-epsilon {big} {EV} --out-dir {OUT}/{ds}/smallat_pgd_sara")
    print(f"python train.py {dargs} {mm} {S} --sara-epsilon {big} --cg-confidence-cap --revisit-memory 4096 {EV} "
          f"--out-dir {OUT}/{ds}/smallat_pgd_sara_rev")
    print(f"python train.py {dargs} {mm} {S} --sara-epsilon {big} --robust-correction {EV} "
          f"--out-dir {OUT}/{ds}/smallat_pgd_sara_robust")
