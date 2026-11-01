"""Phase B: multi-seed controlled comparison on the 10-class benchmarks.

Usage: python experiments/gen_main.py OUT_DIR GROUP [VARIANTS] [SEEDS]
  GROUP: baselines | sara. For GROUP=sara, VARIANTS is "name=args;name=args;..." where args may use the
  placeholders {SMALL}/{BIG} (training radii), {LONG} (twice the default epochs), {LRLO}/{LRHI} (half and twice
  the default learning rate). SEEDS is a comma-separated list (default 0,1,2).
"""
import sys

OUT, GROUP = sys.argv[1], sys.argv[2]
DS = {
    "mnist":   "--dataset mnist --model lenet --epochs 20 --batch-size 128 --lr 0.01",
    "cifar10": "--dataset cifar10 --model resnet18 --epochs 15 --batch-size 256 --lr 0.02",
    "svhn":    "--dataset svhn --model resnet18 --epochs 15 --batch-size 256 --lr 0.02",
}
SMALL = {"mnist": 0.1, "cifar10": 0.01, "svhn": 0.01}
BIG = {"mnist": 0.2, "cifar10": 0.03, "svhn": 0.03}
EV = "--val-size 5000 --eval-every 5 --eval-max-batches 4 --eval-apgd --eval-apgd-iters 30 --eval-autoattack 1000"
S = "--sara --sara-max-correct 64"
LONG = {"mnist": 40, "cifar10": 30, "svhn": 30}
LR = {"mnist": 0.01, "cifar10": 0.02, "svhn": 0.02}
SEEDS = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2]


def variants(ds):
    small, big = SMALL[ds], BIG[ds]
    mm = f"--base-loss pgd --at-epsilon {small} --eval-epsilons {big}"
    if GROUP == "baselines":
        return {
            "standard": "", "at_pgd": "--base-loss pgd", "at_mart": "--base-loss mart",
            "gairat": "--method gairat", "fat": "--method fat", "smallat_pgd": mm,
        }
    extra = sys.argv[3] if len(sys.argv) > 3 else ""
    return {name: args for name, args in (v.split("=", 1) for v in extra.split(";") if v)}


for ds, dargs in DS.items():
    for name, vargs in variants(ds).items():
        vargs = (vargs.replace("{SMALL}", str(SMALL[ds])).replace("{BIG}", str(BIG[ds]))
                 .replace("{LONG}", str(LONG[ds])).replace("{LRLO}", str(LR[ds] / 2))
                 .replace("{LRHI}", str(LR[ds] * 2)))
        for seed in SEEDS:
            print(f"python train.py {dargs} {vargs} --seed {seed} {EV} --out-dir {OUT}/{ds}/{name}/seed{seed}")
