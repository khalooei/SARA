"""Full robustness evaluation of a trained run: AutoAttack with a per-attack breakdown, and a
gradient-masking check by an epsilon sweep.

AutoAttack (standard version: APGD-CE, APGD-T, FAB-T, Square) is run stage by stage with a fixed
seed, each attack applied to the images still classified correctly, on the first test images (the
same images for every run, so results can be paired image by image). The three white-box stages run on
the first --n images; Square, which takes about 90% of the evaluation time, runs on the first
--n-square of them, which gives standard AutoAttack accuracy there. An image that Square breaks after
the gradient-based attacks failed is a sign of gradient masking, so that extra drop is reported. The
epsilon sweep runs PGD-50 in float32 on the first --n-sweep images at several radii; robust accuracy
should fall monotonically and reach about zero at large radii. Per-image correctness (clean and
AutoAttack) is saved so that two runs can be compared with a paired test (McNemar).

Usage: python scripts/eval_robust_suite.py --run-dir runs/std/cifar10/at_pgd_sara/seed0 [--n 5000]
Writes <run-dir>/robust_suite.json.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sara.data import get_gpu_loaders  # noqa: E402
from sara.models import build_model  # noqa: E402


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=1)


def pgd_fp32(model, x, y, eps, steps=50):
    """L_inf PGD with random start, step 2.5 * eps / steps, cross-entropy, float32 gradients."""
    alpha = 2.5 * eps / steps
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = torch.min(torch.max(x_adv.detach() + alpha * g.sign(), x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n", type=int, default=5000, help="test images for the white-box AutoAttack stages")
    p.add_argument("--n-square", type=int, default=2000,
                   help="first test images that also get the Square stage (standard AutoAttack on these)")
    p.add_argument("--n-sweep", type=int, default=1000, help="test images for the epsilon sweep")
    p.add_argument("--eps-sweep", default="2,4,8,12,16,32", help="radii of the sweep, in units of 1/255")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    device = "cuda"
    torch.manual_seed(args.seed)
    model = build_model(cfg["model"], cfg["dataset"]).to(device)
    ckpt = os.path.join(args.run_dir, "model_ema.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(args.run_dir, "model_final.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    eps = cfg["at_epsilon"]
    _, _, test_loader = get_gpu_loaders(cfg["dataset"], batch_size=128, augment=False, val_size=0,
                                        eval_batch_size=args.batch_size, device=device)

    import torchattacks
    aa = torchattacks.AutoAttack(model, norm="Linf", eps=eps, version="standard", n_classes=10, seed=args.seed)
    stages = aa._autoattack.attacks
    names = ["APGD-CE", "APGD-T", "FAB-T", "Square"]
    t0 = time.time()
    clean_bits, wb_bits, robust_bits, sq_mask = [], [], [], []
    after, stage_s = [0] * len(stages), [0.0] * len(stages)
    n = 0
    for x, y in test_loader:
        if n >= args.n:
            break
        x, y = x[: args.n - n].to(device), y[: args.n - n].to(device)
        n += x.shape[0]
        in_sq = torch.arange(n - x.shape[0], n, device=device) < args.n_square
        ok = predict(model, x) == y
        clean_bits.append(ok.clone())
        for i, atk in enumerate(stages):
            ts = time.time()
            if names[i] == "Square":
                wb_bits.append(ok.clone())
                sq_mask.append(in_sq)
            idx = (ok & in_sq if names[i] == "Square" else ok).nonzero().flatten()
            if idx.numel():
                x_adv = atk(x[idx], y[idx])
                ok[idx] = predict(model, x_adv) == y[idx]
            after[i] += int((ok & in_sq).sum()) if names[i] == "Square" else int(ok.sum())
            stage_s[i] += time.time() - ts
        robust_bits.append(ok)
    clean = torch.cat(clean_bits)
    wb = torch.cat(wb_bits)
    sq = torch.cat(sq_mask)
    full = torch.cat(robust_bits)[sq]  # standard AutoAttack (all four stages) on the first n_square images
    n_sq = int(sq.sum())
    t_aa = time.time() - t0

    sweep = []
    xs, ys, m = [], [], 0
    for x, y in test_loader:
        if m >= args.n_sweep:
            break
        xs.append(x[: args.n_sweep - m])
        ys.append(y[: args.n_sweep - m])
        m += xs[-1].shape[0]
    xs, ys = (torch.cat(xs).to(device), torch.cat(ys).to(device)) if xs else (None, None)
    for e in ([float(v) for v in args.eps_sweep.split(",")] if m else []):
        correct = 0
        for i in range(0, m, args.batch_size):
            xb, yb = xs[i:i + args.batch_size], ys[i:i + args.batch_size]
            correct += int((predict(model, pgd_fp32(model, xb, yb, e / 255)) == yb).sum())
        sweep.append({"eps_255": e, "pgd50_acc": 100.0 * correct / m})

    out = {"run_dir": os.path.relpath(args.run_dir), "checkpoint": os.path.basename(ckpt), "n": n, "seed": args.seed,
           "epsilon": eps, "precision": "float32", "clean_acc": 100.0 * float(clean.float().mean()),
           "n_square": n_sq,
           "autoattack_acc": 100.0 * float(full.float().mean()),
           "whitebox_acc": 100.0 * float(wb.float().mean()),
           "whitebox_acc_on_square_subset": 100.0 * float(wb[sq].float().mean()),
           "square_extra_drop": 100.0 * float((wb[sq] & ~full).float().mean()),
           "stages": [{"attack": names[i], "images": n_sq if names[i] == "Square" else n,
                       "robust_acc_after": 100.0 * after[i] / (n_sq if names[i] == "Square" else n),
                       "time_s": round(stage_s[i])} for i in range(len(stages))],
           "eps_sweep": {"n": m, "attack": "PGD-50, step 2.5*eps/50, float32", "results": sweep},
           "per_image": {"clean": "".join("1" if b else "0" for b in clean.tolist()),
                         "whitebox": "".join("1" if b else "0" for b in wb.tolist()),
                         "autoattack": "".join("1" if b else "0" for b in full.tolist())},
           "time_s": {"autoattack": round(t_aa), "total": round(time.time() - t0)}}
    with open(os.path.join(args.run_dir, "robust_suite.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}, indent=1))


if __name__ == "__main__":
    main()
