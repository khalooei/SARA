"""Post-hoc calibration and selective prediction of a trained run, on clean and PGD-20 test inputs.

Temperature scaling: one temperature is fitted by minimizing the NLL of the clean validation split that
was held out during training (the same 5,000 images for every run), then applied to the test set. The
calibration error (15 bins) is reported before and after scaling, on clean inputs and on PGD-20 inputs.

Selective prediction: inputs are ranked by confidence and the model answers only the most confident
fraction. Reported are the area under the risk-coverage curve (AURC, lower is better) and the accuracy at
50% and 80% coverage. Under attack this measures whether low confidence flags the inputs the attack broke.

Usage: python scripts/eval_reliability.py --run-dir runs/std/cifar10/at_pgd_sara/seed0
Writes <run-dir>/reliability.json.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sara.data import get_gpu_loaders  # noqa: E402
from sara.models import build_model  # noqa: E402

BINS = 15


def pgd(model, x, y, eps, steps=20):
    alpha = eps / 4
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(model(x_adv), y), x_adv)[0]
        x_adv = torch.min(torch.max(x_adv.detach() + alpha * g.sign(), x - eps), x + eps).clamp(0, 1)
    return x_adv.detach()


def ece(logits, y):
    conf, pred = logits.softmax(1).max(1)
    ok = (pred == y).float()
    edges = torch.linspace(0, 1, BINS + 1, device=logits.device)
    e = 0.0
    for i in range(BINS):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += float(m.float().mean() * (ok[m].mean() - conf[m].mean()).abs())
    return 100.0 * e


def selective(logits, y):
    conf, pred = logits.softmax(1).max(1)
    err = (pred != y).float()[conf.argsort(descending=True)]
    n = err.numel()
    risk = err.cumsum(0) / torch.arange(1, n + 1, device=err.device)
    def acc_at(c):
        return 100.0 * float(1 - err[: max(1, int(c * n))].mean())
    return {"aurc": 100.0 * float(risk.mean()), "acc_at_50": acc_at(0.5), "acc_at_80": acc_at(0.8)}


def fit_temperature(logits, y):
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


@torch.no_grad()
def logits_of(model, loader, device, limit=None):
    out, ys, n = [], [], 0
    for x, y in loader:
        if limit is not None and n >= limit:
            break
        x, y = x.to(device), y.to(device)
        out.append(model(x).float())
        ys.append(y)
        n += x.shape[0]
    return torch.cat(out), torch.cat(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n-adv", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    torch.manual_seed(args.seed)
    device = "cuda"
    model = build_model(cfg["model"], cfg["dataset"]).to(device)
    ckpt = os.path.join(args.run_dir, "model_ema.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(args.run_dir, "model_final.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    eps = cfg["at_epsilon"]
    _, val_loader, test_loader = get_gpu_loaders(cfg["dataset"], batch_size=128, augment=False,
                                                 val_size=cfg.get("val_size") or 5000,
                                                 eval_batch_size=args.batch_size, device=device)
    val_l, val_y = logits_of(model, val_loader, device)
    t = fit_temperature(val_l, val_y)
    clean_l, clean_y = logits_of(model, test_loader, device)
    adv_l, adv_y, n = [], [], 0
    for x, y in test_loader:
        if n >= args.n_adv:
            break
        k = min(x.shape[0], args.n_adv - n)
        x, y = x[:k].to(device), y[:k].to(device)
        xa = pgd(model, x, y, eps)
        with torch.no_grad():
            adv_l.append(model(xa).float())
        adv_y.append(y)
        n += k
    adv_l, adv_y = torch.cat(adv_l), torch.cat(adv_y)
    out = {"run_dir": os.path.relpath(args.run_dir), "checkpoint": os.path.basename(ckpt), "temperature": t,
           "validation_n": int(val_y.numel()),
           "clean": {"n": int(clean_y.numel()), "ece": ece(clean_l, clean_y), "ece_scaled": ece(clean_l / t, clean_y),
                     **selective(clean_l, clean_y)},
           "pgd20": {"n": int(adv_y.numel()), "ece": ece(adv_l, adv_y), "ece_scaled": ece(adv_l / t, adv_y),
                     **selective(adv_l, adv_y)}}
    with open(os.path.join(args.run_dir, "reliability.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
