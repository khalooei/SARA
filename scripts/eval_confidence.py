"""Confidence, calibration and margin of a trained run, and robustness to an unseen threat model.

SARA trains on corrections that pass a confidence and margin test, so this measures where such an
effect would show: on the full test set (clean) and on PGD-20 adversarial examples (first --n-adv
images) it reports accuracy, expected calibration error (15 equal-width bins), negative log-likelihood,
mean confidence of correct and wrong predictions, and quantiles of the margin (true-class probability
minus the largest other probability; positive means correct). It also reports APGD-CE accuracy under
an L2 attack (eps 0.5, not used in training) on the first --n-l2 images. Histograms for the figures
are saved alongside.

Usage: python scripts/eval_confidence.py --run-dir runs/std/cifar10/at_pgd_sara/seed0
Writes <run-dir>/confidence.json.
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


def summarize(logits, y):
    p = logits.float().softmax(1)
    conf, pred = p.max(1)
    ok = pred == y
    p_true = p.gather(1, y[:, None]).squeeze(1)
    other = p.clone()
    other.scatter_(1, y[:, None], -1.0)
    margin = p_true - other.max(1).values
    edges = torch.linspace(0, 1, BINS + 1, device=p.device)
    ece, rel = 0.0, []
    for i in range(BINS):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += float(m.float().mean() * (ok[m].float().mean() - conf[m].mean()).abs())
            rel.append({"bin": i, "n": int(m.sum()), "conf": float(conf[m].mean()), "acc": float(ok[m].float().mean())})
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=p.device)
    hist = torch.histc(margin, bins=40, min=-1, max=1)
    return {"n": int(y.numel()), "acc": 100.0 * float(ok.float().mean()), "ece": 100.0 * ece,
            "nll": float(F.cross_entropy(logits.float(), y)),
            "conf_correct": float(conf[ok].mean()) if ok.any() else None,
            "conf_wrong": float(conf[~ok].mean()) if (~ok).any() else None,
            "margin_quantiles": dict(zip(["p10", "p25", "p50", "p75", "p90"], [float(v) for v in margin.quantile(q)])),
            "frac_margin_ge_0.5": float((margin >= 0.5).float().mean()),
            "reliability": rel, "margin_hist": {"min": -1, "max": 1, "counts": [int(c) for c in hist.tolist()]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n-adv", type=int, default=5000)
    p.add_argument("--n-l2", type=int, default=2000)
    p.add_argument("--l2-eps", type=float, default=0.5)
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
    _, _, test_loader = get_gpu_loaders(cfg["dataset"], batch_size=128, augment=False, val_size=0,
                                        eval_batch_size=args.batch_size, device=device)
    import torchattacks
    l2 = torchattacks.APGD(model, norm="L2", eps=args.l2_eps, steps=100, loss="ce", n_restarts=1, seed=args.seed)
    clean_l, clean_y, adv_l, adv_y = [], [], [], []
    l2_ok, n_l2, n = 0, 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_l.append(model(x))
            clean_y.append(y)
        if n < args.n_adv:
            k = min(x.shape[0], args.n_adv - n)
            xa = pgd(model, x[:k], y[:k], eps)
            with torch.no_grad():
                adv_l.append(model(xa))
                adv_y.append(y[:k])
        if n_l2 < args.n_l2:
            k = min(x.shape[0], args.n_l2 - n_l2)
            with torch.no_grad():
                l2_ok += int((model(l2(x[:k], y[:k])).argmax(1) == y[:k]).sum())
            n_l2 += k
        n += x.shape[0]
    out = {"run_dir": os.path.relpath(args.run_dir), "checkpoint": os.path.basename(ckpt), "precision": "float32",
           "clean": summarize(torch.cat(clean_l), torch.cat(clean_y)),
           "pgd20": summarize(torch.cat(adv_l), torch.cat(adv_y)),
           "l2_apgd": {"eps": args.l2_eps, "n": n_l2, "acc": 100.0 * l2_ok / n_l2}}
    with open(os.path.join(args.run_dir, "confidence.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("reliability", "margin_hist")}
                          if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
