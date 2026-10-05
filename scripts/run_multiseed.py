"""Multi-seed variance study on MNIST: trains each variant with several seeds and
reports the mean and standard deviation of clean, PGD and APGD accuracy."""
import argparse
import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.optim as optim

from sara.confidence_guard import SARAConfig
from sara.data import DEFAULT_EPSILON, get_loaders
from sara.evaluate import evaluate
from sara.losses import build_base_loss
from sara.models import build_model
from sara.trainer import sara_training_step
from sara.utils import set_seed


def run_one(use_sara, seed, epochs, dataset="mnist", model_name="lenet", base_loss_name="ce", pgd_steps=7):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(dataset, batch_size=128)
    model = build_model(model_name, dataset).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    base_loss = build_base_loss(base_loss_name)
    base_loss_kwargs = {"pgd_steps": pgd_steps} if base_loss_name != "ce" else {}
    eps = DEFAULT_EPSILON[dataset]
    cfg = SARAConfig(epsilon=eps, tau=0.05, xi=0.5, kappa=10)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            sara_training_step(model, optimizer, x, y, base_loss, cfg,
                                base_loss_kwargs=base_loss_kwargs, use_sara=use_sara)
        scheduler.step()

    eval_out = evaluate(model, test_loader, device, eps, pgd_steps=20, run_apgd=True, apgd_iters=50)
    return eval_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--base-loss", default="ce")
    p.add_argument("--pgd-steps", type=int, default=7)
    p.add_argument("--label-prefix", default="")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "multiseed", "mnist_multiseed.jsonl"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    key_base = f"{args.label_prefix}standard" if args.label_prefix else "standard"
    key_sara = f"{args.label_prefix}standard_sara" if args.label_prefix else "standard_sara"
    all_results = {key_base: [], key_sara: []}
    with open(args.out, "a") as f:
        for use_sara, key in [(False, key_base), (True, key_sara)]:
            for seed in args.seeds:
                r = run_one(use_sara, seed, args.epochs, base_loss_name=args.base_loss, pgd_steps=args.pgd_steps)
                r["seed"] = seed
                r["variant"] = key
                print(json.dumps(r))
                f.write(json.dumps(r) + "\n")
                f.flush()
                all_results[key].append(r)

        for key, rs in all_results.items():
            for metric in ["clean_acc", "pgd_acc", "apgd_acc"]:
                vals = [r[metric] for r in rs]
                summary = {"variant": key, "metric": metric, "mean": statistics.mean(vals),
                           "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}
                print(json.dumps(summary))
                f.write(json.dumps(summary) + "\n")
