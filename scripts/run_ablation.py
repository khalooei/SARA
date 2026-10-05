"""Isolated hyperparameter ablation of SARA (Standard+SARA on MNIST/LeNet).

Each of epsilon, xi and kappa is varied while the others are held at their defaults.
Every run reports clean and PGD accuracy together with the empirical local-Lipschitz
estimate of lipschitz.py.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.optim as optim

from sara.confidence_guard import SARAConfig
from sara.data import DEFAULT_EPSILON, get_loaders
from sara.evaluate import evaluate
from sara.lipschitz import local_lipschitz_estimate
from sara.losses import build_base_loss
from sara.models import build_model
from sara.trainer import sara_training_step
from sara.utils import set_seed


def run_one(dataset, model_name, epsilon, tau, xi, kappa, epochs, seed, batch_size=128,
            train_subset=None, test_subset=1000, eval_apgd=False):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(dataset, batch_size=batch_size,
                                             train_subset=train_subset, test_subset=test_subset)
    model = build_model(model_name, dataset).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    base_loss = build_base_loss("ce")
    cfg = SARAConfig(epsilon=epsilon, tau=tau, xi=xi, kappa=kappa)

    last_success_rate = 0.0
    for _ in range(epochs):
        model.train()
        succ = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            stats = sara_training_step(model, optimizer, x, y, base_loss, cfg, use_sara=True)
            succ.append(stats["success_rate"])
        scheduler.step()
        last_success_rate = sum(succ) / max(len(succ), 1)

    eval_out = evaluate(model, test_loader, device, DEFAULT_EPSILON[dataset], pgd_steps=20,
                         run_apgd=eval_apgd, apgd_iters=50)

    model.eval()
    x_probe, _ = next(iter(test_loader))
    x_probe = x_probe[:200].to(device)
    lip = local_lipschitz_estimate(model, x_probe, radius=DEFAULT_EPSILON[dataset], n_dirs=10)

    return {
        "dataset": dataset, "model": model_name, "epsilon": epsilon, "tau": tau, "xi": xi,
        "kappa": kappa, "epochs": epochs, "seed": seed,
        "clean_acc": eval_out["clean_acc"], "pgd_acc": eval_out["pgd_acc"],
        "apgd_acc": eval_out.get("apgd_acc"),
        "local_lipschitz": lip, "final_success_rate": last_success_rate,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--model", default="lenet")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=1000)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "ablation", "ablation.jsonl"))
    p.add_argument("--sweep", choices=["epsilon", "xi", "kappa", "all"], default="all")
    args = p.parse_args()

    default_eps = DEFAULT_EPSILON[args.dataset]
    default_tau, default_xi, default_kappa = 0.05, 0.5, 10

    configs = []
    if args.sweep in ("epsilon", "all"):
        for eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] if args.dataset == "mnist" else [0.01, 0.02, 0.03, 0.05, 0.08]:
            configs.append(dict(epsilon=eps, tau=default_tau, xi=default_xi, kappa=default_kappa, sweep="epsilon"))
    if args.sweep in ("xi", "all"):
        for xi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            configs.append(dict(epsilon=default_eps, tau=default_tau, xi=xi, kappa=default_kappa, sweep="xi"))
    if args.sweep in ("kappa", "all"):
        for kappa in [1, 3, 5, 10, 20]:
            configs.append(dict(epsilon=default_eps, tau=default_tau, xi=default_xi, kappa=kappa, sweep="kappa"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        for c in configs:
            sweep = c.pop("sweep")
            r = run_one(args.dataset, args.model, c["epsilon"], c["tau"], c["xi"], c["kappa"],
                        args.epochs, args.seed, train_subset=args.train_subset, test_subset=args.test_subset)
            r["sweep"] = sweep
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")
            f.flush()
