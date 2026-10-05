"""Hyperparameter search for AT-PGD+SARA on MNIST (round 1).

Proxy runs of 10 epochs with a single seed over lambda_margin and the radius of the
correction search (as a fraction of the adversarial-training epsilon), at fixed kappa,
xi and tau. The best configuration is then retrained with the full protocol by
confirm_best_sara.py.
"""
import argparse
import itertools
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
from sara.losses import build_base_loss
from sara.models import build_model
from sara.trainer import sara_training_step
from sara.utils import set_seed


def run_one(kappa, xi, tau, lambda_margin, sara_eps_mult, epochs, seed=0):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders("mnist", batch_size=128)
    model = build_model("lenet", "mnist").to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    base_loss = build_base_loss("pgd")
    at_eps = DEFAULT_EPSILON["mnist"]
    cfg = SARAConfig(epsilon=at_eps * sara_eps_mult, tau=tau, xi=xi, kappa=kappa, lambda_margin=lambda_margin)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            sara_training_step(model, optimizer, x, y, base_loss, cfg,
                                base_loss_kwargs={"pgd_steps": 7}, use_sara=True)
        scheduler.step()

    eval_out = evaluate(model, test_loader, device, at_eps, pgd_steps=20, run_apgd=True, apgd_iters=50)
    return eval_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out",
                   default=os.path.join(REPO_ROOT, "runs", "sara_hparam_search", "mnist_atpgd_sara_search.jsonl"))
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Grid: lambda_margin x correction-epsilon fraction, at fixed kappa, xi and tau
    # (xi and kappa are studied separately in run_ablation.py).
    grid = list(itertools.product(
        [10],             # kappa
        [0.5],            # xi
        [0.05],           # tau
        [0.1, 0.3, 1.0, 3.0],  # lambda_margin
        [0.25, 0.5, 1.0],       # correction epsilon as a fraction of the AT epsilon
    ))
    print(f"Grid size: {len(grid)}")

    with open(args.out, "a") as f:
        for kappa, xi, tau, lam, eps_mult in grid:
            r = run_one(kappa, xi, tau, lam, eps_mult, args.epochs)
            r.update({"kappa": kappa, "xi": xi, "tau": tau, "lambda_margin": lam,
                       "sara_eps_mult": eps_mult, "epochs": args.epochs})
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")
            f.flush()
