"""Hyperparameter search for AT-PGD+SARA on MNIST (round 2): kappa, tau and
lambda_margin, with the correction radius fixed to the adversarial-training epsilon
(the best setting of round 1).
"""
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


def run_one(kappa, tau, lambda_margin, epochs=10, seed=0):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders("mnist", batch_size=128)
    model = build_model("lenet", "mnist").to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    base_loss = build_base_loss("pgd")
    at_eps = DEFAULT_EPSILON["mnist"]
    cfg = SARAConfig(epsilon=at_eps, tau=tau, xi=0.5, kappa=kappa, lambda_margin=lambda_margin)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            sara_training_step(model, optimizer, x, y, base_loss, cfg,
                                base_loss_kwargs={"pgd_steps": 7}, use_sara=True)
        scheduler.step()

    return evaluate(model, test_loader, device, at_eps, pgd_steps=20, run_apgd=True, apgd_iters=50)


if __name__ == "__main__":
    out_path = os.path.join(REPO_ROOT, "runs", "sara_hparam_search", "mnist_atpgd_sara_search_round2.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid = list(itertools.product(
        [10, 20],              # kappa
        [0.02, 0.05, 0.1, 0.2],  # tau
        [1.0, 3.0],             # lambda_margin (both the default and round-1's best)
    ))
    print(f"Grid size: {len(grid)}")
    with open(out_path, "a") as f:
        for kappa, tau, lam in grid:
            r = run_one(kappa, tau, lam)
            r.update({"kappa": kappa, "tau": tau, "lambda_margin": lam})
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")
            f.flush()
