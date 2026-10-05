"""Retrains the best configuration of the hyperparameter search (lambda_margin = 3.0,
full correction radius, kappa = 10, xi = 0.5, tau = 0.05) with the full 20-epoch
protocol and three seeds.
"""
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


def run_one(seed, epochs=20):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders("mnist", batch_size=128)
    model = build_model("lenet", "mnist").to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    base_loss = build_base_loss("pgd")
    at_eps = DEFAULT_EPSILON["mnist"]
    cfg = SARAConfig(epsilon=at_eps * 1.0, tau=0.05, xi=0.5, kappa=10, lambda_margin=3.0)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            sara_training_step(model, optimizer, x, y, base_loss, cfg,
                                base_loss_kwargs={"pgd_steps": 7}, use_sara=True)
        scheduler.step()

    return evaluate(model, test_loader, device, at_eps, pgd_steps=20, run_apgd=True, apgd_iters=50)


if __name__ == "__main__":
    out_path = os.path.join(REPO_ROOT, "runs", "sara_hparam_search", "mnist_atpgd_sara_best_confirm.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = []
    with open(out_path, "a") as f:
        for seed in [0, 1, 2]:
            r = run_one(seed)
            r["seed"] = seed
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")
            f.flush()
            results.append(r)
        for metric in ["clean_acc", "pgd_acc", "apgd_acc"]:
            vals = [r[metric] for r in results]
            summary = {"metric": metric, "mean": statistics.mean(vals),
                       "std": statistics.pstdev(vals), "n": len(vals)}
            print(json.dumps(summary))
            f.write(json.dumps(summary) + "\n")
