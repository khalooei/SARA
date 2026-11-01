"""Evaluate saved checkpoints under the four non-adversarial corruptions of the paper."""
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from torch.utils.data import DataLoader, TensorDataset

from sara.data import DATA_ROOT
from sara.models import build_model
from scripts.eval_corruptions import evaluate_corruptions

JOBS = [
    ("mnist", "lenet", "runs/pilot/mnist/standard/model_final.pt", "standard"),
    ("mnist", "lenet", "runs/pilot/mnist/standard_sara_rev/model_final.pt", "standard_sara_rev"),
    ("mnist", "lenet", "runs/main/mnist/at_pgd/seed0/model_final.pt", "at_pgd"),
    ("mnist", "lenet", "runs/main/mnist/at_pgd_sara/seed0/model_final.pt", "at_pgd_sara"),
    ("cifar10", "resnet18", "runs/pilot/cifar10/standard/model_final.pt", "standard"),
    ("cifar10", "resnet18", "runs/pilot/cifar10/standard_sara_rev/model_final.pt", "standard_sara_rev"),
    ("cifar10", "resnet18", "runs/main/cifar10/at_pgd/seed0/model_final.pt", "at_pgd"),
    ("cifar10", "resnet18", "runs/main/cifar10/at_pgd_sara/seed0/model_final.pt", "at_pgd_sara"),
]


def main():
    device = torch.device("cpu")
    out_path = os.path.join(REPO_ROOT, "runs", "corruption", "corruption.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w"):
        pass
    for ds, model_name, ckpt, label in JOBS:
        path = os.path.join(REPO_ROOT, ckpt)
        print("eval", ds, label, flush=True)
        cache = torch.load(os.path.join(DATA_ROOT, f"{ds}_test.pt"))
        x, y = cache["x"].float() / 255.0, cache["y"].long()
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(x.shape[0], generator=g)[:2000]
        test_loader = DataLoader(TensorDataset(x[idx], y[idx]), batch_size=256, shuffle=False, num_workers=0)
        model = build_model(model_name, ds).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        out = evaluate_corruptions(model, test_loader, device)
        out["label"] = f"{ds}/{label}"
        out["checkpoint"] = os.path.relpath(ckpt)
        print(json.dumps(out), flush=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(out) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
