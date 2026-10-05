"""Wall-clock time and peak GPU memory of a training step with and without SARA, as a
function of kappa, for an untrained (worst case) and a converged model.
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch

from sara.confidence_guard import SARAConfig
from sara.data import DEFAULT_EPSILON, get_loaders
from sara.losses import build_base_loss
from sara.models import build_model
from sara.trainer import sara_training_step


def bench(dataset, model_name, base_loss_name, use_sara, kappa, n_batches=20, batch_size=128, pgd_steps=7,
          init_checkpoint=None):
    if not torch.cuda.is_available():
        raise RuntimeError("The overhead benchmark measures GPU time and memory and requires CUDA.")
    device = torch.device("cuda")
    train_loader, _ = get_loaders(dataset, batch_size=batch_size, train_subset=n_batches * batch_size + batch_size)
    model = build_model(model_name, dataset).to(device)
    if init_checkpoint:
        model.load_state_dict(torch.load(init_checkpoint, map_location=device))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    base_loss = build_base_loss(base_loss_name)
    base_loss_kwargs = {"pgd_steps": pgd_steps} if base_loss_name != "ce" else {}
    eps = DEFAULT_EPSILON[dataset]
    cfg = SARAConfig(epsilon=eps, tau=0.05, xi=0.5, kappa=kappa)

    # warm-up iterations, excluded from timing
    it = iter(train_loader)
    for _ in range(2):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        sara_training_step(model, optimizer, x, y, base_loss, cfg, base_loss_kwargs, use_sara=use_sara)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    succ_rates = []
    n_challenging = []
    it = iter(train_loader)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        torch.cuda.synchronize()
        s = time.perf_counter()
        stats = sara_training_step(model, optimizer, x, y, base_loss, cfg, base_loss_kwargs, use_sara=use_sara)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - s)
        succ_rates.append(stats["success_rate"])
        n_challenging.append(stats["n_challenging"])
    total_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "dataset": dataset, "model": model_name, "base_loss": base_loss_name,
        "use_sara": use_sara, "kappa": kappa if use_sara else 0,
        "mean_batch_time_ms": 1000 * sum(times) / len(times),
        "total_time_s": total_time,
        "peak_mem_mb": peak_mem,
        "mean_success_rate": sum(succ_rates) / len(succ_rates),
        "mean_n_challenging": sum(n_challenging) / len(n_challenging),
        "batch_size": batch_size,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--model", default="lenet")
    p.add_argument("--base-loss", default="ce")
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "overhead", "overhead.jsonl"))
    p.add_argument("--init-checkpoint", default=None)
    args = p.parse_args()

    results = []
    print("Benchmarking baseline (no SARA)...")
    results.append(bench(args.dataset, args.model, args.base_loss, False, 0, args.n_batches, args.batch_size,
                          init_checkpoint=args.init_checkpoint))
    for kappa in [3, 5, 10, 20]:
        print(f"Benchmarking SARA kappa={kappa}...")
        results.append(bench(args.dataset, args.model, args.base_loss, True, kappa, args.n_batches, args.batch_size,
                              init_checkpoint=args.init_checkpoint))

    with open(args.out, "a") as f:
        for r in results:
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")
