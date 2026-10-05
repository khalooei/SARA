"""Clean, PGD and APGD accuracy evaluation."""
import torch

from .attacks import apgd_attack, pgd_attack
from .utils import amp_ctx


@torch.no_grad()
def _accuracy(model, x, y):
    with amp_ctx():
        logits = model(x)
    return (logits.argmax(dim=1) == y).float().sum().item()


def evaluate(model, loader, device, epsilon, pgd_steps=20, pgd_alpha=None, run_apgd=True, apgd_iters=100,
             max_batches=None):
    model.eval()
    pgd_alpha = pgd_alpha or (epsilon / 4.0)
    n = 0
    correct_clean = 0
    correct_pgd = 0
    correct_apgd = 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        n += x.shape[0]
        correct_clean += _accuracy(model, x, y)

        x_adv = pgd_attack(model, x, y, epsilon, pgd_alpha, pgd_steps)
        correct_pgd += _accuracy(model, x_adv, y)

        if run_apgd:
            x_apgd = apgd_attack(model, x, y, epsilon, n_iter=apgd_iters)
            correct_apgd += _accuracy(model, x_apgd, y)

    out = {
        "n": n,
        "clean_acc": 100.0 * correct_clean / n,
        "pgd_acc": 100.0 * correct_pgd / n,
    }
    if run_apgd:
        out["apgd_acc"] = 100.0 * correct_apgd / n
    return out
