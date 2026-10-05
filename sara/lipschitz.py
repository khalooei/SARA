"""Empirical estimate of the local Lipschitz constant, used to relate SARA's
hyperparameters to the smoothness of the decision function.

We estimate, for each probe point x, the local Lipschitz constant of the
softmax output f_theta as
    L(x) ~= max_{||d||_inf <= r, over n_dirs random directions} ||f(x+d) - f(x)||_2 / ||d||_2
and report the mean over a batch of test points. This is the standard
random-direction finite-difference estimator used in prior local-smoothness
studies. It is a lower bound on the true Lipschitz constant, inexpensive and
reproducible, and is intended for comparing trends across configurations.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def local_lipschitz_estimate(model, x: torch.Tensor, radius: float = 0.03, n_dirs: int = 10) -> float:
    model.eval()
    f0 = F.softmax(model(x), dim=1)
    best = torch.zeros(x.shape[0], device=x.device)
    for _ in range(n_dirs):
        d = torch.empty_like(x).uniform_(-radius, radius)
        x_pert = torch.clamp(x + d, 0, 1)
        d_eff = (x_pert - x).flatten(1)
        f1 = F.softmax(model(x_pert), dim=1)
        num = (f1 - f0).norm(dim=1)
        den = d_eff.norm(dim=1).clamp_min(1e-8)
        ratio = num / den
        best = torch.maximum(best, ratio)
    return float(best.mean().item())
