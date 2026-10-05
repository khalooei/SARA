"""Baseline training steps compared with SARA: Online Hard-Example Mining (OHEM), a
simplified Friendly Adversarial Training (FAT) step, and a simplified Geometry-Aware
Instance-Reweighted Adversarial Training (GAIRAT) step.

All steps share the signature of sara_trainer.sara_training_step, so train.py uses the
same training and evaluation pipeline for every method.
"""
import time

import torch
import torch.nn.functional as F


def ohem_training_step(model, optimizer, x, y, base_loss, cfg, base_loss_kwargs=None, top_frac: float = 0.5):
    """Online Hard-Example Mining (Shrivastava et al., 2016): backpropagate only the
    top-`top_frac` highest-loss samples of the mini-batch. No samples are generated.
    """
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    per_sample = F.cross_entropy(logits, y, reduction="none")
    k = max(1, int(top_frac * x.shape[0]))
    hard_idx = torch.topk(per_sample, k).indices
    loss = per_sample[hard_idx].mean()
    loss.backward()
    optimizer.step()
    return {
        "loss_base": float(loss.item()), "loss_cg": 0.0, "loss_total": float(loss.item()),
        "n_challenging": k, "n_corrected": 0, "success_rate": 0.0, "iters_used": 0,
        "batch_size_effective": x.shape[0], "step_time_s": time.perf_counter() - t0,
    }


def fat_training_step(model, optimizer, x, y, base_loss, cfg, base_loss_kwargs=None, max_steps: int = 10):
    """Simplified Friendly Adversarial Training (Zhang et al., 2020): PGD is stopped for
    each sample as soon as it becomes misclassified (or after max_steps), and the model
    is trained on the resulting adversarial examples.
    """
    t0 = time.perf_counter()
    epsilon = cfg.epsilon
    alpha = epsilon / 4.0
    model.eval()
    x_adv = x.detach().clone()
    delta = torch.zeros_like(x)
    stopped = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for _ in range(max_steps):
        x_cur = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
        logits = model(x_cur)
        pred = logits.argmax(dim=1)
        newly_wrong = (~stopped) & (pred != y)
        x_adv[newly_wrong] = x_cur[newly_wrong].detach()
        stopped = stopped | newly_wrong
        if stopped.all():
            break
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_cur)[0]
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
    x_adv[~stopped] = torch.clamp(x + delta, 0, 1)[~stopped].detach()
    model.train()

    optimizer.zero_grad(set_to_none=True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return {
        "loss_base": float(loss.item()), "loss_cg": 0.0, "loss_total": float(loss.item()),
        "n_challenging": int(stopped.sum().item()), "n_corrected": 0, "success_rate": 0.0, "iters_used": 0,
        "batch_size_effective": x.shape[0], "step_time_s": time.perf_counter() - t0,
    }


def gairat_training_step(model, optimizer, x, y, base_loss, cfg, base_loss_kwargs=None, max_steps: int = 10):
    """Simplified Geometry-Aware Instance-Reweighted Adversarial Training (Zhang et al.,
    2021). During PGD, the first step kappa_i at which each sample becomes misclassified
    is recorded as a proxy for its distance to the decision boundary. The adversarial
    cross-entropy is weighted by w_i = 1 - kappa_i / max_steps, normalized to mean 1,
    so samples that are misclassified early receive larger weights.
    """
    t0 = time.perf_counter()
    epsilon = cfg.epsilon
    alpha = epsilon / 4.0
    model.eval()
    x_adv = x.detach().clone()
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    delta = torch.clamp(x + delta, 0, 1) - x
    B = x.shape[0]
    kappa = torch.full((B,), max_steps, dtype=torch.float32, device=x.device)
    flipped = torch.zeros(B, dtype=torch.bool, device=x.device)

    for step in range(max_steps):
        x_cur = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
        logits = model(x_cur)
        pred = logits.argmax(dim=1)
        newly_wrong = (~flipped) & (pred != y)
        kappa[newly_wrong] = step
        flipped = flipped | newly_wrong
        x_adv = torch.where(newly_wrong.view(-1, *([1] * (x.dim() - 1))), x_cur.detach(), x_adv)

        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_cur)[0]
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(x + delta, 0, 1) - x
    x_adv = torch.where(flipped.view(-1, *([1] * (x.dim() - 1))), x_adv, torch.clamp(x + delta, 0, 1))
    model.train()

    weight = 1.0 - kappa / max_steps
    weight = weight * B / weight.sum().clamp_min(1e-8)  # normalize to mean weight 1

    optimizer.zero_grad(set_to_none=True)
    logits = model(x_adv)
    per_sample = F.cross_entropy(logits, y, reduction="none")
    loss = (weight * per_sample).mean()
    loss.backward()
    optimizer.step()
    return {
        "loss_base": float(loss.item()), "loss_cg": 0.0, "loss_total": float(loss.item()),
        "n_challenging": int(flipped.sum().item()), "n_corrected": 0, "success_rate": 0.0, "iters_used": 0,
        "batch_size_effective": x.shape[0], "step_time_s": time.perf_counter() - t0,
    }
