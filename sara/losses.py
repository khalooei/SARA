"""Base training losses that SARA can be combined with: cross-entropy, PGD adversarial
training, TRADES and MART (Eqs. 1, 3, 4 and 5 of the paper).

Each loss exposes ``compute(model, x, y, epsilon)`` and returns a scalar, so that
``sara_trainer.sara_training_step`` can use any of them as the base loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import pgd_attack
from .utils import amp_ctx


def boosted_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Boosted Cross-Entropy (BCE) from MART (Wang et al., 2020):
    BCE(p, y) = -log(p_y) - log(1 - max_{i != y} p_i)
    """
    probs = F.softmax(logits, dim=1)
    py = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(1e-12)
    probs_masked = probs.clone()
    probs_masked.scatter_(1, y.view(-1, 1), -1.0)
    p_other_max = probs_masked.max(dim=1).values.clamp_min(0.0).clamp_max(1.0 - 1e-12)
    return -torch.log(py) - torch.log((1.0 - p_other_max).clamp_min(1e-12))


class BaseLoss:
    name = "base"
    supports_sample_weight = False

    def compute(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float,
                pgd_steps: int = 10, pgd_alpha: float = None, sample_weight: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class StandardLoss(BaseLoss):
    """Cross-entropy on clean inputs (Eq. 1)."""
    name = "ce"
    supports_sample_weight = True

    def compute(self, model, x, y, epsilon, pgd_steps=10, pgd_alpha=None, sample_weight=None):
        with amp_ctx():
            logits = model(x)
        per_sample = F.cross_entropy(logits.float(), y, reduction="none")
        if sample_weight is not None:
            return (sample_weight * per_sample).mean()
        return per_sample.mean()


class PGDATLoss(BaseLoss):
    """PGD adversarial training (Madry et al., 2018; Eq. 3)."""
    name = "pgd"
    supports_sample_weight = True

    def compute(self, model, x, y, epsilon, pgd_steps=10, pgd_alpha=None, sample_weight=None):
        pgd_alpha = pgd_alpha or (epsilon / 4.0)
        model.eval()
        x_adv = pgd_attack(model, x, y, epsilon, pgd_alpha, pgd_steps)
        model.train()
        with amp_ctx():
            logits = model(x_adv)
        per_sample = F.cross_entropy(logits.float(), y, reduction="none")
        if sample_weight is not None:
            return (sample_weight * per_sample).mean()
        return per_sample.mean()


class TRADESLoss(BaseLoss):
    """TRADES (Zhang et al., 2019; Eq. 4)."""
    name = "trades"

    def __init__(self, beta: float = 6.0):
        self.beta = beta

    def compute(self, model, x, y, epsilon, pgd_steps=10, pgd_alpha=None):
        pgd_alpha = pgd_alpha or (epsilon / 4.0)
        model.eval()
        with torch.no_grad(), amp_ctx():
            logits_nat = model(x).float()
        x_adv = x.detach() + 0.001 * torch.randn_like(x)
        x_adv = x_adv.clamp(0, 1)
        for _ in range(pgd_steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad(), amp_ctx():
                kl = F.kl_div(F.log_softmax(model(x_adv).float(), dim=1),
                              F.softmax(logits_nat, dim=1), reduction="batchmean")
            grad = torch.autograd.grad(kl, x_adv)[0]
            x_adv = x_adv.detach() + pgd_alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon).clamp(0, 1)
        model.train()
        with amp_ctx():
            logits = model(x).float()
            logits_adv = model(x_adv).float()
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1),
                                reduction="batchmean")
        return loss_natural + self.beta * loss_robust


class MARTLoss(BaseLoss):
    """MART (Wang et al., 2020; Eq. 5)."""
    name = "mart"

    def __init__(self, beta: float = 6.0):
        self.beta = beta

    def compute(self, model, x, y, epsilon, pgd_steps=10, pgd_alpha=None):
        pgd_alpha = pgd_alpha or (epsilon / 4.0)
        model.eval()
        x_adv = pgd_attack(model, x, y, epsilon, pgd_alpha, pgd_steps, loss_fn="ce")
        model.train()
        with amp_ctx():
            logits = model(x).float()
            logits_adv = model(x_adv).float()
        bce = boosted_ce(logits_adv, y).mean()
        kl = F.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1),
                       reduction="none").sum(dim=1)
        py = F.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1)
        penalty = (kl * (1.0 - py)).mean()
        return bce + self.beta * penalty


BASE_LOSSES = {
    "ce": StandardLoss,
    "pgd": PGDATLoss,
    "trades": TRADESLoss,
    "mart": MARTLoss,
}


def build_base_loss(name: str) -> BaseLoss:
    if name not in BASE_LOSSES:
        raise ValueError(f"Unknown base loss: {name}")
    return BASE_LOSSES[name]()
