"""Attacks: PGD for adversarial training and evaluation, and APGD / AutoAttack
(via torchattacks) for robustness evaluation."""
import torch
import torch.nn.functional as F

from .utils import amp_ctx


def pgd_attack(model, x, y, epsilon, alpha, steps, loss_fn: str = "ce", random_start: bool = True):
    """Standard L_inf PGD (Madry et al., 2018). Returns adversarial examples in [0, 1]."""
    x_orig = x.detach()
    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = (x_orig + delta).clamp(0, 1).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad(), amp_ctx():
            logits = model(x_adv)
            if loss_fn == "ce":
                loss = F.cross_entropy(logits.float(), y)
            else:
                raise ValueError(loss_fn)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x_orig - epsilon), x_orig + epsilon)
        x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()


def apgd_attack(model, x, y, epsilon, n_iter: int = 100, version: str = "standard"):
    """APGD with the cross-entropy loss (Croce & Hein, 2020), via torchattacks."""
    import torchattacks
    atk = torchattacks.APGD(model, norm="Linf", eps=epsilon, steps=n_iter, loss="ce", n_restarts=1)
    return atk(x, y)


def autoattack(model, x, y, epsilon, n_classes: int = 10, version: str = "standard"):
    import torchattacks
    atk = torchattacks.AutoAttack(model, norm="Linf", eps=epsilon, version=version, n_classes=n_classes)
    return atk(x, y)
