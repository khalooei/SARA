"""Core SARA mechanism: identification of challenging samples, the Confidence Guard
(CG) loss, and the batched inner correction procedure (Algorithm 1).

Sample-level Confidence Guard loss (Eq. 7a of the paper):

    L_CG(x_hat, y; theta) = CE(f_theta(x_hat), y) + lambda_margin * relu(h_theta(x_hat, y) + tau)

where h_theta(x, y) = max_{j != y} f_theta,j(x) - f_theta,y(x) is the classification
margin computed on softmax probabilities. The hinge term vanishes once the margin
condition h_theta <= -tau holds, so the search keeps receiving a gradient until the
acceptance criterion of Eq. 7 is met.

Implementation notes:

* Corrected samples are detached before they enter the outer objective (Eq. 9), so
  no gradient flows through the inner search, as in standard adversarial training.
* Every inner iteration projects onto the l_inf ball of radius epsilon around x_s
  and onto the valid pixel range [0, 1]; the perturbation is re-derived from the
  clipped image, which is the exact Euclidean projection onto their intersection.
* The inner loop processes all challenging samples of a batch as one tensor
  operation (one forward/backward pass per inner iteration), so the cost of SARA
  scales with |S| rather than with the batch size.
* The search runs with the model in evaluation mode, which decouples the samples and
  keeps batch normalization well defined for very small challenging subsets.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .utils import amp_ctx


@dataclass
class SARAConfig:
    epsilon: float = 0.2          # perturbation budget of the correction search
    tau: float = 0.05             # margin threshold
    xi: float = 0.5               # confidence threshold
    kappa: int = 10               # maximum number of inner iterations
    alpha: float = None           # inner step size; defaults to 2 * epsilon / kappa
    lambda_margin: float = 1.0    # weight of the hinge term in L_CG
    random_init: bool = True      # random initialization inside the admissible set
    max_batch_correct: int = None  # correction budget: max. challenging samples per batch
    gamma_reweight: float = 0.0   # confidence-based reweighting of the base loss (0 = off)
    sara_warmup_epochs: int = 0   # epochs trained without the correction mechanism
    cg_damping: bool = False      # scale the CG loss by (1 - correction success rate)
    cg_confidence_cap: bool = False  # cap the cross-entropy term of the outer CG loss at xi

    def step_size(self) -> float:
        return self.alpha if self.alpha is not None else (2.0 * self.epsilon / max(self.kappa, 1))


def scale_xi_for_num_classes(xi_at_10_classes: float, num_classes: int) -> float:
    """Express the confidence threshold relative to the chance level.

    A fixed threshold becomes much stricter as the number of classes grows
    (xi = 0.5 is 5x chance for 10 classes but 50x chance for 100 classes). The
    threshold is therefore defined as a fixed multiple of chance level,
    k = 10 * xi_at_10_classes, and returned as k / num_classes. The function leaves
    10-class problems unchanged: scale_xi_for_num_classes(0.5, 10) == 0.5.
    """
    k = xi_at_10_classes * 10.0
    return k / num_classes


def margin(logits_or_probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """h_theta(x, y) = max_{j != y} f_theta,j(x) - f_theta,y(x)   (Eq. 7)."""
    y_score = logits_or_probs.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits_or_probs.clone()
    masked.scatter_(1, y.view(-1, 1), float("-inf"))
    other_max = masked.max(dim=1).values
    return other_max - y_score


def identify_challenging(model, x: torch.Tensor, y: torch.Tensor, xi: float):
    """S = {(x_i, y_i) in B | argmax_j f_theta,j(x_i) != y_i  or  f_theta,y_i(x_i) < xi}."""
    mask, _ = identify_challenging_and_confidence(model, x, y, xi)
    return mask


def identify_challenging_and_confidence(model, x: torch.Tensor, y: torch.Tensor, xi: float):
    """Return the challenging-sample mask together with the true-class probability
    f_theta,y(x) of every sample in the batch."""
    model.eval()
    with torch.no_grad(), amp_ctx():
        probs = F.softmax(model(x), dim=1).float()
    model.train()
    pred = probs.argmax(dim=1)
    py = probs.gather(1, y.view(-1, 1)).squeeze(1)
    mask = (pred != y) | (py < xi)
    return mask, py


def continuous_confidence_weight(py: torch.Tensor, xi: float, gamma: float) -> torch.Tensor:
    """Per-sample weights for the base loss that increase as the true-class
    confidence falls below xi:

        w_i = 1 + gamma * relu(xi - py_i) / xi,   renormalized to mean 1.

    The weights reuse the confidences computed when identifying the challenging set,
    so they add no forward pass. gamma = 0 recovers uniform weighting.
    """
    if gamma <= 0:
        return None
    w = 1.0 + gamma * torch.relu(xi - py) / max(xi, 1e-8)
    w = w * w.numel() / w.sum().clamp_min(1e-8)
    return w.detach()


def cg_damping_weight(success_rate: float) -> float:
    """Scalar factor (1 - rho) applied to the CG loss, where rho is the fraction of
    processed challenging samples that were corrected successfully (Eq. 10)."""
    return max(1.0 - success_rate, 0.0)


def sample_level_cg_loss(probs: torch.Tensor, y: torch.Tensor, tau: float, lambda_margin: float) -> torch.Tensor:
    """L_CG(x_hat, y; theta) per sample (Eq. 7a), not yet averaged."""
    ce = F.nll_loss(torch.log(probs.clamp_min(1e-12)), y, reduction="none")
    h = margin(probs, y)
    hinge = F.relu(h + tau)
    return ce + lambda_margin * hinge


def correct_challenging_samples(model, x_s: torch.Tensor, y_s: torch.Tensor, cfg: SARAConfig):
    """Batched inner correction of Algorithm 1.

    Returns (x_star, y_star, stats): the successfully corrected samples S*, their
    labels, and per-batch statistics (number of challenging and processed samples,
    number of corrections, success rate, and inner iterations used).
    """
    if x_s.numel() == 0:
        empty = x_s.new_zeros((0, *x_s.shape[1:]))
        return empty, y_s.new_zeros((0,), dtype=y_s.dtype), {"success_rate": 0.0, "iters_used": 0}

    # Correction budget: when more samples are challenging than the budget allows, a
    # random subset is corrected. This bounds the memory of the inner search; at a
    # fixed kappa its latency is dominated by the number of inner iterations.
    uncapped_M = x_s.shape[0]
    if cfg.max_batch_correct is not None and uncapped_M > cfg.max_batch_correct:
        idx = torch.randperm(uncapped_M, device=x_s.device)[: cfg.max_batch_correct]
        x_s, y_s = x_s[idx], y_s[idx]

    M = x_s.shape[0]
    device = x_s.device
    alpha = cfg.step_size()

    # The search runs in evaluation mode, as adversarial-example generation does, so
    # that batch normalization uses running statistics even for a single sample.
    was_training = model.training
    model.eval()

    if cfg.random_init:
        delta = torch.empty_like(x_s).uniform_(-cfg.epsilon, cfg.epsilon)
    else:
        delta = torch.zeros_like(x_s)
    delta = torch.clamp(x_s + delta, 0.0, 1.0) - x_s  # initial point inside the admissible set

    x_star = x_s.clone().detach()
    done = torch.zeros(M, dtype=torch.bool, device=device)
    iters_used = 0

    for k in range(cfg.kappa):
        iters_used = k + 1
        delta = delta.detach().requires_grad_(True)
        x_hat = torch.clamp(x_s + delta, 0.0, 1.0)

        with amp_ctx():
            logits = model(x_hat)
        # Probabilities, logarithms and threshold comparisons are computed in fp32.
        probs = F.softmax(logits.float(), dim=1)
        py = probs.gather(1, y_s.view(-1, 1)).squeeze(1)
        h = margin(probs, y_s)

        newly_satisfied = (~done) & (py >= cfg.xi) & (h <= -cfg.tau)
        if newly_satisfied.any():
            x_star[newly_satisfied] = x_hat[newly_satisfied].detach()
            done = done | newly_satisfied
        if done.all():
            break

        per_sample_loss = sample_level_cg_loss(probs, y_s, cfg.tau, cfg.lambda_margin)
        per_sample_loss = per_sample_loss * (~done).float()  # accepted samples receive no further updates
        loss = per_sample_loss.sum()

        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - alpha * grad
            delta = torch.clamp(delta, -cfg.epsilon, cfg.epsilon)
            x_hat_next = torch.clamp(x_s + delta, 0.0, 1.0)
            delta = x_hat_next - x_s  # projection onto the eps-ball and the valid pixel range

    if was_training:
        model.train()

    success_mask = done
    x_star_sel = x_star[success_mask].detach()
    y_star_sel = y_s[success_mask].detach()
    stats = {
        "n_challenging": uncapped_M,
        "n_processed": M,
        "n_corrected": int(success_mask.sum().item()),
        "success_rate": float(success_mask.float().mean().item()),
        "iters_used": iters_used,
    }
    return x_star_sel, y_star_sel, stats
