"""Outer SARA training step (Algorithm 1, Eq. 9): Loss_SARA = Loss_base(B) + Loss_CG(S*)."""
import time

import torch
import torch.nn.functional as F

from .confidence_guard import (
    SARAConfig,
    cg_damping_weight,
    continuous_confidence_weight,
    correct_challenging_samples,
    identify_challenging_and_confidence,
    margin,
)
from .utils import amp_ctx


def _grad_norm(loss, params):
    """L2 norm of d(loss)/d(params), keeping the graph for the subsequent backward pass.

    Parameters without a gradient path (for example, the CG loss of a batch with no
    corrected sample) contribute zero.
    """
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = sum(g.pow(2).sum() for g in grads if g is not None)
    return float(sq.sqrt().item()) if isinstance(sq, torch.Tensor) else 0.0


def sara_training_step(model, optimizer, x: torch.Tensor, y: torch.Tensor, base_loss,
                        cfg: SARAConfig, base_loss_kwargs=None, use_sara: bool = True, epoch: int = None,
                        diag_grad_norms: bool = False):
    """One SARA (or plain) mini-batch update. Returns a dict of statistics for logging.

    Args:
        use_sara: if False, the step reduces to training with the base loss only.
        epoch: 1-indexed epoch, used by cfg.sara_warmup_epochs; the correction
            mechanism is disabled while epoch <= sara_warmup_epochs.
        diag_grad_norms: if True, additionally record ||grad Loss_base|| and
            ||grad Loss_CG|| (two extra backward passes per step).
    """
    base_loss_kwargs = base_loss_kwargs or {}
    t0 = time.perf_counter()

    stats = {"n_challenging": 0, "n_corrected": 0, "success_rate": 0.0, "iters_used": 0}

    in_warmup = cfg.sara_warmup_epochs > 0 and epoch is not None and epoch <= cfg.sara_warmup_epochs
    sample_weight = None

    if use_sara and not in_warmup:
        challenging_mask, py_all = identify_challenging_and_confidence(model, x, y, cfg.xi)
        x_s, y_s = x[challenging_mask], y[challenging_mask]
        x_star, y_star, corr_stats = correct_challenging_samples(model, x_s, y_s, cfg)
        stats.update(corr_stats)
        if cfg.gamma_reweight > 0 and getattr(base_loss, "supports_sample_weight", False):
            sample_weight = continuous_confidence_weight(py_all, cfg.xi, cfg.gamma_reweight)
    else:
        x_star, y_star = x.new_zeros((0, *x.shape[1:])), y.new_zeros((0,), dtype=y.dtype)

    optimizer.zero_grad(set_to_none=True)

    if x_star.shape[0] > 0:
        # Corrected samples are scored in evaluation mode, with the same batch
        # normalization statistics under which they were accepted. This forward pass
        # precedes the base loss, whose training-mode forward pass updates the running
        # statistics.
        was_training = model.training
        model.eval()
        with amp_ctx():
            logits_star = model(x_star)
        if was_training:
            model.train()
        probs_star = F.softmax(logits_star.float(), dim=1)
        py_star = probs_star.gather(1, y_star.view(-1, 1)).squeeze(1)
        if cfg.cg_confidence_cap:
            # Confidence-capped cross-entropy (Eq. 11): the gradient with respect to
            # py vanishes once py > xi, so accepted samples are not pushed beyond the
            # confidence required by the acceptance criterion.
            py_for_ce = torch.clamp(py_star, max=cfg.xi)
        else:
            py_for_ce = py_star
        ce = -torch.log(py_for_ce.clamp_min(1e-12))
        h = margin(probs_star, y_star)
        hinge = torch.relu(h + cfg.tau)
        loss_cg = (ce + cfg.lambda_margin * hinge).mean()
        if cfg.cg_damping:
            loss_cg = loss_cg * cg_damping_weight(stats["success_rate"])
    else:
        loss_cg = torch.zeros((), device=x.device)

    if sample_weight is not None:
        loss_base = base_loss.compute(model, x, y, cfg.epsilon, sample_weight=sample_weight, **base_loss_kwargs)
    else:
        loss_base = base_loss.compute(model, x, y, cfg.epsilon, **base_loss_kwargs)

    if diag_grad_norms:
        params = [p for p in model.parameters() if p.requires_grad]
        stats["grad_norm_base"] = _grad_norm(loss_base, params)
        stats["grad_norm_cg"] = _grad_norm(loss_cg, params) if loss_cg.requires_grad else 0.0

    loss_total = loss_base + loss_cg
    loss_total.backward()
    # Global gradient-norm clipping, and a skipped update if a non-finite loss or
    # gradient occurs (the boosted cross-entropy of MART can occasionally overflow).
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    if torch.isfinite(loss_total) and torch.isfinite(grad_norm):
        optimizer.step()
    else:
        optimizer.zero_grad(set_to_none=True)

    stats["loss_base"] = float(loss_base.item())
    stats["loss_cg"] = float(loss_cg.item())
    stats["loss_total"] = float(loss_total.item())
    stats["batch_size_effective"] = int(x.shape[0] + x_star.shape[0])
    stats["step_time_s"] = time.perf_counter() - t0
    return stats
