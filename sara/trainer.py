"""Outer SARA training step (Algorithm 1, Eq. 9): Loss_SARA = Loss_base(B) + Loss_CG(S*).

Two optional extensions are available (Section 3.3 of the paper):

* Revisiting (``cfg.revisit_memory > 0``): accepted corrections are stored in a
  memory and re-scored under the current parameters with the confidence-capped CG
  loss (Eq. 11), which is non-zero only for stored samples that have left the
  acceptance region since they were accepted.
* Robust correction (``cfg.robust_correction``): corrected samples are appended to
  the mini-batch of the base loss, so that an adversarial base loss also trains on
  them.
"""
import math
import time

import torch
import torch.nn.functional as F

from .confidence_guard import (
    CorrectionMemory,
    SARAConfig,
    cg_damping_weight,
    continuous_confidence_weight,
    correct_challenging_samples,
    identify_challenging_and_confidence,
    margin,
    sample_level_cg_loss,
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


def _eval_mode_probs(model, x):
    """Softmax outputs in evaluation mode, with a gradient path to the parameters."""
    was_training = model.training
    model.eval()
    with amp_ctx():
        logits = model(x)
    if was_training:
        model.train()
    return F.softmax(logits.float(), dim=1)


def sara_training_step(model, optimizer, x: torch.Tensor, y: torch.Tensor, base_loss,
                        cfg: SARAConfig, base_loss_kwargs=None, use_sara: bool = True, epoch: int = None,
                        diag_grad_norms: bool = False, memory: CorrectionMemory = None):
    """One SARA (or plain) mini-batch update. Returns a dict of statistics for logging.

    Args:
        use_sara: if False, the step reduces to training with the base loss only.
        epoch: 1-indexed epoch, used by cfg.sara_warmup_epochs; the correction
            mechanism is disabled while epoch <= sara_warmup_epochs.
        diag_grad_norms: if True, additionally record ||grad Loss_base|| and
            ||grad Loss_CG|| (two extra backward passes per step).
        memory: memory of accepted corrections, required when cfg.revisit_memory > 0.
    """
    base_loss_kwargs = base_loss_kwargs or {}
    t0 = time.perf_counter()

    stats = {"n_challenging": 0, "n_corrected": 0, "success_rate": 0.0, "iters_used": 0,
             "n_revisited": 0, "revisit_violation_rate": 0.0}

    in_warmup = cfg.sara_warmup_epochs > 0 and epoch is not None and epoch <= cfg.sara_warmup_epochs
    sample_weight = None
    x_rev = y_rev = None

    if use_sara and not in_warmup:
        challenging_mask, py_all = identify_challenging_and_confidence(model, x, y, cfg.xi)
        x_s, y_s = x[challenging_mask], y[challenging_mask]
        x_star, y_star, corr_stats, star_idx = correct_challenging_samples(model, x_s, y_s, cfg,
                                                                            return_index=True)
        # Positions of the corrected samples in the mini-batch.
        star_pos = challenging_mask.nonzero().squeeze(1)[star_idx]
        stats.update(corr_stats)
        if cfg.gamma_reweight > 0 and getattr(base_loss, "supports_sample_weight", False):
            sample_weight = continuous_confidence_weight(py_all, cfg.xi, cfg.gamma_reweight)
        if memory is not None:
            # Samples are drawn before the current corrections are stored, so every
            # revisited sample was accepted under earlier parameters.
            x_rev, y_rev = memory.sample(cfg.revisit_batch)
            memory.push(x_star, y_star)
    else:
        x_star, y_star = x.new_zeros((0, *x.shape[1:])), y.new_zeros((0,), dtype=y.dtype)
        star_pos = None

    optimizer.zero_grad(set_to_none=True)

    if x_star.shape[0] > 0:
        # Corrected samples are scored in evaluation mode, with the same batch
        # normalization statistics under which they were accepted. This forward pass
        # precedes the base loss, whose training-mode forward pass updates the running
        # statistics.
        if cfg.cg_confidence_cap:
            # Eq. 11 caps the cross-entropy term at xi. Every sample of S* was accepted under the current
            # parameters, so its capped loss is exactly -log(xi) with zero gradient (Remark 1). The term is
            # therefore taken at that value: re-scoring the samples in a separate forward pass moves some of them
            # just outside the acceptance region through floating-point differences (another batch composition,
            # bfloat16), and the mean over S* would give each such sample the weight of the whole batch.
            # The prediction at the corrections is still needed, without gradient, as the alignment target.
            with torch.no_grad():
                probs_star = _eval_mode_probs(model, x_star)
            loss_cg = probs_star.new_tensor(-math.log(cfg.xi))
        else:
            probs_star = _eval_mode_probs(model, x_star)
            loss_cg = sample_level_cg_loss(probs_star, y_star, cfg.tau, cfg.lambda_margin).mean()
        if cfg.cg_damping:
            loss_cg = loss_cg * cg_damping_weight(stats["success_rate"])
    else:
        loss_cg = torch.zeros((), device=x.device)

    if x_rev is not None:
        probs_rev = _eval_mode_probs(model, x_rev)
        per_rev = sample_level_cg_loss(probs_rev, y_rev, cfg.tau, cfg.lambda_margin, xi_cap=cfg.xi)
        with torch.no_grad():
            py_rev = probs_rev.gather(1, y_rev.view(-1, 1)).squeeze(1)
            left = (py_rev < cfg.xi) | (margin(probs_rev, y_rev) > -cfg.tau)
        # The constant -log(xi) of samples still inside the acceptance region is
        # subtracted, so the logged loss is zero when no stored sample has left it.
        loss_rev = cfg.revisit_weight * (per_rev + math.log(cfg.xi)).mean()
        stats["n_revisited"] = int(x_rev.shape[0])
        stats["revisit_violation_rate"] = float(left.float().mean().item())
    else:
        loss_rev = torch.zeros((), device=x.device)

    if cfg.robust_correction and x_star.shape[0] > 0:
        x_base, y_base = torch.cat([x, x_star]), torch.cat([y, y_star])
        if sample_weight is not None:
            sample_weight = torch.cat([sample_weight, sample_weight.new_ones(x_star.shape[0])])
    else:
        x_base, y_base = x, y

    if sample_weight is not None:
        loss_base = base_loss.compute(model, x_base, y_base, cfg.base_eps(), sample_weight=sample_weight,
                                      **base_loss_kwargs)
    else:
        loss_base = base_loss.compute(model, x_base, y_base, cfg.base_eps(), **base_loss_kwargs)

    # The natural forward uses the whole batch. A forward on the corrected subset alone
    # fails in BatchNorm when that subset has a single sample and the feature map is 1x1.
    logits_nat = None
    if cfg.nat_weight > 0 or (x_star.shape[0] > 0 and cfg.align_natural_weight > 0):
        with amp_ctx():
            logits_nat = model(x)

    # Alignment (Eq. 10): the prediction at the adversarial example of each corrected sample, and
    # optionally at the natural sample itself, is pulled toward the prediction at its accepted
    # correction, which serves as a fixed target.
    loss_align = torch.zeros((), device=x.device)
    if x_star.shape[0] > 0 and (cfg.align_weight > 0 or cfg.align_natural_weight > 0):
        target = probs_star.detach()
        log_target = torch.log(target.clamp_min(1e-12))
        adv_logits = getattr(base_loss, "last_adv_logits", None)
        adv_x = getattr(base_loss, "last_adv_x", None)
        if cfg.align_weight > 0 and cfg.align_eval_mode and adv_x is not None:
            # Target and adversarial prediction under the same (evaluation-mode) network, as in
            # Proposition 4.
            log_p_adv = torch.log(_eval_mode_probs(model, adv_x[star_pos]).clamp_min(1e-12))
            loss_align = loss_align + cfg.align_weight * (target * (log_target - log_p_adv)).sum() / x.shape[0]
        elif cfg.align_weight > 0 and adv_logits is not None:
            log_p_adv = F.log_softmax(adv_logits[star_pos].float(), dim=1)
            loss_align = loss_align + cfg.align_weight * (target * (log_target - log_p_adv)).sum() / x.shape[0]
        if cfg.align_natural_weight > 0:
            log_p_nat = F.log_softmax(logits_nat[star_pos].float(), dim=1)
            loss_align = loss_align + cfg.align_natural_weight * (target * (log_target - log_p_nat)).sum() / x.shape[0]

    if diag_grad_norms:
        params = [p for p in model.parameters() if p.requires_grad]
        stats["grad_norm_base"] = _grad_norm(loss_base, params)
        stats["grad_norm_cg"] = _grad_norm(loss_cg, params) if loss_cg.requires_grad else 0.0

    if cfg.nat_weight > 0:
        loss_nat = cfg.nat_weight * F.cross_entropy(logits_nat.float(), y)
    else:
        loss_nat = torch.zeros((), device=x.device)

    loss_total = loss_base + loss_cg + loss_rev + loss_align + loss_nat
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
    stats["loss_revisit"] = float(loss_rev.item())
    stats["loss_align"] = float(loss_align.item())
    stats["loss_nat"] = float(loss_nat.item())
    stats["loss_total"] = float(loss_total.item())
    stats["batch_size_effective"] = int(x.shape[0] + x_star.shape[0])
    stats["step_time_s"] = time.perf_counter() - t0
    return stats
