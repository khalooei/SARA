"""Train a classifier with or without SARA and evaluate its clean and adversarial accuracy.

Each run writes its configuration (config.json), one JSON line per epoch (train_log.jsonl)
and the final weights (model_final.pt) to --out-dir.

Example:
    python train.py --dataset mnist --model lenet --base-loss pgd --sara \
        --epochs 20 --eval-apgd --out-dir runs/mnist/at_pgd_sara
"""
import argparse
import json
import os
import time

import torch
import torch.optim as optim

from sara.baselines import fat_training_step, gairat_training_step, ohem_training_step
from sara.confidence_guard import CorrectionMemory, SARAConfig, scale_xi_for_num_classes
from sara.data import DEFAULT_EPSILON, NUM_CLASSES, get_gpu_loaders, get_loaders
from sara.evaluate import evaluate, evaluate_autoattack
from sara.losses import build_base_loss
from sara.models import build_model
from sara.trainer import sara_training_step
from sara.utils import AverageMeter, JsonlLogger, ModelEMA, gpu_mem_mb, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100", "svhn", "tinyimagenet"], required=True)
    p.add_argument("--model", choices=["lenet", "resnet18", "resnet18-cifar"], required=True)
    p.add_argument("--base-loss", choices=["ce", "pgd", "trades", "mart", "fat"], default="ce")
    p.add_argument("--sara", action="store_true")
    p.add_argument("--method", choices=["default", "ohem", "fat", "gairat"], default="default",
                    help="'default': base-loss training, with SARA if --sara is set; "
                         "'ohem', 'fat', 'gairat': baseline training steps from baselines.py")
    p.add_argument("--ohem-top-frac", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)

    # AT / base-loss attack params
    p.add_argument("--at-epsilon", type=float, default=None)
    p.add_argument("--pgd-steps", type=int, default=7)

    # SARA / CG hyperparameters
    p.add_argument("--sara-epsilon", type=float, default=None)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--xi", type=float, default=0.5,
                    help="Confidence threshold for a 10-class problem. With --xi-relative "
                         "(default), it is rescaled to xi*10/num_classes, a fixed multiple of "
                         "chance level (unchanged for 10-class datasets).")
    p.add_argument("--xi-relative", dest="xi_relative", action="store_true", default=True)
    p.add_argument("--no-xi-relative", dest="xi_relative", action="store_false")
    p.add_argument("--kappa", type=int, default=10)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--lambda-margin", type=float, default=1.0)
    p.add_argument("--sara-max-correct", type=int, default=0,
                    help="Correction budget: maximum number of challenging samples processed "
                         "by the inner loop per batch (0 disables the budget).")
    p.add_argument("--sara-warmup-epochs", type=int, default=0,
                    help="Number of initial epochs trained with the base loss only, before the "
                         "correction mechanism is enabled.")
    p.add_argument("--gamma-reweight", type=float, default=0.0,
                    help="Strength of the confidence-based reweighting of the base loss "
                         "(0 disables it). See confidence_guard.continuous_confidence_weight.")
    p.add_argument("--cg-damping", action="store_true",
                    help="Scale the CG loss by (1 - correction success rate) (Eq. 10).")
    p.add_argument("--diag-grad-norms", action="store_true",
                    help="Log the gradient norms of the base loss and of the CG loss (epoch "
                         "means). Costs two extra backward passes per step.")
    p.add_argument("--cg-confidence-cap", action="store_true",
                    help="Cap the cross-entropy term of the outer CG loss at xi (Eq. 11).")
    p.add_argument("--robust-correction", action="store_true",
                    help="Append the corrected samples to the mini-batch of the base loss, so that an "
                         "adversarial base loss also trains on them (Eq. 12).")
    p.add_argument("--revisit-memory", type=int, default=0,
                    help="Capacity of the memory of accepted corrections that are revisited under the "
                         "current parameters with the capped CG loss (Eq. 13); 0 disables revisiting.")
    p.add_argument("--revisit-batch", type=int, default=64,
                    help="Number of stored corrections revisited per step.")
    p.add_argument("--sara-align", type=float, default=0.0,
                    help="Weight of the alignment of the adversarial prediction of each corrected sample "
                         "to the prediction at its correction (Eq. 14); requires an adversarial base loss.")
    p.add_argument("--sara-align-natural", type=float, default=0.0,
                    help="Weight of the alignment of the natural prediction of each corrected sample to the "
                         "prediction at its correction; also applicable without an adversarial base loss.")
    p.add_argument("--revisit-weight", type=float, default=1.0,
                    help="Weight of the capped CG loss on revisited corrections.")
    p.add_argument("--nat-weight", type=float, default=0.0,
                    help="Weight of an extra cross-entropy term on the natural mini-batch; "
                         "recovers clean accuracy of PGD-AT without changing the attack.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                    help="Keep an exponential moving average of the weights with this decay (0 = off). The final "
                         "evaluation then uses the averaged weights; the raw weights are evaluated on clean and PGD "
                         "accuracy.")
    p.add_argument("--at-eps-warmup", type=int, default=0,
                    help="Ramp the training and correction radii linearly to their final values over this many "
                         "epochs (0 = full radius from the first epoch).")
    p.add_argument("--sara-align-eval", action="store_true",
                    help="Score the adversarial examples of the alignment term in evaluation mode, the mode in "
                         "which the targets are computed (Proposition 4); one extra forward pass on S*.")
    p.add_argument("--sara-accept-all", action="store_true",
                    help="Ablation: use every processed correction as an alignment target, skipping the "
                         "confidence and margin test. With --xi 1.0 and no budget it approximates inverse "
                         "adversarial training (Dong et al., 2023).")

    # Eval
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-apgd", action="store_true")
    p.add_argument("--eval-apgd-iters", type=int, default=100)
    p.add_argument("--eval-max-batches", type=int, default=None,
                    help="Limit per-epoch evaluation to this many batches (the final test "
                         "evaluation always uses the full test set).")
    p.add_argument("--eval-epsilons", type=float, nargs="*", default=None,
                    help="Additional perturbation radii for the final test evaluation "
                         "(the training radius is always evaluated).")
    p.add_argument("--eval-autoattack", type=int, default=0,
                    help="Evaluate the final model with the standard AutoAttack suite on the first N "
                         "test images (0 disables it).")

    # Optional subsets of the training and test sets
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--val-size", type=int, default=0,
                    help="Hold out this many training images as a validation set (fixed across seeds); "
                         "per-epoch evaluation then uses the validation set instead of the test set.")
    p.add_argument("--augmix", action="store_true",
                    help="Use AugMix data augmentation (CIFAR10 only).")

    p.add_argument("--data-loader", choices=["gpu", "torch"], default="gpu",
                    help="'gpu': dataset held on the GPU with on-GPU augmentation; 'torch': "
                         "torchvision DataLoader (required for AugMix).")
    p.add_argument("--num-workers", type=int, default=8, help="Data-loading worker processes.")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    val_loader = None
    if args.data_loader == "gpu" and not args.augmix:
        train_loader, val_loader, test_loader = get_gpu_loaders(
            args.dataset, batch_size=args.batch_size, val_size=args.val_size, train_subset=args.train_subset,
            test_subset=args.test_subset, device=device, seed=args.seed)
    else:
        if args.val_size:
            raise ValueError("--val-size requires --data-loader gpu")
        train_loader, test_loader = get_loaders(args.dataset, batch_size=args.batch_size,
                                                 num_workers=args.num_workers, train_subset=args.train_subset,
                                                 test_subset=args.test_subset, augmix=args.augmix)

    model = build_model(args.model, args.dataset).to(device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    base_eps = args.at_epsilon or DEFAULT_EPSILON[args.dataset]
    sara_eps = args.sara_epsilon or base_eps
    effective_xi = args.xi
    if args.xi_relative:
        effective_xi = scale_xi_for_num_classes(args.xi, NUM_CLASSES[args.dataset])
    max_correct = args.sara_max_correct if args.sara_max_correct else None
    cfg = SARAConfig(epsilon=sara_eps, base_epsilon=base_eps, tau=args.tau, xi=effective_xi, kappa=args.kappa,
                      alpha=args.alpha, lambda_margin=args.lambda_margin, max_batch_correct=max_correct,
                      gamma_reweight=args.gamma_reweight, sara_warmup_epochs=args.sara_warmup_epochs,
                      cg_damping=args.cg_damping, cg_confidence_cap=args.cg_confidence_cap,
                      robust_correction=args.robust_correction, revisit_memory=args.revisit_memory,
                      revisit_batch=args.revisit_batch, revisit_weight=args.revisit_weight,
                      align_weight=args.sara_align, align_natural_weight=args.sara_align_natural,
                      nat_weight=args.nat_weight, accept_all=args.sara_accept_all,
                      align_eval_mode=args.sara_align_eval)
    memory = CorrectionMemory(args.revisit_memory) if (args.sara and args.revisit_memory > 0) else None

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "num_classes": NUM_CLASSES[args.dataset], "effective_xi": effective_xi}, f, indent=2)
    base_loss = build_base_loss(args.base_loss)
    base_loss_kwargs = {"pgd_steps": args.pgd_steps} if args.base_loss != "ce" else {}

    logger = JsonlLogger(os.path.join(args.out_dir, "train_log.jsonl"))

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.at_eps_warmup > 0:
            # Linear ramp of the training radius (and of the correction radius) over the first epochs,
            # the standard remedy when adversarial training at the full radius does not start to learn.
            ramp = min(1.0, epoch / args.at_eps_warmup)
            cfg.base_epsilon, cfg.epsilon = base_eps * ramp, sara_eps * ramp
        loss_meter = AverageMeter()
        succ_meter = AverageMeter()
        chal_meter = AverageMeter()
        time_meter = AverageMeter()
        grad_base_meter = AverageMeter()
        grad_cg_meter = AverageMeter()
        viol_meter = AverageMeter()

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.method == "ohem":
                stats = ohem_training_step(model, optimizer, x, y, base_loss, cfg,
                                            top_frac=args.ohem_top_frac)
            elif args.method == "fat":
                stats = fat_training_step(model, optimizer, x, y, base_loss, cfg)
            elif args.method == "gairat":
                stats = gairat_training_step(model, optimizer, x, y, base_loss, cfg)
            else:
                stats = sara_training_step(model, optimizer, x, y, base_loss, cfg,
                                            base_loss_kwargs=base_loss_kwargs, use_sara=args.sara,
                                            epoch=epoch, diag_grad_norms=args.diag_grad_norms, memory=memory)
                if stats["n_revisited"]:
                    viol_meter.update(stats["revisit_violation_rate"], 1)
            if ema is not None:
                ema.update(model)
            loss_meter.update(stats["loss_total"], x.shape[0])
            succ_meter.update(stats["success_rate"], 1)
            chal_meter.update(stats["n_challenging"], 1)
            time_meter.update(stats["step_time_s"], 1)
            if args.diag_grad_norms:
                grad_base_meter.update(stats.get("grad_norm_base", 0.0), 1)
                grad_cg_meter.update(stats.get("grad_norm_cg", 0.0), 1)

        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "loss": loss_meter.avg,
            "mean_challenging_per_batch": chal_meter.avg,
            "mean_correction_success_rate": succ_meter.avg,
            "mean_step_time_s": time_meter.avg,
            "gpu_mem_mb": gpu_mem_mb(),
            "elapsed_s": time.time() - t_start,
        }
        if args.diag_grad_norms:
            epoch_log["grad_norm_base"] = grad_base_meter.avg
            epoch_log["grad_norm_cg"] = grad_cg_meter.avg
        if memory is not None:
            epoch_log["memory_size"] = len(memory)
            epoch_log["revisit_violation_rate"] = viol_meter.avg

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_loader = val_loader if val_loader is not None else test_loader
            eval_out = evaluate(model, eval_loader, device, base_eps, pgd_steps=20,
                                 run_apgd=args.eval_apgd, apgd_iters=args.eval_apgd_iters,
                                 max_batches=args.eval_max_batches)
            prefix = "val" if val_loader is not None else "eval"
            epoch_log.update({f"{prefix}_{k}": v for k, v in eval_out.items()})

        print(json.dumps(epoch_log))
        logger.log(**epoch_log)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))
    logger.close()

    final = {"train_time_s": time.time() - t_start}
    eval_model = model
    if ema is not None:
        # The averaged weights are the reported model; the raw weights are kept for comparison.
        torch.save(ema.module.state_dict(), os.path.join(args.out_dir, "model_ema.pt"))
        final["weights"] = "ema"
        final["raw"] = evaluate(model, test_loader, device, base_eps, pgd_steps=20, run_apgd=False)
        eval_model = ema.module
    for eps in [base_eps] + [e for e in (args.eval_epsilons or []) if e != base_eps]:
        out = evaluate(eval_model, test_loader, device, eps, pgd_steps=20, run_apgd=args.eval_apgd,
                       apgd_iters=args.eval_apgd_iters)
        final[f"test@{eps:g}"] = out
    if val_loader is not None:
        final["val"] = evaluate(eval_model, val_loader, device, base_eps, pgd_steps=20, run_apgd=args.eval_apgd,
                                apgd_iters=args.eval_apgd_iters)
    if args.eval_autoattack:
        final["autoattack"] = evaluate_autoattack(eval_model, test_loader, device, base_eps, args.eval_autoattack,
                                                  NUM_CLASSES[args.dataset])
    print(json.dumps({"final": final}))
    with open(os.path.join(args.out_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)


if __name__ == "__main__":
    main()
