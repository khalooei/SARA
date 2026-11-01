"""Measure the alignment gap eta of Proposition 4 on a trained model.

For challenging training samples, the corrections accepted by the inner search are paired with
PGD adversarial examples of the same natural samples, and

    eta = KL(p(x*) || p(x_adv)).

The script reports the distribution of eta, the fraction of pairs with eta < tau^2 / 2 (where
Proposition 4 guarantees that x_adv is classified correctly), and checks the two inequalities of
the proposition on every pair.

Usage:
    python scripts/measure_alignment_gap.py --run-dir runs/r3/cifar10/at_pgd_sara/seed0 --n 5000
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sara.attacks import pgd_attack  # noqa: E402
from sara.confidence_guard import SARAConfig, correct_challenging_samples, identify_challenging_and_confidence, margin  # noqa: E402
from sara.data import get_gpu_loaders  # noqa: E402
from sara.models import build_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n", type=int, default=5000, help="number of training images to scan")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()

    cfg_run = json.load(open(os.path.join(args.run_dir, "config.json")))
    device = torch.device("cuda")
    model = build_model(cfg_run["model"], cfg_run["dataset"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run_dir, "model_final.pt"), map_location=device))
    model.eval()

    eps = cfg_run["at_epsilon"]
    xi = cfg_run["effective_xi"]
    tau = cfg_run["tau"]
    cfg = SARAConfig(epsilon=cfg_run["sara_epsilon"] or eps, tau=tau, xi=xi, kappa=cfg_run["kappa"],
                     alpha=cfg_run["alpha"], lambda_margin=cfg_run["lambda_margin"])
    train_loader, _, _ = get_gpu_loaders(cfg_run["dataset"], batch_size=args.batch_size, augment=False,
                                         val_size=cfg_run.get("val_size", 0), train_subset=args.n)

    etas, conf_ok, margin_ok, adv_correct = [], [], [], []
    n_chal = n_seen = 0
    for x, y in train_loader:
        n_seen += x.shape[0]
        mask, _ = identify_challenging_and_confidence(model, x, y, xi)
        model.eval()
        x_s, y_s = x[mask], y[mask]
        n_chal += x_s.shape[0]
        x_star, y_star, _, idx = correct_challenging_samples(model, x_s, y_s, cfg, return_index=True)
        if x_star.shape[0] == 0:
            continue
        x_nat = x_s[idx]
        x_adv = pgd_attack(model, x_nat, y_star, eps, eps / 4, cfg_run["pgd_steps"])
        with torch.no_grad():
            p_star = F.softmax(model(x_star).float(), dim=1)
            p_adv = F.softmax(model(x_adv).float(), dim=1)
        eta = (p_star * (p_star.clamp_min(1e-12).log() - p_adv.clamp_min(1e-12).log())).sum(1)
        q_y = p_adv.gather(1, y_star.view(-1, 1)).squeeze(1)
        h_adv = margin(p_adv, y_star)
        etas.append(eta)
        conf_ok.append(q_y >= xi - (eta / 2).sqrt() - 1e-6)
        margin_ok.append(h_adv <= -tau + (2 * eta).sqrt() + 1e-6)
        adv_correct.append(p_adv.argmax(1) == y_star)

    if not etas:
        print(json.dumps({"run_dir": os.path.relpath(args.run_dir), "n_scanned": n_seen, "n_challenging": n_chal,
                          "n_accepted": 0}))
        return
    eta = torch.cat(etas)
    q = torch.quantile(eta, torch.tensor([0.1, 0.5, 0.9], device=eta.device)).tolist()
    out = {
        "run_dir": os.path.relpath(args.run_dir), "n_scanned": n_seen, "n_challenging": n_chal,
        "n_accepted": int(eta.numel()),
        "tau": tau, "threshold_tau2_over_2": tau ** 2 / 2,
        "eta_p10": q[0], "eta_median": q[1], "eta_p90": q[2], "eta_mean": float(eta.mean()),
        "frac_eta_below_threshold": float((eta < tau ** 2 / 2).float().mean()),
        "frac_confidence_bound_holds": float(torch.cat(conf_ok).float().mean()),
        "frac_margin_bound_holds": float(torch.cat(margin_ok).float().mean()),
        "frac_adv_correct": float(torch.cat(adv_correct).float().mean()),
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(args.run_dir, "alignment_gap.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
