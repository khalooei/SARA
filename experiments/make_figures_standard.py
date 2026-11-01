"""Figures of the standard-protocol study (Appendix D), drawn from the run outputs in runs/std.

  r2_paired_diff    SARA minus each comparison, same seed: per-seed points, mean and 95% paired-t interval
  r2_eps_sweep      PGD-50 accuracy against the attack radius (gradient-masking check), mean over seeds
  r2_tradeoff       clean against AutoAttack accuracy, one point per run
  r2_dynamics       correction acceptance rate per epoch (SARA) and validation PGD accuracy per epoch
  r2_margins        distribution of the probability margin on clean and PGD-20 test inputs

Every figure skips what is not available yet, so it can be re-run as results arrive.
Usage: python experiments/make_figures_standard.py runs/std OUT_DIR
"""
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DS = {"cifar10": "CIFAR10", "svhn": "SVHN"}
LONG = {"cifar10": 44, "svhn": 29}
STD = {"cifar10": 30, "svhn": 20}
# Okabe-Ito colours, one per method, used in every figure.
STYLE = {"at_pgd": ("AT-PGD", "#0072B2"), "at_pgd_sara": ("AT-PGD+SARA", "#D55E00"),
         "at_pgd_matched": ("AT-PGD, longer", "#56B4E9"), "at_pgd_sara_ep": ("AT-PGD+SARA, longer", "#E69F00"),
         "at_pgd_sara_acceptall": ("SARA without acceptance test", "#009E73"),
         "inverse_at_like": ("Inverse-AT-like", "#CC79A7")}
T975 = {1: 12.706, 2: 4.303, 3: 3.182}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})


def run_name(ds, key):
    return f"at_pgd_sara_ep{LONG[ds]}" if key == "at_pgd_sara_ep" else key


def load(ds, key):
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, ds, run_name(ds, key), "seed*"))):
        s = int(d.rsplit("seed", 1)[1])
        r = {}
        if os.path.exists(os.path.join(d, "final.json")):
            f = json.load(open(os.path.join(d, "final.json")))
            t = next(v for k, v in f.items() if k.startswith("test@"))
            r.update(clean=t["clean_acc"], pgd=t["pgd_acc"], apgd=t.get("apgd_acc"),
                     aa=f.get("autoattack", {}).get("autoattack_acc"))
        for name in ("robust_suite", "confidence"):
            p = os.path.join(d, name + ".json")
            if os.path.exists(p):
                r[name] = json.load(open(p))
        if "robust_suite" in r:
            r["aa"] = r["robust_suite"]["autoattack_acc"]  # 2,000 images, fixed seed, replaces the 1,000-image value
            r["wb"] = r["robust_suite"]["whitebox_acc"]
        p = os.path.join(d, "train_log.jsonl")
        if os.path.exists(p):
            r["log"] = [json.loads(line) for line in open(p) if line.strip()]
        if r:
            out[s] = r
    return out


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def paired_diff():
    comps = [("at_pgd_sara", "at_pgd", "SARA vs AT-PGD, same epochs"),
             ("at_pgd_sara", "at_pgd_matched", "SARA vs AT-PGD trained 1.46x longer"),
             ("at_pgd_sara_ep", "at_pgd_matched", "SARA vs AT-PGD, both longer"),
             ("at_pgd_sara", "at_pgd_sara_acceptall", "SARA vs no acceptance test"),
             ("at_pgd_sara", "inverse_at_like", "SARA vs inverse-AT-like")]
    metrics = [("pgd", "PGD-20"), ("apgd", "APGD-CE"), ("aa", "AutoAttack")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
    for ax, ds in zip(axes, DS):
        rows, labels = [], []
        for a, b, lab in comps:
            A, B = load(ds, a), load(ds, b)
            seeds = sorted(s for s in set(A) & set(B) if "pgd" in A[s] and "pgd" in B[s])
            if not seeds:
                continue
            for m, mlab in metrics:
                d = [A[s][m] - B[s][m] for s in seeds if A[s].get(m) is not None and B[s].get(m) is not None]
                if d:
                    rows.append(d)
                    labels.append(f"{lab}: {mlab}")
        y = list(range(len(rows)))[::-1]
        for yi, d in zip(y, rows):
            mu = sum(d) / len(d)
            if len(d) > 1:
                sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
                h = T975[len(d) - 1] * sd / math.sqrt(len(d))
                ax.plot([mu - h, mu + h], [yi, yi], color="#555555", lw=1.2, solid_capstyle="round")
            ax.scatter(d, [yi] * len(d), s=9, color="#999999", zorder=3, linewidths=0)
            ax.scatter([mu], [yi], s=22, color="#D55E00" if mu > 0 else "#0072B2", zorder=4, marker="D")
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ds == "cifar10" else [""] * len(labels))
        ax.set_title(DS[ds])
        ax.set_xlabel("difference in accuracy (points)")
        ax.grid(axis="y", visible=False)
    axes[1].set_yticks(axes[1].get_yticks())
    fig.suptitle("Paired differences, same seed: points = seeds, diamond = mean, bar = 95% paired-t interval",
                 fontsize=7.5, y=1.02)
    save(fig, "r2_paired_diff")


def eps_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, ds in zip(axes, DS):
        for key, (lab, col) in STYLE.items():
            runs = [r["robust_suite"]["eps_sweep"]["results"] for r in load(ds, key).values() if "robust_suite" in r]
            if not runs:
                continue
            eps = [p["eps_255"] for p in runs[0]]
            acc = [sum(r[i]["pgd50_acc"] for r in runs) / len(runs) for i in range(len(eps))]
            seeds = f"{len(runs)} seed{'s' if len(runs) > 1 else ''}"
            ax.plot(eps, acc, marker="o", ms=3, lw=1.5, color=col, label=f"{lab} ({seeds})")
        ax.axvline(8, color="#999999", lw=0.8, ls="--")
        ax.set_title(DS[ds])
        ax.set_xlabel("attack radius (x/255), PGD-50")
        ax.set_ylabel("robust accuracy (%)")
        ax.set_ylim(0, None)
    axes[0].legend(frameon=False)
    save(fig, "r2_eps_sweep")


def tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, ds in zip(axes, DS):
        for key, (lab, col) in STYLE.items():
            pts = [(r["clean"], r["aa"]) for r in load(ds, key).values() if r.get("aa") is not None]
            if pts:
                ax.scatter(*zip(*pts), s=18, color=col, label=lab, alpha=0.9, linewidths=0)
        ax.set_title(DS[ds])
        ax.set_xlabel("clean accuracy (%)")
        ax.set_ylabel("AutoAttack accuracy (%)")
    axes[0].legend(frameon=False)
    save(fig, "r2_tradeoff")


def dynamics():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ds, ls in (("cifar10", "-"), ("svhn", "--")):
        for s, r in load(ds, "at_pgd_sara").items():
            if "log" in r:
                axes[0].plot([e["epoch"] for e in r["log"]],
                             [100 * e["mean_correction_success_rate"] for e in r["log"]],
                             ls=ls, lw=1.2, color="#D55E00", alpha=0.5 + 0.2 * s,
                             label=f"{DS[ds]}" if s == 0 else None)
    axes[0].set_title("SARA: accepted corrections")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("% of challenging samples")
    axes[0].legend(frameon=False)
    for ax, ds in zip(axes[1:], DS):
        for key in ("at_pgd", "at_pgd_matched", "at_pgd_sara", "at_pgd_sara_ep"):
            lab, col = STYLE[key]
            curves = [[(e["epoch"], e["val_pgd_acc"]) for e in r["log"] if "val_pgd_acc" in e]
                      for r in load(ds, key).values() if "log" in r]
            curves = [c for c in curves if c]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            ax.plot([c[0] for c in curves[0][:n]], [sum(c[i][1] for c in curves) / len(curves) for i in range(n)],
                    marker="o", ms=2.5, lw=1.3, color=col, label=lab)
        ax.set_title(f"{DS[ds]}: validation PGD-20")
        ax.set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[2].legend(frameon=False)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "r2_dynamics")


def margins():
    have = any("confidence" in r for ds in DS for r in load(ds, "at_pgd_sara").values())
    if not have:
        print("skip r2_margins (no confidence.json yet)")
        return
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=False)
    for j, ds in enumerate(DS):
        for k, split in enumerate(("clean", "pgd20")):
            ax = axes[2 * j + k]
            for key in ("at_pgd", "at_pgd_sara"):
                lab, col = STYLE[key]
                hs = [r["confidence"][split]["margin_hist"] for r in load(ds, key).values() if "confidence" in r]
                if not hs:
                    continue
                counts = [sum(h["counts"][i] for h in hs) for i in range(len(hs[0]["counts"]))]
                tot = sum(counts)
                w = 2 / len(counts)
                xs = [-1 + w * (i + 0.5) for i in range(len(counts))]
                ax.step(xs, [c / tot for c in counts], where="mid", color=col, lw=1.2, label=lab)
            ax.axvline(0, color="#999999", lw=0.7)
            ax.set_title(f"{DS[ds]}, {'clean' if split == 'clean' else 'PGD-20'}")
            ax.set_xlabel("probability margin")
    axes[0].legend(frameon=False)
    save(fig, "r2_margins")


def calibration():
    """Reliability under PGD-20 (bins pooled over seeds) and two summary measures per method."""
    keys = ("at_pgd", "at_pgd_sara", "at_pgd_matched")
    if not any("confidence" in r for r in load("cifar10", "at_pgd_sara").values()):
        print("skip r2_calibration (no confidence.json yet)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, ds in zip(axes[:2], DS):
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
        for key in keys:
            lab, col = STYLE[key]
            runs = [r["confidence"]["pgd20"]["reliability"] for r in load(ds, key).values() if "confidence" in r]
            if not runs:
                continue
            pts = []
            for i in range(15):
                bins = [b for rel in runs for b in rel if b["bin"] == i]
                n = sum(b["n"] for b in bins)
                if n >= 20:
                    pts.append((sum(b["conf"] * b["n"] for b in bins) / n, sum(b["acc"] * b["n"] for b in bins) / n))
            ax.plot(*zip(*pts), marker="o", ms=2.5, lw=1.3, color=col, label=lab)
        ax.set_title(f"{DS[ds]}, PGD-20 inputs")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, loc="upper left")
    ax = axes[2]
    width = 0.26
    labels = []
    for j, (ds, metric) in enumerate([(d, m) for d in DS for m in ("ece", "conf_err")]):
        labels.append(f"{DS[ds]}\n{'ECE' if metric == 'ece' else 'conf. err.'}")
        for k, key in enumerate(keys):
            vals = []
            for r in load(ds, key).values():
                if "confidence" not in r:
                    continue
                c = r["confidence"]["pgd20"]
                vals.append(c["ece"] if metric == "ece" else
                            100.0 * sum(b["n"] * (1 - b["acc"]) for b in c["reliability"] if b["bin"] >= 12) / c["n"])
            if vals:
                mu = sum(vals) / len(vals)
                sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1))
                ax.bar(j + (k - 1) * width, mu, width, yerr=sd, color=STYLE[key][1], capsize=1.5,
                       error_kw={"lw": 0.7}, label=STYLE[key][0] if j == 0 else None)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("% (lower is better)")
    ax.set_title("PGD-20: ECE and confident errors")
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "r2_calibration")


def corruption():
    keys = ("at_pgd", "at_pgd_sara", "at_pgd_matched")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
    any_data = False
    for ax, ds in zip(axes, DS):
        names = ["gaussian_noise", "gaussian_blur", "occlusion", "low_light"]
        width = 0.26
        for k, key in enumerate(keys):
            rows = []
            for d in glob.glob(os.path.join(RUNS, ds, key, "seed*", "corruption.jsonl")):
                lines = [json.loads(line) for line in open(d) if line.strip()]
                if lines:
                    rows.append(lines[-1])
            if not rows:
                continue
            any_data = True
            means = [sum(sum(r[f"{n}_sev{v}"] for v in (1, 3, 5)) / 3 for r in rows) / len(rows) for n in names]
            ax.bar([i + (k - 1) * width for i in range(len(names))], means, width, color=STYLE[key][1],
                   label=f"{STYLE[key][0]} ({len(rows)} seeds)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", " ") for n in names])
        ax.set_title(f"{DS[ds]}: accuracy under corruption (mean of severities 1, 3, 5)")
        ax.set_ylabel("accuracy (%)")
        ax.grid(axis="x", visible=False)
    if not any_data:
        plt.close(fig)
        print("skip r2_corruption (no corruption.jsonl yet)")
        return
    axes[0].legend(frameon=False, fontsize=6.5)
    save(fig, "r2_corruption")


def ablation():
    """Acceptance-test ablation: clean and AutoAttack accuracy per seed for SARA and the two variants."""
    keys = [("at_pgd", "AT-PGD"), ("at_pgd_sara", "AT-PGD+SARA"),
            ("at_pgd_sara_acceptall", "without\nacceptance test"), ("inverse_at_like", "inverse-AT-like")]
    dsets = [ds for ds in DS if load(ds, "at_pgd_sara_acceptall")]
    if not dsets:
        print("skip r2_ablation (no ablation runs yet)")
        return
    fig, axes = plt.subplots(len(dsets), 2, figsize=(7.2, 2.3 * len(dsets)), squeeze=False)
    for i, ds in enumerate(dsets):
        for j, (metric, title) in enumerate((("clean", "clean accuracy"), ("aa", "AutoAttack accuracy"))):
            ax = axes[i][j]
            names = []
            for key, lab in keys:
                vals = [r[metric] for r in load(ds, key).values() if r.get(metric) is not None]
                if not vals:
                    continue
                names.append(lab)
                x = len(names) - 1
                col = STYLE.get(key, ("", "#777777"))[1]
                mu = sum(vals) / len(vals)
                ax.plot([x - 0.28, x + 0.28], [mu, mu], color=col, lw=2.4, solid_capstyle="round", zorder=2)
                ax.scatter([x + d for d in (-0.08, 0.0, 0.08)[:len(vals)]], vals, s=16, color=col, zorder=3,
                           edgecolors="white", linewidths=0.5)
                ax.annotate(f"{mu:.1f}", (x + 0.32, mu), va="center", fontsize=6.8, color="#333333")
            lo = min(min(r[metric] for r in load(ds, key).values() if r.get(metric) is not None)
                     for key, _ in keys if any(r.get(metric) is not None for r in load(ds, key).values()))
            hi = max(max(r[metric] for r in load(ds, key).values() if r.get(metric) is not None)
                     for key, _ in keys if any(r.get(metric) is not None for r in load(ds, key).values()))
            ax.set_ylim(lo - 1.0, hi + 1.0)
            ax.set_xlim(-0.5, len(keys) - 0.3)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=6.8)
            ax.set_title(f"{DS[ds]}: {title}")
            ax.set_ylabel("%")
            ax.grid(axis="x", visible=False)
    fig.subplots_adjust(hspace=0.6, wspace=0.28)
    save(fig, "r2_ablation")


if __name__ == "__main__":
    for f in (paired_diff, eps_sweep, tradeoff, dynamics, margins, calibration, corruption, ablation):
        try:
            f()
        except Exception as exc:  # one missing input must not stop the other figures
            print(f"skip {f.__name__}: {type(exc).__name__}: {exc}")
