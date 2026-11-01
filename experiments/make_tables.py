"""Write the LaTeX tables of the second-revision experiments from the run outputs.

    python experiments/make_tables.py OUT_DIR

Every number is read from the final.json files below runs/main, runs/tune and runs/pilot; nothing is typed by
hand. Means and population standard deviations are taken over the seed*/ subdirectories of each configuration.
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = [("mnist", "MNIST", "0.2"), ("cifar10", "CIFAR10", "0.03"), ("svhn", "SVHN", "0.03")]
METRICS = ["clean", "pgd", "apgd", "aa"]


def load(path):
    d = json.loads(Path(path).read_text())
    t = next(v for k, v in d.items() if k.startswith("test@"))
    return {"clean": t["clean_acc"], "pgd": t["pgd_acc"], "apgd": t.get("apgd_acc", np.nan),
            "aa": d.get("autoattack", {}).get("autoattack_acc", np.nan), "val": d["val"]["pgd_acc"]}


# The tuned CIFAR10 batch has 64 images. A correction budget of 64 rewrites the whole batch.
# Budget 32 is half the batch and is the value selected on the validation split.
OVERRIDES = {("cifar10", "tuned_at_pgd_sara"): ("runs/pilot", "tuned_b32")}


def seeds(ds, name, root="runs/main"):
    root, name = OVERRIDES.get((ds, name), (root, name))
    runs = [load(f) for f in sorted(Path(root, ds, name).glob("seed*/final.json"))]
    return runs if len(runs) >= 2 else None


def stat(runs, m):
    a = np.array([r[m] for r in runs])
    return a.mean(), a.std()


def cell(mean, std, bold=False):
    s = f"{mean:.2f}" + (rf"{{\scriptsize$\pm${std:.2f}}}" if std is not None else "")
    return rf"\textbf{{{s}}}" if bold else s


# Table: multi-seed controlled comparison -------------------------------------------------------------------
PAIRS = [("standard", "Standard", None), ("fat", "FAT", "fat_sara"), ("gairat", "GAIRAT", None),
         ("at_pgd", "AT-PGD", "at_pgd_sara"), ("at_trades", "AT-TRADES", "at_trades_sara"),
         ("at_mart", "AT-MART", "at_mart_sara"),
         ("at_pgd_nat", "AT-PGD + natural CE", "at_pgd_sara_nat")]
lines = []
for ds, title, eps in DATASETS:
    lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}} ($\varepsilon={eps}$)}} \\")
    for base, label, sara in PAIRS:
        rb = seeds(ds, base)
        rs = seeds(ds, sara) if sara else None
        if rb is None:
            continue
        rows = [(label, rb)] + ([(r"\ +SARA", rs)] if rs else [])
        for lab, runs in rows:
            cells = []
            for m in METRICS:
                mean, std = stat(runs, m)
                bold = False
                if rs is not None:
                    other = rs if runs is rb else rb
                    bold = mean > stat(other, m)[0]
                cells.append(cell(mean, std, bold))
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
(OUT / "r2_main_rows.tex").write_text("\n".join(lines) + "\n")

# Table: tuning versus SARA (2 x 2 design) and longer schedule ---------------------------------------------
lines = []
for ds, title, eps in DATASETS:
    tuned_name = "tuned_at_pgd"
    blocks = [("Shared configuration (Table~\\ref{tab:repro})", "at_pgd", "at_pgd_sara"),
              ("Tuned AT-PGD configuration (best of 11 trials)", tuned_name, tuned_name + "_sara"),
              ("Shared configuration, twice the epochs", "at_pgd_long", "at_pgd_sara_long")]
    lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}} ($\varepsilon={eps}$)}} \\")
    for blabel, base, sara in blocks:
        rb, rs = seeds(ds, base), seeds(ds, sara)
        if rb is None:
            continue
        base_stat = {m: stat(rb, m) for m in METRICS}
        sara_stat = {m: stat(rs, m) for m in METRICS} if rs is not None else None

        def paired(stats, other):
            out = []
            for m in METRICS:
                mean, std = stats[m]
                bold = other is not None and mean > other[m][0]
                out.append(cell(mean, std, bold))
            return out

        lines.append(rf"{blabel} & AT-PGD & " + " & ".join(paired(base_stat, sara_stat)) + r" \\")
        if sara_stat is not None:
            lines.append(r" & AT-PGD+SARA & " + " & ".join(paired(sara_stat, base_stat)) + r" \\")
            d = [sara_stat[m][0] - base_stat[m][0] for m in METRICS]
            lines.append(r" & $\Delta$ (SARA) & " + " & ".join(f"{v:+.2f}" for v in d) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
(OUT / "r2_tuning_rows.tex").write_text("\n".join(lines) + "\n")

# Table: equal-budget selection on the validation split ----------------------------------------------------
SARA_TRIALS = ["at_pgd_sara", "at_pgd_sara_cap", "at_pgd_sara_rev", "at_pgd_sara_robust", "at_pgd_sara_robust_cap",
               "at_pgd_sara_robust_rev", "at_pgd_sara_align1", "at_pgd_sara_align3", "at_pgd_sara_align1_rev",
               "at_pgd_sara_align1_xi03", "at_pgd_sara_align3_xi03"]
summary = {}
lines = []
for ds, title, _eps in DATASETS:
    at = [("default", load(f"runs/main/{ds}/at_pgd/seed0/final.json"))]
    at += [(p.parent.parent.name, load(p)) for p in sorted(Path("runs/tune", ds).glob("*/seed0/final.json"))]
    sa = [(n, load(f"runs/pilot/{ds}/{n}/final.json")) for n in SARA_TRIALS
          if Path(f"runs/pilot/{ds}/{n}/final.json").exists()]
    best_at = max(at, key=lambda t: t[1]["val"])
    best_sa = max(sa, key=lambda t: t[1]["val"])
    summary[ds] = {"at_trials": len(at), "sara_trials": len(sa), "best_at": best_at[0], "best_sara": best_sa[0]}
    for lab, (_name, r), n in [("AT-PGD", best_at, len(at)), ("AT-PGD+SARA", best_sa, len(sa))]:
        lines.append(rf"{title if lab == 'AT-PGD' else ''} & {lab} & {n} & {r['val']:.2f} & "
                     + " & ".join(f"{r[m]:.2f}" for m in METRICS) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
(OUT / "r2_budget_rows.tex").write_text("\n".join(lines) + "\n")
(OUT / "r2_budget_selection.json").write_text(json.dumps(summary, indent=2))

# Table: ablation of the second-revision components (seed 0, validation and test) --------------------------
ABL = [("at_pgd", "AT-PGD (no SARA)"), ("at_pgd_sara", "SARA, Eq.~(9), correction budget 64"),
       ("at_pgd_sara_cap", "\\ + confidence cap"), ("at_pgd_sara_rev", "\\ + cap + revisiting memory"),
       ("at_pgd_sara_robust", "\\ + robust correction"), ("at_pgd_sara_robust_cap", "\\ + robust correction + cap"),
       ("at_pgd_sara_align1", "\\ + cap + alignment, $\\beta=1$"),
       ("at_pgd_sara_align3", "\\ + cap + alignment, $\\beta=3$"),
       ("at_pgd_sara_align1_rev", "\\ + cap + alignment, $\\beta=1$, + memory"),
       ("at_pgd_sara_align3_xi03", "\\ + cap + alignment, $\\beta=3$, $\\xi=0.3$ (selected)")]
lines = []
for name, label in ABL:
    cells = []
    for ds, _, _ in DATASETS:
        f = Path(f"runs/pilot/{ds}/{name}/final.json")
        if name == "at_pgd" and not f.exists():
            f = Path(f"runs/main/{ds}/at_pgd/seed0/final.json")
        if f.exists():
            r = load(f)
            cells += [f"{r['val']:.2f}", f"{r['pgd']:.2f}", f"{r['apgd']:.2f}"]
        else:
            cells += ["--"] * 3
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
(OUT / "r2_ablation_rows.tex").write_text("\n".join(lines) + "\n")
print("tables written to", OUT)
print(json.dumps(summary, indent=2))
