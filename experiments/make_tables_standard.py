"""LaTeX rows of the standard-protocol table (Appendix D), from the run outputs.

One row per method with at least one finished seed: mean +- population standard deviation over seeds of
clean, PGD-20 and APGD-CE accuracy (full test set), AutoAttack accuracy (first 2,000 test images, fixed seed,
from robust_suite.json), calibration error and confident errors under PGD-20 (confidence.json), and GPU hours.
Usage: python experiments/make_tables_standard.py runs/std OUT_DIR
Writes OUT_DIR/r2_standard_{cifar10,svhn}_rows.tex.
"""
import glob
import json
import math
import os
import sys

RUNS, OUT = sys.argv[1], sys.argv[2]
LONG = {"cifar10": 44, "svhn": 29}
STD = {"cifar10": 30, "svhn": 20}


def methods(ds):
    return [("at_pgd", f"AT-PGD, {STD[ds]} epochs"), ("at_pgd_sara", f"AT-PGD+SARA, {STD[ds]} epochs"),
            ("at_pgd_matched", f"AT-PGD, {LONG[ds]} epochs (equal compute)"),
            (f"at_pgd_sara_ep{LONG[ds]}", f"AT-PGD+SARA, {LONG[ds]} epochs"),
            ("at_pgd_sara_acceptall", "SARA without acceptance test"),
            ("inverse_at_like", "Inverse-AT-like")]


def row(ds, run):
    vals = []
    for d in sorted(glob.glob(os.path.join(RUNS, ds, run, "seed*"))):
        need = [os.path.join(d, n) for n in ("final.json", "robust_suite.json", "confidence.json")]
        if not all(os.path.exists(p) for p in need):
            continue
        f, rs, c = (json.load(open(p)) for p in need)
        t = next(v for k, v in f.items() if k.startswith("test@"))
        a = c["pgd20"]
        vals.append([t["clean_acc"], t["pgd_acc"], t["apgd_acc"], rs["autoattack_acc"], a["ece"],
                     100.0 * sum(b["n"] * (1 - b["acc"]) for b in a["reliability"] if b["bin"] >= 12) / a["n"],
                     f["train_time_s"] / 3600])
    if not vals:
        return None
    n = len(vals)
    cols = list(zip(*vals))
    cells = []
    for j, col in enumerate(cols):
        mu = sum(col) / n
        if j == len(cols) - 1:
            cells.append(f"{mu:.2f}")
        else:
            sd = math.sqrt(sum((x - mu) ** 2 for x in col) / n)
            cells.append(f"{mu:.2f} $\\pm$ {sd:.2f}")
    return n, cells


os.makedirs(OUT, exist_ok=True)
for ds in ("cifar10", "svhn"):
    lines = []
    for run, label in methods(ds):
        r = row(ds, run)
        if r:
            lines.append(f"{label} & {r[0]} & " + " & ".join(r[1]) + " \\\\")
    with open(os.path.join(OUT, f"r2_standard_{ds}_rows.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(ds, len(lines), "rows")
