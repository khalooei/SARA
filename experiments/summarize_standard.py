"""Summarize the standard-protocol study (experiments/standard_protocol.sh).

Writes summary.md (mean +- population std over seeds, paired differences, alignment gap) and one
LaTeX row file per dataset, table_rows_<dataset>.tex.

Usage: python experiments/summarize_standard.py RUNS_DIR
"""
import glob
import json
import os
import statistics as st
import sys

RUNS = sys.argv[1]
METRICS = ["clean", "pgd", "apgd", "aa"]


def label(run, default_epochs):
    if run == "at_pgd":
        return f"AT-PGD, {default_epochs} epochs"
    if run == "at_pgd_sara":
        return f"AT-PGD+SARA, {default_epochs} epochs"
    if run == "at_pgd_matched":
        return "AT-PGD, compute-matched epochs"
    if run.startswith("pilot_"):
        return "pilot " + run[len("pilot_"):] + " (10 epochs)"
    if run.startswith("at_pgd_ep"):
        return f"AT-PGD, {run[len('at_pgd_ep'):]} epochs (compute-matched)"
    return {"at_pgd_sara_evalalign": "AT-PGD+SARA, evaluation-mode alignment",
            "at_pgd_sara_acceptall": "SARA without acceptance test",
            "inverse_at_like": "Inverse-AT-like (all samples, no test)"}.get(run, run)


def order(run):
    return {"at_pgd": 0, "at_pgd_sara": 1, "at_pgd_sara_acceptall": 3, "inverse_at_like": 4}.get(run, 2)


def load(ds, run):
    out = {}
    for f in sorted(glob.glob(os.path.join(RUNS, ds, run, "seed*", "final.json"))):
        seed = int(f.split("seed")[-1].split("/")[0])
        j = json.load(open(f))
        t = [v for k, v in j.items() if k.startswith("test@")][0]
        out[seed] = {"clean": t["clean_acc"], "pgd": t["pgd_acc"], "apgd": t.get("apgd_acc"),
                     "aa": j.get("autoattack", {}).get("autoattack_acc"), "time": j["train_time_s"]}
    return out


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{st.mean(vals):.2f} ± {st.pstdev(vals):.2f}"


lines = []
for ds in sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d))):
    epochs = {"cifar10": 30, "svhn": 20}.get(ds, "?")
    runs = sorted((r for r in os.listdir(os.path.join(RUNS, ds)) if load(ds, r)), key=order)
    data = {r: load(ds, r) for r in runs}
    lines += [f"## {ds}", "", "| Method | Seeds | Clean | PGD-20 | APGD-CE | AutoAttack (1k) | Train time (s) |",
              "|---|---|---|---|---|---|---|"]
    tex = []
    for r in runs:
        d = data[r]
        row = [fmt([d[s][m] for s in d]) for m in METRICS]
        lines.append(f"| {label(r, epochs)} | {len(d)} | " + " | ".join(row) +
                     f" | {st.mean(d[s]['time'] for s in d):.0f} |")
        if not r.startswith("pilot_"):
            cells = " & ".join(c.replace("±", "$\\pm$") for c in row)
            tex.append(f"{label(r, epochs)} & {len(d)} & {cells} \\\\")
    open(os.path.join(RUNS, f"table_rows_{ds}.tex"), "w").write("\n".join(tex) + "\n")

    lines += ["", "Paired differences (same seed, AT-PGD+SARA minus the other run):", ""]
    for b in runs:
        if b == "at_pgd_sara" or "at_pgd_sara" not in data:
            continue
        seeds = sorted(set(data["at_pgd_sara"]) & set(data[b]))
        if not seeds:
            continue
        diffs = {m: [data["at_pgd_sara"][s][m] - data[b][s][m] for s in seeds
                     if data["at_pgd_sara"][s][m] is not None and data[b][s][m] is not None] for m in METRICS}
        lines.append(f"- vs {label(b, epochs)} (seeds {seeds}): " + ", ".join(
            f"{m} {st.mean(v):+.2f}" for m, v in diffs.items() if v))

    for f in sorted(glob.glob(os.path.join(RUNS, ds, "at_pgd_sara", "seed*", "alignment_gap.json"))):
        e = json.load(open(f))
        if not e.get("n_accepted"):
            continue
        lines.append(f"- eta ({f.split('/')[-2]}): median {e['eta_median']:.4f}, p10 {e['eta_p10']:.4f}, "
                     f"p90 {e['eta_p90']:.4f}; below tau^2/2 = {e['threshold_tau2_over_2']:.5f}: "
                     f"{100 * e['frac_eta_below_threshold']:.1f}%; Prop. 4 bounds hold: "
                     f"{100 * e['frac_confidence_bound_holds']:.1f}% / {100 * e['frac_margin_bound_holds']:.1f}%; "
                     f"adversarial example correct: {100 * e['frac_adv_correct']:.1f}% (n={e['n_accepted']})")
    lines.append("")

open(os.path.join(RUNS, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
