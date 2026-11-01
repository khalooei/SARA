"""Paired statistics for the standard-protocol study: SARA against each comparison run, same seed.

For every metric: the per-seed differences, their mean with a 95% paired-t interval, and the number of
seeds with a positive difference. For AutoAttack, when both runs have robust_suite.json, an exact
McNemar test on the images both runs were evaluated on (pooled over seeds and per seed), which uses the
image-level pairing and is far more sensitive than a test over three seeds. A two-sided Wilcoxon test
over three seeds cannot reach p < 0.25, so it is not reported.

Usage: python experiments/paired_stats.py runs/std > results/revision2/standard/paired_stats.md
"""
import glob
import json
import math
import os
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs/std"
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # two-sided 95% t quantiles by degrees of freedom
PAIRS = {"cifar10": [("at_pgd_sara", "at_pgd"), ("at_pgd_sara", "at_pgd_matched"),
                     ("at_pgd_sara_ep44", "at_pgd_matched"), ("at_pgd_sara", "at_pgd_sara_acceptall"),
                     ("at_pgd_sara", "inverse_at_like")],
         "svhn": [("at_pgd_sara", "at_pgd"), ("at_pgd_sara", "at_pgd_matched"),
                  ("at_pgd_sara_ep29", "at_pgd_matched"), ("at_pgd_sara", "at_pgd_sara_acceptall")]}


def load(ds, run):
    out = {}
    for f in glob.glob(os.path.join(RUNS, ds, run, "seed*", "final.json")):
        seed = int(f.split("seed")[-1].split("/")[0])
        d = json.load(open(f))
        t = next(v for k, v in d.items() if k.startswith("test@"))
        r = {"clean": t["clean_acc"], "pgd": t["pgd_acc"], "apgd": t.get("apgd_acc"),
             "aa1k": d.get("autoattack", {}).get("autoattack_acc")}
        rs = os.path.join(os.path.dirname(f), "robust_suite.json")
        if os.path.exists(rs):
            s = json.load(open(rs))
            r["aa2k"] = s["autoattack_acc"]
            r["whitebox5k"] = s["whitebox_acc"]
            r["aa_bits"] = s["per_image"]["autoattack"]
            r["wb_bits"] = s["per_image"]["whitebox"]
            r["square_drop"] = s["square_extra_drop"]
        cf = os.path.join(os.path.dirname(f), "confidence.json")
        if os.path.exists(cf):
            c = json.load(open(cf))
            for split, tag in (("clean", "clean"), ("pgd20", "adv")):
                r[f"ece_{tag}"] = c[split]["ece"]
                r[f"nll_{tag}"] = c[split]["nll"]
                # confident mistakes: wrong predictions made with confidence above 0.8, % of all inputs
                r[f"conf_err_{tag}"] = 100.0 * sum(b["n"] * (1 - b["acc"]) for b in c[split]["reliability"]
                                                   if b["bin"] >= 12) / c[split]["n"]
            r["l2"] = c["l2_apgd"]["acc"]
        rl = os.path.join(os.path.dirname(f), "reliability.json")
        if os.path.exists(rl):
            c = json.load(open(rl))
            r["ece_clean_ts"] = c["clean"]["ece_scaled"]
            r["ece_adv_ts"] = c["pgd20"]["ece_scaled"]
            r["aurc_adv"] = c["pgd20"]["aurc"]
            r["sel80_adv"] = c["pgd20"]["acc_at_80"]
        co = os.path.join(os.path.dirname(f), "corruption.jsonl")
        if os.path.exists(co):
            rows = [json.loads(line) for line in open(co) if line.strip()]
            if rows:
                v = [x for k, x in rows[-1].items() if "_sev" in k]
                r["corruption_mean"] = sum(v) / len(v)
        out[seed] = r
    return out


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value: b = only run A robust, c = only run B robust."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    print("# Paired statistics, standard protocol\n")
    print("Differences are the first run minus the second, same seed (test set). Accuracy metrics: higher is better. "
          "ece_* (expected calibration error, %), nll_* and conf_err_* "
          "(confident mistakes: wrong with confidence > 0.8, "
          "% of inputs): lower is better. adv = PGD-20 inputs; l2 = APGD-CE, L2, eps 0.5; corruption_mean = mean "
          "accuracy over 4 corruptions x 3 severities. *_ts: after temperature scaling fitted on the clean validation "
          "split; aurc_adv: area under the risk-coverage curve on PGD-20 inputs (lower is better); sel80_adv: "
          "accuracy on the 80% most confident PGD-20 inputs.\n")
    for ds, pairs in PAIRS.items():
        print(f"## {ds}\n")
        for a, b in pairs:
            A, B = load(ds, a), load(ds, b)
            seeds = sorted(set(A) & set(B))
            if not seeds:
                continue
            print(f"### {a} vs {b} (seeds {seeds})\n")
            print("| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |")
            print("|---|---|---|---|---|")
            for m in ["clean", "pgd", "apgd", "aa1k", "aa2k", "whitebox5k", "l2", "corruption_mean",
                      "ece_clean", "ece_adv", "nll_adv", "conf_err_clean", "conf_err_adv",
                      "ece_clean_ts", "ece_adv_ts", "aurc_adv", "sel80_adv"]:
                d = [A[s][m] - B[s][m] for s in seeds if A[s].get(m) is not None and B[s].get(m) is not None]
                if not d:
                    continue
                mu = sum(d) / len(d)
                if len(d) > 1:
                    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
                    h = T975[len(d) - 1] * sd / math.sqrt(len(d))
                    ci = f"[{mu - h:+.2f}, {mu + h:+.2f}]"
                else:
                    ci = "n/a (1 seed)"
                diffs = ", ".join(f"{x:+.2f}" for x in d)
                print(f"| {m} | {diffs} | {mu:+.2f} | {ci} | {sum(x > 0 for x in d)}/{len(d)} |")
            for key, label in (("aa_bits", "AutoAttack (all four stages, first 2,000 images)"),
                               ("wb_bits", "AutoAttack white-box stages (first 5,000 images)")):
                bb = cc = 0
                per = []
                for s in seeds:
                    if key in A[s] and key in B[s]:
                        n = min(len(A[s][key]), len(B[s][key]))
                        b1 = sum(1 for i in range(n) if A[s][key][i] == "1" and B[s][key][i] == "0")
                        c1 = sum(1 for i in range(n) if A[s][key][i] == "0" and B[s][key][i] == "1")
                        bb, cc = bb + b1, cc + c1
                        per.append(f"seed {s}: n={n}, only {a} robust {b1}, only {b} robust {c1}, "
                                   f"p={mcnemar_p(b1, c1):.3g}")
                if per:
                    print(f"\n{label}, McNemar on the same images: pooled only-{a} {bb}, only-{b} {cc}, "
                          f"exact p = {mcnemar_p(bb, cc):.3g}")
                    for line in per:
                        print(f"- {line}")
            drops = [f"{r}: " + ", ".join(f"{X[s]['square_drop']:.2f}" for s in seeds if "square_drop" in X[s])
                     for r, X in ((a, A), (b, B)) if any("square_drop" in X[s] for s in seeds)]
            if drops:
                print("\nAccuracy removed by Square after the white-box attacks had failed, points "
                      "(gradient-masking check): "
                      + "; ".join(drops))
            print()


if __name__ == "__main__":
    main()
