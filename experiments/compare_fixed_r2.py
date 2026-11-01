"""Compare the AT-PGD+SARA runs of Section 4.6 rerun with the corrected implementation
(results/revision2/main_corrected) with the original runs (results/revision2/main) and with AT-PGD.
Writes results/revision2/main_corrected/compare.md."""
import glob
import json
import statistics as st


def metrics(path):
    d = json.load(open(path))
    t = [v for k, v in d.items() if k.startswith("test@")][0]
    return {"clean": t["clean_acc"], "pgd": t["pgd_acc"], "apgd": t.get("apgd_acc"),
            "aa": d.get("autoattack", {}).get("autoattack_acc")}


def mean(runs, m):
    v = [r[m] for r in runs if r.get(m) is not None]
    return f"{st.mean(v):.2f}" if v else "--"


lines = ["| Dataset | Run | Seeds | Clean | PGD | APGD | AA (1k) |", "|---|---|---|---|---|---|---|"]
for ds in ["mnist", "cifar10", "svhn"]:
    for label, pattern in [("AT-PGD", f"results/revision2/main/{ds}/at_pgd/seed*/final.json"),
                           ("AT-PGD+SARA, original", f"results/revision2/main/{ds}/at_pgd_sara/seed*/final.json"),
                           ("AT-PGD+SARA, corrected",
                            f"results/revision2/main_corrected/{ds}/at_pgd_sara/seed*/final.json")]:
        runs = [metrics(p) for p in sorted(glob.glob(pattern))]
        if runs:
            lines.append(f"| {ds} | {label} | {len(runs)} | " + " | ".join(mean(runs, m) for m in
                                                                        ["clean", "pgd", "apgd", "aa"]) + " |")
open("results/revision2/main_corrected/compare.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
