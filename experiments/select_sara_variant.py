"""Select the SARA variant of the standard-protocol study on the CIFAR10 validation split.

Reads the pilot runs (experiments/standard_protocol.sh, RUN=pilot_*), chooses the SARA variant with the highest
validation PGD accuracy (validation APGD accuracy breaks ties) and writes, in OUT_DIR:

  selected_sara_args.txt  SARA arguments used by every later run of the study
  compute_factor.txt      training time of the selected variant relative to AT-PGD (at least 1.0), which sets the
                          number of epochs of the compute-matched AT-PGD run
  selection.json          validation metrics of every pilot and the choice

Test-set metrics are never read.

Usage: python experiments/select_sara_variant.py OUT_DIR
"""
import glob
import json
import os
import re
import sys

OUT = sys.argv[1]
pilots = {}
for d in sorted(glob.glob(os.path.join(OUT, "cifar10", "pilot_*", "seed0"))):
    final = os.path.join(d, "final.json")
    if not os.path.exists(final):
        continue
    j = json.load(open(final))
    args = open(os.path.join(d, "run_args.txt")).read().strip()
    pilots[d.split(os.sep)[-2]] = {"val": j["val"], "train_time_s": j["train_time_s"], "args": args}

if "pilot_at" not in pilots or len(pilots) < 2:
    sys.exit(f"not enough pilot runs in {OUT}/cifar10: {sorted(pilots)}")

variants = {k: v for k, v in pilots.items() if k != "pilot_at"}
best = max(variants, key=lambda k: (variants[k]["val"]["pgd_acc"], variants[k]["val"]["apgd_acc"]))
sara_args = re.sub(r"--epochs \d+|--eval-autoattack \d+|--sara(?=\s|$)", "", variants[best]["args"])
sara_args = " ".join(sara_args.split())
factor = max(1.0, variants[best]["train_time_s"] / pilots["pilot_at"]["train_time_s"])

open(os.path.join(OUT, "selected_sara_args.txt"), "w").write(sara_args + "\n")
open(os.path.join(OUT, "compute_factor.txt"), "w").write(f"{factor:.3f}\n")
record = {
    "criterion": "highest validation PGD-20 accuracy (validation APGD-CE breaks ties); test metrics not used",
    "selected": best, "selected_args": sara_args, "compute_factor": round(factor, 3),
    "pilots": {k: {"val_clean": v["val"]["clean_acc"], "val_pgd": v["val"]["pgd_acc"], "val_apgd": v["val"]["apgd_acc"],
                   "train_time_s": round(v["train_time_s"]), "args": v["args"]} for k, v in pilots.items()},
}
json.dump(record, open(os.path.join(OUT, "selection.json"), "w"), indent=2)
print(json.dumps(record, indent=2))
