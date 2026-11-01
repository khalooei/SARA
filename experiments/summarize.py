"""Print final test (and validation) accuracies of every run below a directory.

    python experiments/summarize.py runs/pilot          # one line per run
    python experiments/summarize.py runs/main --seeds   # mean and population std over seed*/ subdirectories
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
by_seed = "--seeds" in sys.argv[2:]


def metrics(d):
    out = {}
    for k, v in d.items():
        if k.startswith("test@"):
            eps = k[5:]
            out["clean"] = v["clean_acc"]
            out[f"pgd@{eps}"] = v["pgd_acc"]
            if "apgd_acc" in v:
                out[f"apgd@{eps}"] = v["apgd_acc"]
        elif k == "val":
            out["val_pgd"] = v["pgd_acc"]
        elif k == "autoattack":
            out["aa"] = v["autoattack_acc"]
    out["time_s"] = d["train_time_s"]
    return out


if not by_seed:
    for f in sorted(root.rglob("final.json")):
        d = json.loads(f.read_text())
        parts = []
        for k, v in d.items():
            if k.startswith("test@"):
                parts.append(f"{k[5:]}: {v['clean_acc']:.2f}/{v['pgd_acc']:.2f}/{v.get('apgd_acc', float('nan')):.2f}")
            elif k == "val":
                parts.append(f"val {v['clean_acc']:.2f}/{v['pgd_acc']:.2f}/{v.get('apgd_acc', float('nan')):.2f}")
            elif k == "autoattack":
                parts.append(f"AA {v['autoattack_acc']:.2f}")
        print(f"{str(f.parent.relative_to(root)):45s} " + " | ".join(parts) + f"  ({d['train_time_s']:.0f}s)")
    sys.exit()

groups = defaultdict(list)
for f in sorted(root.rglob("seed*/final.json")):
    groups[str(f.parent.parent.relative_to(root))].append(metrics(json.loads(f.read_text())))
for name, runs in groups.items():
    keys = [k for k in runs[0] if all(k in r for r in runs)]
    cells = []
    for k in keys:
        a = np.array([r[k] for r in runs])
        cells.append(f"{k} {a.mean():.0f}" if k == "time_s" else f"{k} {a.mean():.2f}±{a.std():.2f}")
    print(f"{name:40s} n={len(runs)}  " + "  ".join(cells))
