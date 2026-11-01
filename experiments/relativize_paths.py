"""Rewrite the out_dir recorded in run configurations to a path relative to the repository root.

Runs launched with an absolute --out-dir record that path in config.json; this makes the released
configurations independent of the machine on which they were produced.

Usage: python experiments/relativize_paths.py DIR [DIR ...]
"""
import json
import os
import sys


def relativize(path: str) -> str:
    path = os.path.normpath(path)
    marker = os.sep + "runs" + os.sep
    i = path.rfind(marker)
    return path[i + 1:] if i >= 0 else path


def main():
    changed = 0
    for root_dir in sys.argv[1:]:
        for root, _, files in os.walk(root_dir):
            if "config.json" not in files:
                continue
            p = os.path.join(root, "config.json")
            cfg = json.load(open(p))
            rel = relativize(cfg.get("out_dir", ""))
            if rel != cfg.get("out_dir"):
                cfg["out_dir"] = rel
                with open(p, "w") as f:
                    json.dump(cfg, f, indent=2)
                changed += 1
    print(f"{changed} configuration(s) updated")


if __name__ == "__main__":
    main()
