#!/usr/bin/env bash
# Phase C of the controlled study (Section 4.6): variants aimed at clean accuracy (FAT with SARA,
# a natural cross-entropy term, alignment on natural inputs), next to the selected alignment configuration.
set -e
cd "$(dirname "$0")/.."
mkdir -p runs
SA="--sara --sara-max-correct 64 --cg-confidence-cap --sara-align 3 --xi 0.3"
python experiments/gen_main.py runs/main sara \
  "fat_sara=--base-loss fat ${SA};at_pgd_sara_nat=--base-loss pgd ${SA} --nat-weight 1 --sara-align-natural 1;at_pgd_nat=--base-loss pgd --nat-weight 1" \
  > runs/phaseC_commands.txt
# seed-0 pilots of extra clean-oriented variants (three datasets)
python experiments/gen_main.py runs/pilot3 sara \
  "standard_sara_nat=--sara --sara-max-correct 64 --cg-confidence-cap --sara-align-natural 1;at_pgd_sara_an=--base-loss pgd ${SA} --sara-align-natural 1" \
  0 >> runs/phaseC_commands.txt
wc -l runs/phaseC_commands.txt
