# Released run records

Configurations, per-epoch logs, final metrics and evaluation outputs of the experiments reported in the paper.
Model weights are not included. Every number in the tables of the paper is read from the `final.json` and
`robust_suite.json` files below; nothing is typed by hand.

| Directory | Paper | Contents |
|---|---|---|
| `revision1/driver_logs/` | Tables 5–15 | Driver logs of the controlled-protocol experiments of the first revision: one line per epoch (training loss, challenging samples per batch, correction success rate, step time, memory) and the final clean, PGD and APGD accuracies |
| `revision2/main/` | Tables 7, 10, 17 and 18 | Three-seed comparisons under the shared configuration, including the doubled schedule and the natural cross-entropy term |
| `revision2/main_corrected/` | Section 3.3 (item 6) | The AT-PGD+SARA runs of Section 4.6 repeated with the corrected implementation; `compare.md` sets them against the original runs |
| `revision2/tune/` | Table 18 | The eleven-configuration tuning sweep of AT-PGD |
| `revision2/pilot/` | Table 16 | The corrective-term ablation; `pilot/cifar10/tuned_b32/` holds the tuned CIFAR10 comparison with a correction budget of 32 |
| `revision2/standard/` | Appendix D | The standard-protocol study: validation-based variant selection (`selection.json`), compute factor, evaluation outputs of every run (below), alignment-gap measurements and paired statistics (`paired_stats.md`) |
| `revision2/tables/` | Section 4.6 | LaTeX table rows regenerated from the records above |

Each run directory contains `config.json` (all command-line arguments, including the seed), `train_log.jsonl`
(one line per epoch) and `final.json` (final metrics). The runs of `standard/` and `main_corrected/` also keep the
console output (`console.log`), and those of `standard/` the evaluation outputs: `robust_suite.json` (AutoAttack
stage by stage, per-image outcomes, epsilon sweep), `confidence.json` (calibration under attack, temperature
scaling, L2 robustness), `reliability.json` and `corruption.jsonl`.

## Regenerating the tables

```bash
python experiments/make_tables_standard.py results/revision2/standard OUT_DIR   # rows of Table D.1
python experiments/compare_fixed_r2.py                                           # main_corrected/compare.md
```

The first command reproduces the rows of Table D.1 exactly. `experiments/make_tables.py` writes the rows of
Section 4.6 from a `runs/` directory with the same layout as `revision2/`.

## Notes on the records

- In the first revision, some SARA runs with the unbounded Confidence Guard loss diverged (training loss `NaN`,
  Section 4.5 of the paper). These runs were stopped by hand, so their driver logs end with the shell message
  `Killed`; the logs are kept unchanged.
- Attempts that stopped at start-up because the shared GPU had no free memory produced no results, are used in no
  table, and are not included.
- Paths in the records are relative to the repository root.
