# Paired statistics, standard protocol

Differences are the first run minus the second, same seed (test set). Accuracy metrics: higher is better. ece_* (expected calibration error, %), nll_* and conf_err_* (confident mistakes: wrong with confidence > 0.8, % of inputs): lower is better. adv = PGD-20 inputs; l2 = APGD-CE, L2, eps 0.5; corruption_mean = mean accuracy over 4 corruptions x 3 severities. *_ts: after temperature scaling fitted on the clean validation split; aurc_adv: area under the risk-coverage curve on PGD-20 inputs (lower is better); sel80_adv: accuracy on the 80% most confident PGD-20 inputs.

## cifar10

### at_pgd_sara vs at_pgd (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | -0.07, -0.26, -0.43 | -0.25 | [-0.70, +0.19] | 0/3 |
| pgd | +0.24, +0.01, -0.33 | -0.03 | [-0.74, +0.69] | 2/3 |
| apgd | +0.12, -0.18, -0.28 | -0.11 | [-0.63, +0.40] | 1/3 |
| aa1k | -0.20, +0.10, +0.40 | +0.10 | [-0.65, +0.85] | 2/3 |
| aa2k | +0.35, -0.15, -0.25 | -0.02 | [-0.82, +0.78] | 1/3 |
| whitebox5k | +0.04, -0.28, +0.04 | -0.07 | [-0.53, +0.39] | 2/3 |
| l2 | -0.95, -1.00, +0.65 | -0.43 | [-2.76, +1.90] | 1/3 |
| corruption_mean | +0.10, -0.18, +0.02 | -0.02 | [-0.37, +0.33] | 2/3 |
| ece_clean | +0.28, +0.42, +0.32 | +0.34 | [+0.16, +0.52] | 3/3 |
| ece_adv | -0.79, -0.77, -0.90 | -0.82 | [-0.99, -0.65] | 0/3 |
| nll_adv | -0.00, -0.00, -0.00 | -0.00 | [-0.01, +0.00] | 0/3 |
| conf_err_clean | +0.02, -0.07, -0.07 | -0.04 | [-0.17, +0.09] | 1/3 |
| conf_err_adv | -0.42, -0.96, -0.60 | -0.66 | [-1.34, +0.02] | 0/3 |
| ece_clean_ts | -0.14, -0.17, -0.34 | -0.22 | [-0.48, +0.05] | 0/3 |
| ece_adv_ts | -0.24, +0.06, -0.48 | -0.22 | [-0.90, +0.46] | 1/3 |
| aurc_adv | -0.15, -0.11, +0.21 | -0.02 | [-0.50, +0.47] | 1/3 |
| sel80_adv | -0.10, +0.08, -0.45 | -0.16 | [-0.82, +0.51] | 1/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara 152, only-at_pgd 153, exact p = 1
- seed 0: n=2000, only at_pgd_sara robust 51, only at_pgd robust 44, p=0.538
- seed 1: n=2000, only at_pgd_sara robust 48, only at_pgd robust 51, p=0.841
- seed 2: n=2000, only at_pgd_sara robust 53, only at_pgd robust 58, p=0.704

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara 358, only-at_pgd 368, exact p = 0.738
- seed 0: n=5000, only at_pgd_sara robust 114, only at_pgd robust 112, p=0.947
- seed 1: n=5000, only at_pgd_sara robust 110, only at_pgd robust 124, p=0.395
- seed 2: n=5000, only at_pgd_sara robust 134, only at_pgd robust 132, p=0.951

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara: 0.00, 0.00, 0.00; at_pgd: 0.00, 0.00, 0.00

### at_pgd_sara vs at_pgd_matched (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | -2.78, -2.93, -2.92 | -2.88 | [-3.09, -2.67] | 0/3 |
| pgd | +1.04, +0.96, +0.84 | +0.95 | [+0.70, +1.20] | 3/3 |
| apgd | +0.99, +0.81, +0.88 | +0.89 | [+0.67, +1.12] | 3/3 |
| aa1k | +0.90, +0.80, +0.40 | +0.70 | [+0.04, +1.36] | 3/3 |
| aa2k | +0.85, -0.10, +0.70 | +0.48 | [-0.79, +1.75] | 2/3 |
| whitebox5k | +0.42, -0.12, +0.20 | +0.17 | [-0.51, +0.84] | 2/3 |
| l2 | +0.40, +0.20, +0.80 | +0.47 | [-0.29, +1.23] | 3/3 |
| corruption_mean | -1.72, -1.50, -1.06 | -1.43 | [-2.26, -0.59] | 0/3 |
| ece_clean | +4.76, +5.18, +4.80 | +4.91 | [+4.33, +5.49] | 3/3 |
| ece_adv | -9.05, -9.21, -9.56 | -9.27 | [-9.92, -8.63] | 0/3 |
| nll_adv | -0.14, -0.14, -0.15 | -0.14 | [-0.16, -0.12] | 0/3 |
| conf_err_clean | -0.48, -0.51, -0.71 | -0.57 | [-0.88, -0.26] | 0/3 |
| conf_err_adv | -5.36, -6.76, -6.28 | -6.13 | [-7.90, -4.37] | 0/3 |
| ece_clean_ts | -0.93, -0.55, -0.98 | -0.82 | [-1.41, -0.23] | 0/3 |
| ece_adv_ts | -3.28, -2.91, -4.34 | -3.51 | [-5.35, -1.67] | 0/3 |
| aurc_adv | -1.57, -1.43, -1.73 | -1.57 | [-1.94, -1.21] | 0/3 |
| sel80_adv | +1.15, +1.15, +0.98 | +1.09 | [+0.84, +1.34] | 3/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara 260, only-at_pgd_matched 231, exact p = 0.206
- seed 0: n=2000, only at_pgd_sara robust 89, only at_pgd_matched robust 72, p=0.207
- seed 1: n=2000, only at_pgd_sara robust 82, only at_pgd_matched robust 84, p=0.938
- seed 2: n=2000, only at_pgd_sara robust 89, only at_pgd_matched robust 75, p=0.31

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara 606, only-at_pgd_matched 581, exact p = 0.486
- seed 0: n=5000, only at_pgd_sara robust 208, only at_pgd_matched robust 187, p=0.314
- seed 1: n=5000, only at_pgd_sara robust 199, only at_pgd_matched robust 205, p=0.804
- seed 2: n=5000, only at_pgd_sara robust 199, only at_pgd_matched robust 189, p=0.648

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara: 0.00, 0.00, 0.00; at_pgd_matched: 0.00, 0.00, 0.00

### at_pgd_sara_ep44 vs at_pgd_matched (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | +0.01, -0.25, -0.17 | -0.14 | [-0.47, +0.19] | 1/3 |
| pgd | +0.50, +0.43, +0.10 | +0.34 | [-0.19, +0.87] | 3/3 |
| apgd | +0.54, +0.24, -0.17 | +0.20 | [-0.68, +1.09] | 2/3 |
| aa1k | +1.00, +0.60, +0.00 | +0.53 | [-0.72, +1.78] | 2/3 |
| aa2k | +0.75, -0.55, +0.55 | +0.25 | [-1.49, +1.99] | 2/3 |
| whitebox5k | +0.16, +0.02, -0.20 | -0.01 | [-0.46, +0.44] | 2/3 |
| l2 | +0.55, -0.15, +0.25 | +0.22 | [-0.66, +1.09] | 2/3 |
| corruption_mean | +0.07, +0.10, +0.35 | +0.17 | [-0.21, +0.56] | 3/3 |
| ece_clean | +0.42, +0.55, +0.58 | +0.52 | [+0.31, +0.73] | 3/3 |
| ece_adv | -0.81, -1.04, -1.20 | -1.02 | [-1.51, -0.52] | 0/3 |
| nll_adv | -0.01, -0.02, -0.03 | -0.02 | [-0.04, -0.01] | 0/3 |
| conf_err_clean | -0.04, -0.11, -0.34 | -0.16 | [-0.55, +0.23] | 0/3 |
| conf_err_adv | -0.58, -1.84, -1.52 | -1.31 | [-2.94, +0.31] | 0/3 |
| ece_clean_ts | -0.40, -0.00, -0.36 | -0.25 | [-0.80, +0.29] | 0/3 |
| ece_adv_ts | -0.48, -0.15, -0.28 | -0.30 | [-0.71, +0.10] | 0/3 |
| aurc_adv | -0.10, -0.25, -0.23 | -0.19 | [-0.40, +0.01] | 0/3 |
| sel80_adv | +0.30, +0.42, -0.25 | +0.16 | [-0.73, +1.05] | 2/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara_ep44 213, only-at_pgd_matched 198, exact p = 0.49
- seed 0: n=2000, only at_pgd_sara_ep44 robust 70, only at_pgd_matched robust 55, p=0.21
- seed 1: n=2000, only at_pgd_sara_ep44 robust 62, only at_pgd_matched robust 73, p=0.39
- seed 2: n=2000, only at_pgd_sara_ep44 robust 81, only at_pgd_matched robust 70, p=0.416

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara_ep44 472, only-at_pgd_matched 473, exact p = 1
- seed 0: n=5000, only at_pgd_sara_ep44 robust 162, only at_pgd_matched robust 154, p=0.694
- seed 1: n=5000, only at_pgd_sara_ep44 robust 154, only at_pgd_matched robust 153, p=1
- seed 2: n=5000, only at_pgd_sara_ep44 robust 156, only at_pgd_matched robust 166, p=0.616

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara_ep44: 0.00, 0.00, 0.00; at_pgd_matched: 0.00, 0.00, 0.00

### at_pgd_sara vs at_pgd_sara_acceptall (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | +1.93, +1.96, +2.27 | +2.05 | [+1.59, +2.52] | 3/3 |
| pgd | +0.38, +0.70, +0.90 | +0.66 | [+0.01, +1.31] | 3/3 |
| apgd | +0.28, +0.30, +0.76 | +0.45 | [-0.23, +1.12] | 3/3 |
| aa1k | +1.80, +2.20, +1.80 | +1.93 | [+1.36, +2.51] | 3/3 |
| aa2k | +2.15, +1.75, +1.45 | +1.78 | [+0.91, +2.66] | 3/3 |
| whitebox5k | +1.92, +1.28, +1.62 | +1.61 | [+0.81, +2.40] | 3/3 |
| l2 | -0.20, +1.00, +0.95 | +0.58 | [-1.10, +2.27] | 2/3 |
| corruption_mean | +1.58, +1.89, +1.79 | +1.75 | [+1.36, +2.15] | 3/3 |
| ece_clean | -5.82, -5.31, -6.05 | -5.73 | [-6.67, -4.78] | 0/3 |
| ece_adv | +6.07, +5.81, +5.93 | +5.94 | [+5.61, +6.26] | 3/3 |
| nll_adv | -0.01, -0.01, -0.02 | -0.01 | [-0.02, +0.00] | 0/3 |
| conf_err_clean | +0.17, +0.20, +0.16 | +0.18 | [+0.12, +0.23] | 3/3 |
| conf_err_adv | +2.20, +1.82, +2.06 | +2.03 | [+1.55, +2.50] | 3/3 |
| ece_clean_ts | -0.99, -0.64, -0.95 | -0.86 | [-1.34, -0.38] | 0/3 |
| ece_adv_ts | +3.22, +2.40, +2.06 | +2.56 | [+1.08, +4.05] | 3/3 |
| aurc_adv | -0.34, -0.51, -0.52 | -0.46 | [-0.71, -0.20] | 0/3 |
| sel80_adv | +0.58, +0.73, +0.63 | +0.64 | [+0.45, +0.83] | 3/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara 234, only-at_pgd_sara_acceptall 127, exact p = 1.92e-08
- seed 0: n=2000, only at_pgd_sara robust 87, only at_pgd_sara_acceptall robust 44, p=0.000216
- seed 1: n=2000, only at_pgd_sara robust 76, only at_pgd_sara_acceptall robust 41, p=0.00156
- seed 2: n=2000, only at_pgd_sara robust 71, only at_pgd_sara_acceptall robust 42, p=0.00815

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara 565, only-at_pgd_sara_acceptall 324, exact p = 5.55e-16
- seed 0: n=5000, only at_pgd_sara robust 199, only at_pgd_sara_acceptall robust 103, p=3.55e-08
- seed 1: n=5000, only at_pgd_sara robust 177, only at_pgd_sara_acceptall robust 113, p=0.000205
- seed 2: n=5000, only at_pgd_sara robust 189, only at_pgd_sara_acceptall robust 108, p=3.02e-06

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara: 0.00, 0.00, 0.00; at_pgd_sara_acceptall: 0.00, 0.05, 0.00

### at_pgd_sara vs inverse_at_like (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | +18.36, +8.62, +8.68 | +11.89 | [-2.04, +25.81] | 3/3 |
| pgd | +8.54, +3.92, +3.68 | +5.38 | [-1.43, +12.19] | 3/3 |
| apgd | +8.20, +3.67, +3.75 | +5.21 | [-1.23, +11.65] | 3/3 |
| aa1k | +5.90, +3.30, +3.20 | +4.13 | [+0.33, +7.94] | 3/3 |
| aa2k | +8.40, +4.05, +3.85 | +5.43 | [-0.95, +11.82] | 3/3 |
| whitebox5k | +8.12, +3.68, +3.50 | +5.10 | [-1.40, +11.60] | 3/3 |
| l2 | +11.15, +5.20, +7.20 | +7.85 | [+0.33, +15.37] | 3/3 |
| corruption_mean | +11.03, +4.81, +4.69 | +6.84 | [-2.16, +15.85] | 3/3 |
| ece_clean | -2.26, +1.42, +1.53 | +0.23 | [-5.13, +5.59] | 2/3 |
| ece_adv | +6.20, +0.36, +0.31 | +2.29 | [-6.12, +10.70] | 3/3 |
| nll_adv | -0.23, -0.09, -0.10 | -0.14 | [-0.34, +0.06] | 0/3 |
| conf_err_clean | +0.19, -0.09, -0.17 | -0.02 | [-0.49, +0.45] | 1/3 |
| conf_err_adv | +3.24, +0.54, +0.40 | +1.39 | [-2.58, +5.37] | 3/3 |
| ece_clean_ts | +0.64, +0.47, +0.32 | +0.48 | [+0.09, +0.87] | 3/3 |
| ece_adv_ts | +6.85, +1.39, +0.80 | +3.01 | [-5.27, +11.30] | 3/3 |
| aurc_adv | -10.57, -4.32, -4.45 | -6.44 | [-15.32, +2.43] | 0/3 |
| sel80_adv | +9.22, +4.85, +4.38 | +6.15 | [-0.49, +12.79] | 3/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara 558, only-inverse_at_like 232, exact p = 8.37e-32
- seed 0: n=2000, only at_pgd_sara robust 263, only inverse_at_like robust 95, p=2.29e-19
- seed 1: n=2000, only at_pgd_sara robust 151, only inverse_at_like robust 70, p=5.3e-08
- seed 2: n=2000, only at_pgd_sara robust 144, only inverse_at_like robust 67, p=1.23e-07

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara 1325, only-inverse_at_like 560, exact p = 2.76e-71
- seed 0: n=5000, only at_pgd_sara robust 636, only inverse_at_like robust 230, p=9.64e-45
- seed 1: n=5000, only at_pgd_sara robust 345, only inverse_at_like robust 161, p=1.92e-16
- seed 2: n=5000, only at_pgd_sara robust 344, only inverse_at_like robust 169, p=8.64e-15

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara: 0.00, 0.00, 0.00; inverse_at_like: 0.00, 0.00, 0.00

## svhn

### at_pgd_sara vs at_pgd (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | +0.35, +0.04, -0.10 | +0.09 | [-0.47, +0.66] | 2/3 |
| pgd | +0.54, +0.69, -0.02 | +0.40 | [-0.53, +1.33] | 2/3 |
| apgd | +0.48, +0.58, +0.15 | +0.40 | [-0.16, +0.97] | 3/3 |
| aa1k | +0.10, -1.00, -1.70 | -0.87 | [-3.12, +1.39] | 1/3 |
| aa2k | +0.00, +0.00, -0.35 | -0.12 | [-0.62, +0.39] | 0/3 |
| whitebox5k | +0.04, +0.06, -0.40 | -0.10 | [-0.75, +0.55] | 2/3 |
| l2 | +0.55, +0.30, +0.10 | +0.32 | [-0.24, +0.88] | 3/3 |
| corruption_mean | +0.06, +0.08, -0.08 | +0.02 | [-0.20, +0.24] | 2/3 |
| ece_clean | +1.52, +1.15, +1.24 | +1.30 | [+0.82, +1.78] | 3/3 |
| ece_adv | -2.00, -2.01, -1.97 | -1.99 | [-2.04, -1.95] | 0/3 |
| nll_adv | -0.02, -0.02, -0.01 | -0.02 | [-0.03, -0.01] | 0/3 |
| conf_err_clean | -0.03, -0.02, -0.06 | -0.03 | [-0.09, +0.03] | 0/3 |
| conf_err_adv | -0.88, -1.02, -1.36 | -1.09 | [-1.70, -0.47] | 0/3 |
| ece_clean_ts | +0.30, +0.24, +0.01 | +0.19 | [-0.19, +0.57] | 3/3 |
| ece_adv_ts | -0.92, -0.58, -0.29 | -0.60 | [-1.38, +0.19] | 0/3 |
| aurc_adv | -0.23, -0.37, -0.27 | -0.29 | [-0.46, -0.12] | 0/3 |
| sel80_adv | +0.70, +0.40, -0.25 | +0.28 | [-0.92, +1.49] | 2/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara 115, only-at_pgd 122, exact p = 0.697
- seed 0: n=2000, only at_pgd_sara robust 36, only at_pgd robust 36, p=1
- seed 1: n=2000, only at_pgd_sara robust 41, only at_pgd robust 41, p=1
- seed 2: n=2000, only at_pgd_sara robust 38, only at_pgd robust 45, p=0.51

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara 304, only-at_pgd 319, exact p = 0.575
- seed 0: n=5000, only at_pgd_sara robust 99, only at_pgd robust 97, p=0.943
- seed 1: n=5000, only at_pgd_sara robust 107, only at_pgd robust 104, p=0.891
- seed 2: n=5000, only at_pgd_sara robust 98, only at_pgd robust 118, p=0.196

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara: 0.00, 0.00, 0.00; at_pgd: 0.05, 0.00, 0.00

### at_pgd_sara vs at_pgd_matched (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | -0.12, -0.47, -0.49 | -0.36 | [-0.88, +0.16] | 0/3 |
| pgd | +2.87, +3.00, +2.40 | +2.76 | [+1.96, +3.55] | 3/3 |
| apgd | +3.04, +3.19, +2.77 | +3.00 | [+2.48, +3.52] | 3/3 |
| aa1k | +2.30, +1.70, -0.60 | +1.13 | [-2.67, +4.94] | 2/3 |
| aa2k | +1.90, +1.90, +0.35 | +1.38 | [-0.84, +3.61] | 3/3 |
| whitebox5k | +2.06, +1.96, +1.28 | +1.77 | [+0.71, +2.82] | 3/3 |
| l2 | +3.60, +3.20, +3.50 | +3.43 | [+2.92, +3.95] | 3/3 |
| corruption_mean | -0.43, -0.49, -0.54 | -0.49 | [-0.63, -0.35] | 0/3 |
| ece_clean | +5.24, +4.88, +5.01 | +5.04 | [+4.58, +5.51] | 3/3 |
| ece_adv | -11.16, -10.38, -10.81 | -10.79 | [-11.75, -9.82] | 0/3 |
| nll_adv | -0.27, -0.26, -0.25 | -0.26 | [-0.28, -0.24] | 0/3 |
| conf_err_clean | -0.33, -0.34, -0.35 | -0.34 | [-0.36, -0.32] | 0/3 |
| conf_err_adv | -9.14, -8.48, -9.54 | -9.05 | [-10.38, -7.72] | 0/3 |
| ece_clean_ts | +0.16, +0.04, -0.11 | +0.03 | [-0.30, +0.36] | 2/3 |
| ece_adv_ts | -5.29, -3.76, -4.02 | -4.36 | [-6.39, -2.33] | 0/3 |
| aurc_adv | -4.45, -4.00, -4.26 | -4.24 | [-4.81, -3.67] | 0/3 |
| sel80_adv | +4.22, +3.70, +3.65 | +3.86 | [+3.07, +4.65] | 3/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara 187, only-at_pgd_matched 104, exact p = 1.31e-06
- seed 0: n=2000, only at_pgd_sara robust 70, only at_pgd_matched robust 32, p=0.000213
- seed 1: n=2000, only at_pgd_sara robust 68, only at_pgd_matched robust 30, p=0.000156
- seed 2: n=2000, only at_pgd_sara robust 49, only at_pgd_matched robust 42, p=0.53

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara 532, only-at_pgd_matched 267, exact p = 4.21e-21
- seed 0: n=5000, only at_pgd_sara robust 188, only at_pgd_matched robust 85, p=4.18e-10
- seed 1: n=5000, only at_pgd_sara robust 182, only at_pgd_matched robust 84, p=1.8e-09
- seed 2: n=5000, only at_pgd_sara robust 162, only at_pgd_matched robust 98, p=8.66e-05

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara: 0.00, 0.00, 0.00; at_pgd_matched: 0.05, 0.05, 0.00

### at_pgd_sara_ep29 vs at_pgd_matched (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | +0.31, +0.08, +0.06 | +0.15 | [-0.19, +0.49] | 3/3 |
| pgd | +0.55, +0.73, -0.02 | +0.42 | [-0.54, +1.39] | 2/3 |
| apgd | +0.48, +0.59, +0.10 | +0.39 | [-0.25, +1.03] | 3/3 |
| aa1k | +0.20, +0.40, -1.00 | -0.13 | [-2.01, +1.75] | 2/3 |
| aa2k | +0.15, +0.40, -0.40 | +0.05 | [-0.97, +1.07] | 2/3 |
| whitebox5k | +0.26, +0.06, +0.04 | +0.12 | [-0.18, +0.42] | 3/3 |
| l2 | +0.45, +0.15, +0.55 | +0.38 | [-0.13, +0.90] | 3/3 |
| corruption_mean | +0.03, -0.20, -0.12 | -0.10 | [-0.39, +0.20] | 1/3 |
| ece_clean | +1.30, +0.99, +1.28 | +1.19 | [+0.76, +1.61] | 3/3 |
| ece_adv | -2.42, -1.56, -1.69 | -1.89 | [-3.03, -0.75] | 0/3 |
| nll_adv | -0.04, -0.03, -0.04 | -0.04 | [-0.05, -0.02] | 0/3 |
| conf_err_clean | +0.00, -0.02, -0.05 | -0.02 | [-0.10, +0.05] | 1/3 |
| conf_err_adv | -1.64, -1.16, -2.46 | -1.75 | [-3.39, -0.12] | 0/3 |
| ece_clean_ts | +0.57, +0.36, +0.27 | +0.40 | [+0.01, +0.79] | 3/3 |
| ece_adv_ts | -1.69, +0.01, +0.09 | -0.53 | [-3.02, +1.96] | 2/3 |
| aurc_adv | -0.98, -0.40, -0.68 | -0.69 | [-1.40, +0.03] | 0/3 |
| sel80_adv | +0.77, +0.20, +0.02 | +0.33 | [-0.64, +1.31] | 3/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara_ep29 127, only-at_pgd_matched 124, exact p = 0.9
- seed 0: n=2000, only at_pgd_sara_ep29 robust 37, only at_pgd_matched robust 34, p=0.813
- seed 1: n=2000, only at_pgd_sara_ep29 robust 47, only at_pgd_matched robust 39, p=0.451
- seed 2: n=2000, only at_pgd_sara_ep29 robust 43, only at_pgd_matched robust 51, p=0.47

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara_ep29 355, only-at_pgd_matched 337, exact p = 0.518
- seed 0: n=5000, only at_pgd_sara_ep29 robust 113, only at_pgd_matched robust 100, p=0.411
- seed 1: n=5000, only at_pgd_sara_ep29 robust 125, only at_pgd_matched robust 122, p=0.899
- seed 2: n=5000, only at_pgd_sara_ep29 robust 117, only at_pgd_matched robust 115, p=0.948

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara_ep29: 0.05, 0.00, 0.05; at_pgd_matched: 0.05, 0.05, 0.00

### at_pgd_sara vs at_pgd_sara_acceptall (seeds [0, 1, 2])

| Metric | Per-seed differences | Mean | 95% paired-t interval | Seeds > 0 |
|---|---|---|---|---|
| clean | +0.33, -0.20, -0.06 | +0.02 | [-0.66, +0.71] | 1/3 |
| pgd | -0.49, -0.50, -0.86 | -0.61 | [-1.14, -0.09] | 0/3 |
| apgd | -0.45, -0.41, -0.67 | -0.51 | [-0.85, -0.17] | 0/3 |
| aa1k | +0.50, +0.40, -1.60 | -0.23 | [-3.18, +2.71] | 2/3 |
| aa2k | +0.60, +0.45, -0.65 | +0.13 | [-1.56, +1.83] | 2/3 |
| whitebox5k | +0.84, +0.54, -0.20 | +0.39 | [-0.94, +1.72] | 2/3 |
| l2 | -0.40, -1.85, -1.20 | -1.15 | [-2.95, +0.65] | 0/3 |
| corruption_mean | +0.34, -0.24, +0.11 | +0.07 | [-0.65, +0.79] | 2/3 |
| ece_clean | -1.49, -1.92, -1.78 | -1.73 | [-2.28, -1.19] | 0/3 |
| ece_adv | +1.35, +2.02, +1.33 | +1.56 | [+0.58, +2.54] | 3/3 |
| nll_adv | +0.00, +0.02, +0.02 | +0.01 | [-0.01, +0.03] | 3/3 |
| conf_err_clean | +0.02, +0.06, +0.06 | +0.04 | [-0.02, +0.11] | 3/3 |
| conf_err_adv | +0.54, +1.10, +0.84 | +0.83 | [+0.13, +1.52] | 3/3 |
| ece_clean_ts | -0.09, -0.50, -0.32 | -0.30 | [-0.81, +0.21] | 0/3 |
| ece_adv_ts | +0.62, +1.21, +1.16 | +0.99 | [+0.19, +1.80] | 3/3 |
| aurc_adv | +0.10, +0.64, +0.68 | +0.47 | [-0.33, +1.28] | 3/3 |
| sel80_adv | -0.15, -0.65, -1.05 | -0.62 | [-1.74, +0.50] | 0/3 |

AutoAttack (all four stages, first 2,000 images), McNemar on the same images: pooled only-at_pgd_sara 124, only-at_pgd_sara_acceptall 116, exact p = 0.651
- seed 0: n=2000, only at_pgd_sara robust 42, only at_pgd_sara_acceptall robust 30, p=0.195
- seed 1: n=2000, only at_pgd_sara robust 51, only at_pgd_sara_acceptall robust 42, p=0.407
- seed 2: n=2000, only at_pgd_sara robust 31, only at_pgd_sara_acceptall robust 44, p=0.165

AutoAttack white-box stages (first 5,000 images), McNemar on the same images: pooled only-at_pgd_sara 351, only-at_pgd_sara_acceptall 292, exact p = 0.0221
- seed 0: n=5000, only at_pgd_sara robust 130, only at_pgd_sara_acceptall robust 88, p=0.00537
- seed 1: n=5000, only at_pgd_sara robust 123, only at_pgd_sara_acceptall robust 96, p=0.0787
- seed 2: n=5000, only at_pgd_sara robust 98, only at_pgd_sara_acceptall robust 108, p=0.531

Accuracy removed by Square after the white-box attacks had failed, points (gradient-masking check): at_pgd_sara: 0.00, 0.00, 0.00; at_pgd_sara_acceptall: 0.05, 0.00, 0.00

