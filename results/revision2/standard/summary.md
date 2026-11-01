## cifar10

| Method | Seeds | Clean | PGD-20 | APGD-CE | AutoAttack (1k) | Train time (s) |
|---|---|---|---|---|---|---|
| AT-PGD, 30 epochs | 3 | 80.52 ± 0.08 | 49.95 ± 0.22 | 49.67 ± 0.25 | 46.60 ± 0.22 | 4068 |
| AT-PGD+SARA, 30 epochs | 3 | 80.27 ± 0.09 | 49.92 ± 0.02 | 49.56 ± 0.09 | 46.70 ± 0.08 | 5918 |
| pilot b6_eval (10 epochs) | 1 | 66.35 | 39.57 | 39.36 | -- | 2123 |
| pilot b3_k20 (10 epochs) | 1 | 66.93 | 41.02 | 40.75 | -- | 2646 |
| pilot at (10 epochs) | 1 | 67.57 | 41.09 | 41.06 | -- | 1362 |
| pilot b3 (10 epochs) | 1 | 67.06 | 40.97 | 40.81 | -- | 2112 |
| AT-PGD, compute-matched epochs | 3 | 83.15 ± 0.08 | 48.98 ± 0.07 | 48.67 ± 0.13 | 46.00 ± 0.24 | 6382 |
| at_pgd_sara_ep44 | 3 | 83.01 ± 0.09 | 49.32 ± 0.11 | 48.87 ± 0.17 | 46.53 ± 0.17 | 8436 |
| pilot b3_eval (10 epochs) | 1 | 66.77 | 40.75 | 40.66 | -- | 2039 |
| pilot b10 (10 epochs) | 1 | 66.70 | 40.86 | 40.69 | -- | 2422 |
| pilot b3_r2 (10 epochs) | 1 | 66.78 | 40.91 | 40.86 | -- | 2010 |
| pilot b3_xi02 (10 epochs) | 1 | 67.44 | 41.60 | 41.36 | -- | 1990 |
| pilot b6 (10 epochs) | 1 | 67.29 | 41.31 | 41.10 | -- | 2205 |
| SARA without acceptance test | 3 | 78.22 ± 0.24 | 49.26 ± 0.23 | 49.11 ± 0.13 | 44.77 ± 0.12 | 6067 |
| Inverse-AT-like (all samples, no test) | 3 | 68.38 ± 4.53 | 44.54 ± 2.22 | 44.35 ± 2.17 | 42.57 ± 1.32 | 8763 |

Paired differences (same seed, AT-PGD+SARA minus the other run):

- vs AT-PGD, 30 epochs (seeds [0, 1, 2]): clean -0.25, pgd -0.03, apgd -0.11, aa +0.10
- vs pilot b6_eval (10 epochs) (seeds [0]): clean +13.99, pgd +10.37, apgd +10.12
- vs pilot b3_k20 (10 epochs) (seeds [0]): clean +13.41, pgd +8.92, apgd +8.73
- vs pilot at (10 epochs) (seeds [0]): clean +12.77, pgd +8.85, apgd +8.42
- vs pilot b3 (10 epochs) (seeds [0]): clean +13.28, pgd +8.97, apgd +8.67
- vs AT-PGD, compute-matched epochs (seeds [0, 1, 2]): clean -2.88, pgd +0.95, apgd +0.89, aa +0.70
- vs at_pgd_sara_ep44 (seeds [0, 1, 2]): clean -2.74, pgd +0.60, apgd +0.69, aa +0.17
- vs pilot b3_eval (10 epochs) (seeds [0]): clean +13.57, pgd +9.19, apgd +8.82
- vs pilot b10 (10 epochs) (seeds [0]): clean +13.64, pgd +9.08, apgd +8.79
- vs pilot b3_r2 (10 epochs) (seeds [0]): clean +13.56, pgd +9.03, apgd +8.62
- vs pilot b3_xi02 (10 epochs) (seeds [0]): clean +12.90, pgd +8.34, apgd +8.12
- vs pilot b6 (10 epochs) (seeds [0]): clean +13.05, pgd +8.63, apgd +8.38
- vs SARA without acceptance test (seeds [0, 1, 2]): clean +2.05, pgd +0.66, apgd +0.45, aa +1.93
- vs Inverse-AT-like (all samples, no test) (seeds [0, 1, 2]): clean +11.89, pgd +5.38, apgd +5.21, aa +4.13
- eta (seed0): median 0.5013, p10 0.2645, p90 0.8629; below tau^2/2 = 0.00125: 0.0%; Prop. 4 bounds hold: 100.0% / 100.0%; adversarial example correct: 0.0% (n=165)
- eta (seed1): median 0.4982, p10 0.2555, p90 0.8692; below tau^2/2 = 0.00125: 0.0%; Prop. 4 bounds hold: 100.0% / 100.0%; adversarial example correct: 0.0% (n=163)
- eta (seed2): median 0.5231, p10 0.2310, p90 0.8565; below tau^2/2 = 0.00125: 0.0%; Prop. 4 bounds hold: 100.0% / 100.0%; adversarial example correct: 0.0% (n=142)

## svhn

| Method | Seeds | Clean | PGD-20 | APGD-CE | AutoAttack (1k) | Train time (s) |
|---|---|---|---|---|---|---|
| AT-PGD, 20 epochs | 3 | 90.91 ± 0.05 | 49.49 ± 0.05 | 47.54 ± 0.03 | 42.60 ± 0.36 | 4075 |
| AT-PGD+SARA, 20 epochs | 3 | 91.00 ± 0.23 | 49.89 ± 0.26 | 47.94 ± 0.22 | 41.73 ± 0.74 | 5512 |
| at_pgd_sara_ep29 | 3 | 91.51 ± 0.17 | 47.55 ± 0.31 | 45.33 ± 0.26 | 40.47 ± 0.24 | 8468 |
| AT-PGD, compute-matched epochs | 3 | 91.36 ± 0.06 | 47.13 ± 0.02 | 44.94 ± 0.05 | 40.60 ± 0.51 | 5941 |
| SARA without acceptance test | 3 | 90.98 ± 0.09 | 50.50 ± 0.09 | 48.45 ± 0.11 | 41.97 ± 0.25 | 5483 |

Paired differences (same seed, AT-PGD+SARA minus the other run):

- vs AT-PGD, 20 epochs (seeds [0, 1, 2]): clean +0.09, pgd +0.40, apgd +0.40, aa -0.87
- vs at_pgd_sara_ep29 (seeds [0, 1, 2]): clean -0.51, pgd +2.34, apgd +2.61, aa +1.27
- vs AT-PGD, compute-matched epochs (seeds [0, 1, 2]): clean -0.36, pgd +2.76, apgd +3.00, aa +1.13
- vs SARA without acceptance test (seeds [0, 1, 2]): clean +0.02, pgd -0.61, apgd -0.51, aa -0.23
- eta (seed0): median 0.3637, p10 0.1729, p90 0.7573; below tau^2/2 = 0.00125: 0.0%; Prop. 4 bounds hold: 100.0% / 100.0%; adversarial example correct: 0.0% (n=1035)
- eta (seed1): median 0.3754, p10 0.1578, p90 0.7564; below tau^2/2 = 0.00125: 0.0%; Prop. 4 bounds hold: 100.0% / 100.0%; adversarial example correct: 0.0% (n=1025)
- eta (seed2): median 0.3460, p10 0.1573, p90 0.7175; below tau^2/2 = 0.00125: 0.0%; Prop. 4 bounds hold: 100.0% / 100.0%; adversarial example correct: 0.0% (n=999)

