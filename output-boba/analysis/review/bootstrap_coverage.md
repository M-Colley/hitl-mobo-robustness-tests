# Coverage of the landscape-cluster bootstrap at 20 clusters

Produced by `scripts/bootstrap_coverage.py` on 2026-09-22; R = 10000 simulated studies, B = 2000 replicates per study, B = 20000 for the intervals on the actual data; seed 20260922; 22 s.

## Question

A reviewer asked whether a 95% interval that resamples 20 landscape clusters covers 95% of the time, and asked for percentile intervals beside BCa. Every headline interval in the paper is a landscape-cluster bootstrap of a ratio of landscape means or of a landscape mean. The selection share and the remedy recoveries are printed with percentile intervals (`decompose_regret.py`, `replay_hitl_remedies.py`); BCa is used in the currency fit (`analyse_boba_robustness.py`), which this check does not cover.

## Method

The 20 observed landscapes are treated as the population, so the true value is the estimand computed on all 20. Each simulated study draws 20 landscapes with replacement, builds all four intervals from the same B cluster-bootstrap replicates of its own 20, and is scored on whether each interval covers the true value. This is the standard way to check a bootstrap procedure's coverage when the population is unknown: it is the outer level of an iterated (double) bootstrap (Hall 1986; Beran 1987; Davison & Hinkley 1997, ch. 5).

- **percentile**: 2.5% and 97.5% quantiles of the replicate estimates.
- **BCa**: z0 from the share of replicates below the point estimate, acceleration from the leave-one-cluster-out jackknife (the formula in `analyse_boba_robustness._bca_interval`).
- **bootstrap-t**: studentised with the linearisation (delta-method) standard error of a ratio of means, recomputed in every replicate.
- **t**: estimate +/- t(19, 0.975) x delta-method SE (with the n - 1 correction), the standard cluster-robust interval with G - 1 degrees of freedom (Cameron & Miller 2015).

Monte Carlo standard error of a coverage near 95% is 0.22 points at R = 10000. The four methods share each study's replicates, so their coverages are compared pairwise (exact McNemar test on the studies one covers and the other misses).

**Machinery check.** On 4,000 samples of 20 standard normals, where the answer is known, the same code gives t 95.2% (exact: 95.0%), percentile 93.0% (first-order theory: 92.9%), BCa 93.0% and bootstrap-t 95.2%. The percentile and BCa intervals of a mean undercover at n = 20 even for normal data, because they use normal rather than t quantiles and a plug-in (divide-by-n) spread.

**What this cannot detect.** Replacing the population by the empirical distribution measures only the coverage error that comes from having 20 clusters. It cannot detect bias that the empirical distribution itself carries: if the 20 landscapes are unrepresentative of the landscapes the claim is about, every simulated study inherits that. It also treats landscapes as independent clusters. The landscapes share their random seeds by design; positive correlation between landscapes would make every cluster interval too narrow, and this simulation cannot see it. Finally, a study resampled from 20 points holds on average 12.8 distinct landscapes, so the simulated studies are less diverse than real draws from a continuous population; the estimate is first-order accurate for the true coverage, not exact.

## Results

### Selection share of the deployed excess, 1 sigma, onset 0

Paper: 41.8% [34, 50] (percentile, decompose_regret.py). Point estimate here: **41.8%** (8,000 runs over 20 landscapes). On the actual data the BCa constants are z0 = -0.026 and a = -0.0161; the estimator's bias across simulated studies is +0.09 points.

| interval | on the actual data | simulated coverage | interval wholly below / above truth | mean width | actual width |
|---|---|---|---|---|---|
| percentile | [33.6, 50.2] | 92.2% (+/- 0.3) | 2.9% / 4.9% | 16.1 | 16.6 |
| BCa | [33.1, 49.7] | 92.1% (+/- 0.3) | 3.2% / 4.7% | 16.2 | 16.6 |
| bootstrap-t | [32.5, 50.7] | 93.8% (+/- 0.2) | 2.2% / 4.0% | 17.7 | 18.2 |
| t (delta method) | [32.7, 51.0] | 94.7% (+/- 0.2) | 2.0% / 3.4% | 17.6 | 18.3 |

- percentile vs BCa: 63 studies covered only by percentile, 59 only by BCa (McNemar p = 0.786).
- percentile vs bootstrap-t: 10 studies covered only by percentile, 174 only by bootstrap-t (McNemar p = 8.26e-40).
- BCa vs bootstrap-t: 5 studies covered only by BCa, 173 only by bootstrap-t (McNemar p = 7.56e-45).
- Cross-check with `scipy.stats.bootstrap` (independent implementation, B = 20000): percentile [33.6, 50.5], BCa [33.1, 50.0].

### Shortlist m = 3 recovery, main sweep, pooled

Paper: 24% [18, 30] (percentile, replay_hitl_remedies.py). Point estimate here: **24.0%** (12,800 runs over 20 landscapes). On the actual data the BCa constants are z0 = +0.011 and a = +0.0127; the estimator's bias across simulated studies is -0.03 points.

| interval | on the actual data | simulated coverage | interval wholly below / above truth | mean width | actual width |
|---|---|---|---|---|---|
| percentile | [18.0, 29.9] | 92.9% (+/- 0.3) | 4.2% / 3.0% | 11.6 | 12.0 |
| BCa | [18.2, 30.2] | 94.0% (+/- 0.2) | 3.2% / 2.8% | 11.6 | 12.0 |
| bootstrap-t | [17.6, 31.0] | 96.4% (+/- 0.2) | 1.9% / 1.7% | 13.0 | 13.4 |
| t (delta method) | [17.4, 30.6] | 94.6% (+/- 0.2) | 3.2% / 2.2% | 12.7 | 13.2 |

- percentile vs BCa: 10 studies covered only by percentile, 123 only by BCa (McNemar p = 6.74e-26).
- percentile vs bootstrap-t: 0 studies covered only by percentile, 355 only by bootstrap-t (McNemar p = 2.73e-107).
- BCa vs bootstrap-t: 0 studies covered only by BCa, 242 only by bootstrap-t (McNemar p = 2.83e-73).
- Cross-check with `scipy.stats.bootstrap` (independent implementation, B = 20000): percentile [18.0, 30.1], BCa [18.1, 30.3].

### Deployed cost (excess / opt_z), 1 sigma, onset 0

Paper: 29.7%. Point estimate here: **29.7%** (8,000 runs over 20 landscapes). On the actual data the BCa constants are z0 = +0.012 and a = +0.0176; the estimator's bias across simulated studies is +0.00 points.

| interval | on the actual data | simulated coverage | interval wholly below / above truth | mean width | actual width |
|---|---|---|---|---|---|
| percentile | [26.6, 32.9] | 92.7% (+/- 0.3) | 4.5% / 2.8% | 6.1 | 6.3 |
| BCa | [26.7, 33.1] | 93.3% (+/- 0.3) | 3.7% / 3.0% | 6.2 | 6.3 |
| bootstrap-t | [26.5, 33.6] | 95.6% (+/- 0.2) | 2.4% / 2.0% | 6.9 | 7.1 |
| t (delta method) | [26.2, 33.2] | 94.6% (+/- 0.2) | 3.6% / 1.8% | 6.7 | 7.0 |

- percentile vs BCa: 37 studies covered only by percentile, 95 only by BCa (McNemar p = 4.64e-07).
- percentile vs bootstrap-t: 3 studies covered only by percentile, 293 only by bootstrap-t (McNemar p = 6.79e-83).
- BCa vs bootstrap-t: 0 studies covered only by BCa, 232 only by bootstrap-t (McNemar p = 2.9e-70).
- Cross-check with `scipy.stats.bootstrap` (independent implementation, B = 20000): percentile [26.6, 33.0], BCa [26.8, 33.2].

## Recommendation

At nominal 95% and 20 clusters the percentile interval covers 92.2-92.9%, BCa 92.1-94.0%, the studentised bootstrap 93.8-96.4% and the t interval 94.6-94.7%. The method closest to nominal on every estimand is **t (delta method)** (worst miss 0.4 points), at 9-10% more width than the percentile interval.

BCa buys almost nothing here: its constants are small on every estimand (|z0| <= 0.026, |a| <= 0.018), its endpoints on the actual data move by at most 0.5 points from the percentile ones, and it still undercovers. The shortfall of both is the textbook small-sample narrowness of any interval built on normal quantiles and a plug-in spread (the machinery check shows the same 2-point shortfall on normal data), not skew that BCa could correct.

Per estimand, what to print (headline interval first, percentile alongside):

- Selection share of the deployed excess, 1 sigma, onset 0: 41.8% [32.7, 51.0] (t (delta method), simulated coverage 94.7%); percentile [33.6, 50.2] (92.2%).
- Shortlist m = 3 recovery, main sweep, pooled: 24.0% [17.4, 30.6] (t (delta method), simulated coverage 94.6%); percentile [18.0, 29.9] (92.9%).
- Deployed cost (excess / opt_z), 1 sigma, onset 0: 29.7% [26.2, 33.2] (t (delta method), simulated coverage 94.6%); percentile [26.6, 32.9] (92.7%).

## Per-landscape inputs

All in units of the landscape's achievable improvement (opt_z). The deployed-cost column is computed from `cell_means.csv` independently and equals the selection share's denominator to within 8.3e-17.

| landscape | selection loss | deployed excess | shortlist gain | shortlist cost |
|---|---|---|---|---|
| ackley | 0.0278 | 0.2849 | 0.0103 | 0.1998 |
| branin | 0.1680 | 0.2334 | 0.0913 | 0.2251 |
| eggholder | 0.1251 | 0.2118 | 0.0226 | 0.1782 |
| griewank | 0.1495 | 0.3870 | 0.0647 | 0.2915 |
| hartmann_3 | 0.1113 | 0.3057 | 0.0412 | 0.2396 |
| hartmann_6 | 0.0673 | 0.4659 | 0.0301 | 0.2135 |
| hicks_law | 0.0441 | 0.2577 | 0.0421 | 0.1320 |
| levy_10 | 0.1585 | 0.3075 | 0.0961 | 0.3156 |
| michalewicz | 0.0753 | 0.2463 | 0.0007 | 0.1794 |
| moving_peaks | 0.1794 | 0.3599 | 0.0553 | 0.2829 |
| powell | 0.2084 | 0.2462 | 0.1144 | 0.2615 |
| power_law_practice | 0.1940 | 0.4096 | 0.0766 | 0.2364 |
| rastrigin | 0.1610 | 0.3039 | 0.0198 | 0.2840 |
| rosenbrock | 0.1451 | 0.1832 | 0.1265 | 0.2646 |
| schwefel | 0.1124 | 0.2580 | 0.0158 | 0.1807 |
| shekel | 0.0074 | 0.1867 | 0.0040 | 0.1026 |
| steering_law | 0.1710 | 0.2875 | 0.0615 | 0.1759 |
| stevens | 0.0992 | 0.2932 | 0.0655 | 0.1951 |
| weber_fechner | 0.1424 | 0.3411 | 0.0882 | 0.2463 |
| yerkes_dodson | 0.1382 | 0.3702 | 0.0485 | 0.2724 |
