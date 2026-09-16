# Preregistered confirmatory hypotheses

**Written 2026-09-09, before any run on seeds 27–36 existed.** Everything below
was selected from the screening sweep (seeds 7–16, `output-boba/`), which is
exactly why it cannot also be tested there. The confirmatory arm re-runs the
same design on ten fresh seeds and tests only the claims listed here, with the
tests and thresholds fixed in advance.

## Design of the confirmatory run

Identical to the screening design except for the seeds and the restriction to
the gaussian error model:

- 20 benchmarks (`--functions all`), 12 acquisitions (`--acq all`)
- 50 iterations, 5 initial samples
- gaussian error at 0.05, 0.25, 1.0, 5.0 landscape SDs
- onsets 0 and 20
- **seeds 27–36**, which appear in no screening analysis

The gaussian restriction is pre-specified: screening found the error process
changes the cost by only 1.32×, so it is not one of the claims worth confirming,
and dropping it buys the seeds.

## H1 — Mediation (primary)

*Screening estimate:* regressing post-onset excess regret on the one-shot
selection loss and the error magnitude jointly, β(frag) = +0.66 [0.28, 0.74],
β(log σₑ) = −0.07 with p = 0.44; R² rises from 0.53 (descriptors alone) to 0.70.

**H1a.** β(frag_at_c) > 0 with a bootstrap interval excluding zero.
**H1b.** β(log σₑ) is not distinguishable from zero: its 95% interval contains 0.
**H1c.** The joint model's R² exceeds the descriptors-only model's by ≥ 0.10.

Confirmed only if all three hold. H1b is a null and is stated as such
deliberately: the claim is that the magnitude adds nothing once the mediator is
in, and a confirmatory run that merely fails to reject is weak evidence. It is
reported with the interval, not only the p-value, so that a wide interval reads
as uninformative rather than as support.

## H2 — Onset (primary)

*Screening estimate:* at 1σ the mean fraction of achievable gain lost is 0.1409
at onset 0 and 0.0281 at onset 20, a ratio of 5.0.

**H2.** At 1σ, mean fragility at onset 0 exceeds that at onset 20 by a factor
≥ 3, and the paired difference over the 20 benchmarks is significant by a
two-sided Wilcoxon signed-rank test at α = 0.05.

## H3 — Acquisition families (secondary)

*Screening estimate:* mean rank qUCB 3.95, qNEI 4.34, UCB 4.41 against PI 7.66,
LogPI 7.48; Kendall's W = 0.247.

**H3a.** The mean rank of {qUCB, qNEI, UCB} is lower (more robust) than that of
{PI, LogPI, qPI}, with a two-sided Wilcoxon over the 20 benchmarks significant
at α = 0.05 after Benjamini–Hochberg correction across H3a and H3b.
**H3b.** Kendall's W over the 20 benchmarks, averaged over conditions, is
< 0.40 — that is, the landscapes continue to disagree substantially about which
acquisition wins.

## H4 — Dose–response (secondary)

**H4.** Mean fragility is monotone increasing in the error magnitude at both
onsets, across all four levels.

## Controls that must also hold

These are not hypotheses; they are conditions for the run being interpretable at
all. If any fails, the confirmatory result is void rather than negative.

- Model-free (`random`, `sobol`) excess regret is exactly 0.
- The seed panel is balanced: every cell carries all ten fresh seeds.
- No benchmark has headroom below 0.10.
- Acquisition-optimisation fallbacks (`acq_opt_failed`) total zero.

## What will be reported

The screening and confirmatory estimates side by side for every claim, whether
or not they agree. A claim that fails to confirm is reported as failing to
confirm; nothing here is re-specified after the fresh seeds are seen, and no
claim not listed above is tested on them.
