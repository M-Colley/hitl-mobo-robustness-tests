# Ten ways to improve the human-in-the-loop process, built and measured

21 September 2026. Ten ideas, each aimed at the term the decomposition says the
cost actually sits in: selection, not search. Five needed no new simulation and
are answered below. Five needed new arms; those are implemented, smoke-tested
and running.

Everything is scored by the standard-process estimand, never against the arm's
own clean twin: `cost = ref_noisy - ref_clean`, `gain = ref_noisy - trt_noisy`,
`price = trt_clean - ref_clean`, aggregated as ratios of landscape means in
units of the achievable improvement, with a landscape bootstrap.

New code: `scripts/replay_hitl_remedies.py`, `scripts/design_rules_from_pilot.py`,
`run_boba_hitl_ideas.ps1`, and five arms in the two simulators.

---

## Answered: the five that needed no new simulation

### 1. Ship a shortlist, not a design — the strongest result of the ten

Ship the top *m* designs by the cautious rule and let a later, cheaper decision
pick among them. Scored as the regret of the best of the *m*. No extra human
trial, only a wider deliverable.

| shipped | main sweep | spike, 15% at 20 SD | capped scale |
|---|---|---|---|
| 1 design (the cautious rule alone) | +0.4% | +19.8% | +29.6% |
| 3 designs | **+24.0%** [18, 30] | +46.7% [35, 58] | +46.5% [40, 54] |
| 5 designs | **+34.1%** [28, 40] | +55.6% [44, 66] | +54.4% [47, 62] |

Price without error: +0.007 to +0.016 of the achievable improvement. On the main
sweep, three designs beat every end-of-study procedure that spends trials, and
they cost none. The whole gain is the shortlist: at m = 1 the same rule recovers
nothing.

### 2. Rank-based inference — free, and it pays exactly where predicted

Refit the surrogate on the ratings' normal scores instead of their values and
ship its argmax. Ranks are invariant to any strictly monotone distortion of the
scale and are robust to one wild rating, so this should pay against spikes and a
saturating cap and cost against clean gaussian noise, where discarding the
magnitudes throws information away. It does both.

| | ordinal, cautious form | the cautious rule it replaces |
|---|---|---|
| main sweep (gaussian, bias, drift, AR(1)) | **-1.1%** [-11, +8] | +0.4% |
| spike, 15% of trials at 20 SD | **+50.5%** [36, 63] | +19.8% |
| capped rating scale | **+36.1%** [29, 43] | +29.6% |

At 15% spikes it recovers about as much as relevance pursuit (+38%) for no
modelling at all, and it needs no assumption about the fault.

### 3. Targeted replication in the final sitting — a clean null

The fixed tournament gives one look to each of the top *k*. The LUCB version
gives each successive look to the empirical leader or to its closest challenger,
so it can look at one pair four times instead of four designs once.

| k = 8 looks | fixed, one each | LUCB |
|---|---|---|
| main sweep | +9.7% [4.8, 14.0] | +9.0% [4.1, 13.4] |
| spike | +59.6% [48, 69] | +54.3% [43, 64] |
| capped scale | +45.9% [39, 52] | +41.3% [35, 48] |

LUCB is never ahead and is behind on both hard arms. The gain from a final
sitting is having looks at all, not where they are put. That is worth saying
because "replicate only where it matters" is the obvious fix for the failure of
blind early replication, and it is not the fix.

### 4. Size the study from a pilot — a clean null

A budget-neutral reallocation: a third of a portfolio of studies gets T = 100
and the rest T = 25, which costs exactly what giving everyone T = 50 costs. Who
should get the large budget?

| sigma_e | flat T = 50 | chosen by the pilot's frag | chosen at random | hindsight best |
|---|---|---|---|---|
| 0.05 | 0.0158 | 0.0102 | 0.0091 | 0.0022 |
| 0.25 | 0.0969 | 0.0738 | 0.0731 | 0.0575 |
| 1 | 0.3022 | 0.2762 | 0.2807 | 0.2313 |
| 5 | 0.7078 | 0.7235 | 0.6909 | 0.6084 |

Deployed cost of error per study, lower is better. The spread allocation beats
the flat one, but a random choice of who gets the large budget does as well as
the pilot-guided one, and at 5 sigma the pilot is worse. The predictor does not
guide. What the table does show is that the cost is convex in the budget, so an
unequal allocation is better than an equal one however it is chosen.

### 5. Screen the instrument before the study — it works

A saturating rating scale is among the most expensive faults measured. The
screen is the headroom above the cap: how far the reachable optimum sits above
the instrument's ceiling, which a pilot can estimate.

| cap | median extra cost | rho(headroom, extra cost) | low headroom | high headroom |
|---|---|---|---|---|
| fixed at the 0.9 quantile | +0.219 | **+0.65** | +0.153 | +0.316 |
| re-anchored to the best so far | +0.104 | +0.31 | +0.083 | +0.170 |

The damage doubles from the low-headroom half to the high-headroom half, and the
pilot can tell which half a study is in. Re-anchoring the cap halves the median
damage, which is the cheapest remedy in the whole project.

---

## Answered: the five that needed new arms

Twenty landscapes, seeds 7-11, both onsets, four magnitudes, LogEI and qNEI,
12,915 runs. Each prediction below was written before the sweep, in the commit
that built the arm. One holds, two are nulls, two fail.

| arm | trajectory | deployed | price | prediction |
|---|---|---|---|---|
| self-reported confidence | +16% [-4, +36] | **+15%** [+6, +23] | -0.001 | held |
| interleaved anchors, gaussian | -6% [-23, +14] | +1% [-8, +10] | +0.007 | failed |
| interleaved anchors, drift | -9% [-31, +11] | -6% [-15, +1] | +0.007 | failed |
| late re-presentation | -13% [-44, +14] | +1% [-11, +11] | +0.002 | no effect |
| anchored rating, gaussian | -18% [-25, -12] | -27% [-36, -19] | +0.000 | held |
| anchored rating, bias | -16% [-21, -10] | -15% [-26, -5] | +0.000 | failed |
| anchored rating, drift | -11% [-21, -3] | -17% [-30, -6] | +0.000 | failed |
| ship-rule acquisition | -287% [-366, -219] | -107% [-159, -64] | +0.244 | failed |

### The one that works: ask the rater how sure they were

A self-reported precision, used as a per-trial observation variance, recovers
**15% [6, 23]** of the deployed cost of error at a price of essentially zero.
That is the arm to keep, and it is worth setting beside the known-noise arm,
which hands the GP the TRUE noise variance and recovers nothing (-3%). The
difference is not accuracy, it is kind: a constant is what the GP already
learns, and a per-trial report is heteroscedastic, saying which trials were
hard. The report here is coarse on purpose, a log-normal multiplier of SD 0.5
on the trial's true squared error, so this is not the ceiling of the idea.

### The two nulls

**Interleaved anchors do not pay.** Every fifth trial rating a fixed anchor
buys +1% [-8, +10] under gaussian error and -6% [-15, +1] under drift, at a
price of 0.007. The prediction was that they would fix the identification
failure behind the block-handover result, and they do not: the trials they
cost are worth more than the drift estimate they buy. A linear detrend on a
dozen anchor readings is a weak instrument, and the drift the sweep injects is
small next to the fresh noise on top of it.

**Late re-presentation does nothing.** Holding the first five proposals back
and rating them late moves the deployed cost by +1% [-11, +11]. So the onset
effect is not about when a design is judged; something else about early error
makes it expensive, and this arm rules out one explanation of it.

### The two failures, and what they say

**The anchored rating loses everywhere, including where it should have won.**
Judging the proposal beside the incumbent cancels the error the pair shares
and differences the fresh part, so the idiosyncratic noise grows by sqrt(2).
The prediction was that this would pay under bias and drift and cost under
gaussian. It costs under all three: -27% deployed under gaussian, -15% under
bias, -17% under drift. The reason is visible in hindsight. A constant offset
was never expensive, because the loop standardises its targets and an argmax
is shift-invariant; the drift ramp is partly absorbed by the GP's mean. So the
arm pays sqrt(2) more noise for cancelling something that was nearly free.
This is the sharpest negative result of the ten: a comparison is worth having
only against a fault the rating scale cannot absorb, which is what the
elicitation appendix's narrow reach already suggested.

**The ship-rule acquisition is a bad optimiser, so the idea is untested.**
It loses 107% of the deployed cost, but the number that matters is its price
WITHOUT error: +0.244 of the achievable improvement, by far the largest price
in the project. An acquisition that is this much worse on a clean run has not
been tested as a remedy for noise; it has been shown to search badly. The
one-step lookahead on the lower bound is too conservative: penalising a
candidate for the uncertainty it will still have after one rating suppresses
exploration almost entirely. The idea, valuing a rating by what it does to the
decision rather than to the posterior maximum, is not refuted by this. A
version that keeps the ship-rule threshold but not the post-rating penalty
would be the next thing to try.

---

## The ten, in one table

| # | idea | verdict | deployed recovery |
|---|---|---|---|
| 1 | ship a shortlist of 3 to 5 | **works** | +24% to +34% |
| 2 | rank-based inference, against outliers | **works** | +50% spike, +36% cap |
| 3 | screen the instrument for a ceiling | **works** | halves the cap's damage |
| 4 | ask the rater how sure they were | **works** | +15% |
| 5 | LUCB allocation of the final sitting | null | -0.7pp against one look each |
| 6 | size the study from a pilot | null | no better than random |
| 7 | interleaved anchors | null | +1% |
| 8 | late re-presentation | null | +1% |
| 9 | anchored rating | fails | -27% |
| 10 | ship-rule acquisition | untested, bad optimiser | -107%, price +0.24 |

Four work, four are nulls, one fails, one is not a fair test of its own idea.
The four that work share a shape: none of them changes the search. Three change
what the study deploys or claims, and the fourth changes what the rater is
asked. That is the same grain as everything else in this project.
