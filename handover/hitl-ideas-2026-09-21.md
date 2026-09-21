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

## Built and running: the five that needed new arms

All five are implemented, smoke-tested, and reach the run filename; the two that
change the clean run also write the clean-run marker. `run_boba_hitl_ideas.ps1`
owns the ordering. Twenty landscapes, seeds 7-11, both onsets, four magnitudes.

| arm | flag | what it does | prediction |
|---|---|---|---|
| anchored rating | `--anchor-rating` | the proposal is judged beside the incumbent, so the error shared by the pair cancels and the fresh part is differenced (sqrt(2) larger) | pays under bias and drift, costs under gaussian |
| self-reported confidence | `--observation-noise self_report` | the rater says how sure they were; the GP takes it as a per-trial variance | should beat the known-noise arm, which recovers nothing, because it is heteroscedastic |
| interleaved anchors | `--anchor-every 5 --anchor-model detrend` | every fifth trial rates a fixed anchor; the anchors identify the rater's drift, which is removed | should fix the block-handover identification failure |
| late re-presentation | `--hold-early 5 --hold-until 0.6` | the first five proposals are rated late instead of early | tests whether the onset effect is about when a design is judged |
| ship-rule acquisition | `--acq-list shiplcb` | values a rating by what it does to the cautious ship rule, not to the posterior maximum | should beat EI on the deployed design and not on the trajectory |

### Notes for whoever picks this up

- The anchored rating and `--observation-noise known` are refused together: the
  variance of a differenced rating is not the injected one.
- `--anchor-every` and `--hold-early` each need their own output directory; the
  marker enforces it.
- The new acquisition is APPENDED to `EXTENSION_ACQUISITION_CHOICES`, at index
  20, because the index seeds the jitter stream.
- `replay_end_of_study` now admits the spike process and takes a `--variants`
  filter, so a directory holding several arms can be replayed one arm at a time.
