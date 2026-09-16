# Accounting for human and sensor error in HITL (MO)BO: what the numbers say, and what to change

*2026-09-12, revised 2026-09-13 after the adaptation experiments ran. Every number below is from this
repository's runs; the paper (`paper/main.tex`, Appendices H and I) and
`docs/known-function-arm-2026-09-06.md` (§7.10, §7.13) give the sources. "Measured" means we ran it;
"proposed" means we did not.*

## 1. The question, and the short answer

People and sensors make mistakes. How much do those mistakes cost a human-in-the-loop optimization,
how many extra (expensive) trials would it take to absorb them, and what should change in the process
so that we do not have to pay for them in trials?

**Short answer.** At the rater-noise levels measured in three real studies (0.74–3.35 landscape SDs),
feedback error costs 12–20% of the improvement a clean study would have found. That loss cannot be
bought back with more trials: at one landscape SD of noise from the first rating, the median run needs
41 extra trials to match a clean 25-trial study (a budget of ×2.6), and 39% of runs never match it
within 100 trials. Only small noise (≤0.25σ) is repaired by budget (+1 to +7 trials).

So the lever would have to be the method or the process, and we tested the obvious changes against
the standard process (§4). **Most do not help.** Rating the first ten designs twice, re-rating the
incumbent, qKG and a noisy-input GP all do worse under error than doing nothing. Replication gives up designs that are worth more in a
fifty-trial study than the averaging buys; why qKG and the noisy-input GP lose is not yet tested. Telling the GP its true noise level recovers nothing, and neither does a heavy-tailed (Student-t)
likelihood for misclicks. Two changes help:

- the **observed-maximum incumbent** recovers a fifth of the cost along the way (19%) and cuts the
  extra trials one-SD error needs to match a clean ten-trial study from 8 to 6, but the design it
  deploys is worse (by 10% of the cost, and by about a third with late error at 1σ and above);
- **re-rating the top three designs in the last six trials** recovers 4% of the deployed design's
  cost, and 18% when large error arrives late.

For input errors, recording what was actually applied is still the largest measured gain, and the
damage can be forecast before the study: frag computed from a ten-trial pilot predicts the cost as
well as the exact version. The practical consequence is to plan for the loss (§3.1), protect the
record (Q1) and reduce error at the source (Q2, Q3), rather than expect a surrogate or acquisition
change to remove it.

## 2. What was measured

### 2.1 Cost in regret

| | 0.05σ | 0.25σ | 1σ | 5σ |
|---|---|---|---|---|
| error from trial 1 | 0.9% | 5.3% | 14.1% | 22.0% |
| error from trial 21 (of 50) | 0.3% | 1.2% | 2.8% | 4.9% |

Fraction of the achievable improvement destroyed, 20 landscapes × 10 model-based acquisitions × 4 error
processes × 10 seeds. The kind of error (gaussian, bias, drift, AR(1)) changes this by 1.3×; the
magnitude by 23×, the landscape by 8×, the onset by 5×, the acquisition by 2.8×.

### 2.2 Cost in trials (the budget question)

Extra trials a noisy run needs to come within 1% of the achievable improvement of its clean twin's
result (budget-100 arm, gaussian error, six acquisitions; median over runs; "never" = not within 100
trials):

| σ_e | match a clean 25-trial study, error from trial 1 | match a clean 50-trial study, error from trial 41 |
|---|---|---|
| 0.05σ | +1 (never 4%) | 0 (never 1%) |
| 0.25σ | +7 (never 14%) | 0 (never 3%) |
| 1σ | +41 (never 39%) | 0 (never 8%) |
| 5σ | >78 (never 70%) | 0 (never 12%) |

Rule of thumb for planning: **≤0.25σ, budget ×1.05–1.3; ≥1σ, budgeting does not work.** The same
count on the input-error arms (match a clean 10-trial study within 50): a 5% slip of each coordinate's
range costs a median of 3 extra trials (never 7%), a 15% slip 8 (never 15%), a 40% slip 24 (never
40%); a misclick on 15% of trials costs 0 (never 9%), on 40% of trials 6 (never 23%). Logging the
actual design roughly halves the slip numbers (2 / 4 / 14).

### 2.3 What drives the cost, and what does not

- **The loss is informational, not a mis-specified noise model.** Supplying the GP with the true
  observation variance recovers nothing: −3% of the cost on both responses when scored against the
  standard process (§4), although the fitted noise is an eighth of the truth for the first ten
  iterations.
- **The incumbent is a cheap lever, for the trajectory.** Switching improvement-based acquisitions
  from the posterior-mean incumbent to the observed maximum recovers 19% [10, 27] of the trajectory
  cost against the standard process, most of it with error from the first rating, but deploys a
  design worse by 10% [5, 15] of the cost.
- **Acquisition family matters 2.8×**: UCB, qUCB and qNEI are the most robust; PI, LogPI, qPI and
  Greedy fill the bottom four places. Which one wins on a given landscape is mostly a property of the
  landscape (median Kendall's W = 0.25 across conditions).
- **Methods built for noise do not recover the cost.** An earlier version of this document said qKG
  and replication recover a third. That compared each method with its own clean run, which a
  replicating method handicaps. Against the standard process both do worse under error (§4).
- **Timing matters 5×.** Error arriving after trial 20 of 50 costs a fifth of error from trial 1; the
  ratio grows with the budget (2.3× / 5.1× / 11.7× at T = 25 / 50 / 100).
- **The rating scale cuts both ways.** Rounding to a 0.55σ grid is itself a 0.16σ error and makes
  small noise six times worse; clipping bounds gross ratings and helps marginally at 5σ.
- **Input error.** A 15% slip on every trial costs about what 1σ rating noise costs; half of a small
  slip's cost is the mislabelled record rather than the displacement; the design the log recommends is
  2.5× worse than the design the person actually experienced; a misclick is cheap below 40% of the
  trials.
- **A one-shot statistic predicts the cost.** frag(σ_e), the expected loss from one greedy pick among
  256 quasi-random designs observed with N(0, σ_e²) error, explains half the variation in cost on its
  own (R² = 0.48) and stays informative beside every landscape descriptor.
- **Human-oracle cross-check.** The same experiment on three fitted-to-ratings oracles shows the same
  structure at about a third of the magnitude; improving the oracles by up to 0.29 held-out R² does
  not change it; the direction of that structure, not its magnitude, is confirmed on the human arm.
  Early replication on those oracles points the same way as on the landscapes, though not
  significantly (−8% of the deployed design's cost, interval [−17, 6], three datasets).

## 3. Adaptations, with verdicts

Ordered by where they sit in the process. Verdicts: **works** (recovers cost, interval excludes
zero), **small**, **mixed**, **hurts** (does worse under error than the standard process),
**no effect**, **untested**.

### 3.1 Before the study: plan against the measured noise

**P1. Measure rater noise in the pilot and place it on the dose table.** *Measured.* A pilot with
repeated ratings of the same design gives σ_e by the nearest-neighbour (nugget) estimator used in §4
of the paper; dividing by the spread of a pilot surrogate over the design box puts it in landscape
units. Table 2.1 then gives the expected loss and Table 2.2 says whether extra trials will repair it.
The three real studies land at 0.74–3.35σ, the regime where budgeting does not work.

**P2. Forecast with frag(σ_e) from a pilot surrogate.** *Works (E7).* frag computed on the posterior
mean of the study's own GP after its first ten clean trials correlates 0.94 with the exact frag across
landscapes, and 0.85 with the measured cost (the exact frag: 0.84). Within a single magnitude neither
ranks landscapes well, so it says how damaging a noise level is, not which of two similar problems is
worse. At 5σ a ten-point pilot underestimates frag by about a third on average.

**P3. Choose the instrument for the noise level.** *Measured.* At ≤0.25σ the discretisation of a
7-point composite scale dominates the loss (3.8% against 0.6%), so use a fine-grained or continuous
response; at ≥1σ clipping helps marginally, so keep bounds.

### 3.2 The surrogate

**S1. Use the observed maximum as the incumbent for improvement-based acquisitions.** *Works on the
trajectory, hurts the deployed design.* Against the standard process it recovers 19% [10, 27] of the
trajectory cost at a price of 0.1 points without error: +67% at 0.05σ, +47% at 0.25σ, +24% at 1σ and
+10% at 5σ with error from trial 1. It cuts the extra trials needed to match a clean standard 10-trial
study from 8 (never 21%) to 6 (13%) at 1σ, and from 36 (48%) to 21 (39%) at 5σ. But the design it
deploys is worse by 10% [5, 15] of the cost, mostly when error arrives late (−27% at 1σ, −36% at 5σ);
the reason is not isolated. Use it when the study is judged by how fast it finds good designs or may
stop early. If only the final recommendation matters, the standard incumbent deploys a better design;
combining S1 with A4 is untested.

**S2. Do not spend effort on the observation-noise hyperparameter.** *No effect.* The true variance
recovers −3% [−16, 8] of the trajectory cost and −3% [−11, 5] of the deployed design's.

**S3. A noisy-input GP for slips.** *Mixed, net harmful (E3).* A slip is input noise, x' = x + δ; the
first-order remedy inflates each observation's variance by the squared gradient of the posterior mean
times the slip variance (McHutchon & Rasmussen, 2011). With error from the first trial it recovers 76%
[27, 245] of a 1% slip's trajectory cost and 17% of a 5% slip's. With error from trial 21 it recovers −129% of a 15% slip's cost and −205% of a 40% slip's. A
likely but untested reason: the search then sits near a steep peak whose flanks have the largest
gradients, so the observations that locate the peak are discounted most. Pooled, −28% [−43, −13] of the trajectory cost and −19% [−31, −9] of the
deployed design's. Do not use it as a default; logging the applied design (Q1) is the better fix.

**S4. A heavy-tailed likelihood for misclicks and gross sensor faults.** *No effect (E2).* A misclick
records the rating of a random design against the proposed one, an outlier rather than noise, so a
Student-t likelihood (Martinez-Cantin et al., 2018) should discount it. A variational Student-t GP in
place of the standard one, under misclicks on 1–40% of trials, recovers −2% [−38, 31] of the
trajectory cost and −13% [−50, 16] of the deployed design's, at a price of 0.2 and 1.5 points. Where
misclicks cost most, on 40% of trials from the first, it recovers −1% [−10, 6] of the trajectory
cost, and the median run still needs 8 extra trials to match a clean standard ten-trial study (never
24% against 28%). Misclicks cost little to begin with. Gross sensor faults, which can be far larger,
were not simulated, so for them the question is open.

**S5. Bias and drift do not need their own model.** *Measured.* Systematic bias costs the same as
random error of the same size (and is invisible to a standardising GP when present from the start);
drift costs 1.03× gaussian. A change-point or drift term would gain at most the 20% by which a
late-arriving bias exceeds gaussian noise. Low priority, untested.

### 3.3 The acquisition and the trial schedule

**A1. Prefer UCB / qUCB / qNEI; do not use PI, LogPI, qPI or Greedy under noise.** *Measured.* 2.8×
between the best and worst family at 1σ; the bottom four are the PI family and Greedy.

**A2. A noise-aware acquisition when the noise is ≥1σ.** *Hurts; the earlier recommendation is
withdrawn.* Against the standard process (the mean over its ten acquisitions), qKG recovers −4% [−23,
14] of the trajectory cost and −32% [−42, −24] of the deployed design's, at a price of 2.7 and 3.6
points without error. It helps only with error from the first trial at ≥0.25σ (+43% at 0.25σ, +12% at
1σ, +23% at 5σ) and hurts with late error. Re-rating the incumbent every second trial recovers −61%
[−88, −33] and −5% [−12, 3], at a price of 5.4 and 4.3 points.

**A3. Concentrate replication in the first trials.** *Hurts (E4, E1).* Rating each of the first ten
proposals twice recovers −88% [−113, −67] of the trajectory cost and −14% [−21, −8] of the deployed
design's, at a price of 5.1 and 1.6 points, and needs more extra trials than the standard process at
1σ (a median of 8 against 6). Averaging two ratings only halves the error variance, the GP already
pools neighbouring observations, and the ten designs given up cost more. Adding the observed-max
incumbent does not rescue it (−93% and −44%). Early replication is worse than uniform replication on
the trajectory (−88% against −61%) and on the deployed design (−14% against −5%), though the two replicate different numbers of trials and
are scored against different references (LogEI and qNEI, against the mean of ten acquisitions);
in raw regret the ordering is the same.

**A4. Re-evaluate before you deploy.** *Small (E5).* Spending the last six trials re-rating the three
designs with the best mean rating, and deploying by mean, costs 1% of the trajectory cost and recovers 4% [0.04, 9] of the deployed design's, reaching 18% [5, 29] when 5σ error arrives late. Under an
unnoticed slip it recovers nothing (−6% [−15, 1]) and hurts with large late slips (−36% at 15%, −74%
at 40%): the re-ratings land on displaced points too. Cheap insurance against rating noise; no help
with slips unless the applied design is logged.

### 3.4 The interface and the process

**Q1. Record what was applied, not what was proposed.** *Measured.* Half of a small slip's cost (2.7
of 5.4 points at 5%; 4.4 of 11.5 at 15%) is the mislabelled record. Any instrumented interface, sensor
readback or confirmation step that logs the actual configuration removes it at no trial cost. This is
the single largest measured gain in the input-error arm, and larger than any model-side change above.

**Q2. Make misclicks rare, not impossible.** *Measured.* A misclick on 5% of trials costs 2.8%, on 15%
6.8%. A confirmation step for out-of-range or unusual settings, or showing the applied configuration
before the rating, keeps the rate in the cheap regime.

**Q3. Reduce error at the source in the early window, without spending trials.** *Proposed.* Error
from trial 1 costs 5× error from trial 21, but A3 shows that buying precision with repeat trials does
not pay. What remains is precision that costs no designs: clearer instructions, practice trials before
the study starts, attention checks, longer exposure, and several raters rating the same trial in
parallel. The last divides σ_e by √n, so Table 2.1 gives its value (from 1σ to 0.5σ with four raters).
Untested as a process.

**Q4. Detect onset.** *Proposed.* A repeated anchor stimulus every k trials gives a running estimate of
rater consistency; a jump flags fatigue, drift or a changed rater. The value is bounded by the
late-onset column of Table 2.1 (≤5% at 5σ), so this is a safeguard, not a main lever.

### 3.5 A planning card

| pilot σ_e (landscape SD) | expected loss, error from trial 1 | budget fix? | do this | avoid |
|---|---|---|---|---|
| ≤0.25 | ≤5% | yes, ×1.05–1.3 | fine-grained scale (P3); observed-max incumbent, which recovers about half (S1) | replication (A3) |
| ≈1 | ≈14% | no (39% never recover) | UCB/qNEI (A1); observed-max incumbent if the trajectory matters (S1); re-rate the top three before deploying (A4); reduce σ_e at the source (Q3) | early or uniform replication, qKG, a known noise level (A2, A3, S2) |
| ≥3 | ≥20% | no | as above, where A4 matters most for late error; forecast the loss with a pilot frag (P2) and ask whether the study can answer its question at all | as above |
| any, with a physical interface or sensors | + slip/misclick cost | no | log the applied design (Q1); confirm unusual settings (Q2) | a noisy-input GP as a default (S3); re-rating to fix slips (A4); a Student-t likelihood for misclicks (S4) |

## 4. How the adaptations were scored, and what they found

A process change can alter the clean run too. A repeated rating of an exact objective carries no
information, so a replicating method's clean run is handicapped, and its excess over its own clean
run makes it look better than it is. Each change is therefore scored against the **standard**
process. Per paired cell (landscape, acquisition, magnitude, onset, seed), with regret as a fraction
of the achievable improvement:

    recovered = (R_std,noisy − R_adapt,noisy) / (R_std,noisy − R_std,clean)
    price     =  R_adapt,clean − R_std,clean

Both are aggregated as ratios of landscape means, with 95% landscape-bootstrap intervals, on two
responses: the post-onset trajectory regret and the regret of the design the experimenter would
deploy. Where the change is a different acquisition (qKG, replication), the standard process is the
mean over its ten acquisitions. Every arm runs 20 landscapes, seeds 7–11, onsets at trials 1 and 21,
four magnitudes, with LogEI and qNEI unless stated; 50 trials. Scripts:
`scripts/analyse_boba_adaptations.py` (outputs `output-boba/analysis/adaptations_recovery.csv` and
`adaptations_extra_runs.csv`); runners `run_boba_adapt*.ps1`, `run_fitted_adapt.ps1`.

| id | change | error | recovered, trajectory | recovered, deployed | price (trajectory / deployed, points) | verdict |
|---|---|---|---|---|---|---|
| — | observed-max incumbent (re-score) | gaussian | +19% [10, 27] | −10% [−15, −5] | 0.1 / 0.2 | trajectory only |
| — | GP given the true noise variance (re-score) | gaussian | −3% [−16, 8] | −3% [−11, 5] | 0.4 / 0.0 | no effect |
| E4 | first ten proposals rated twice | gaussian | −88% [−113, −67] | −14% [−21, −8] | 5.1 / 1.6 | hurts |
| E1 | … plus observed-max incumbent (LogEI) | gaussian | −93% [−127, −58] | −44% [−57, −32] | 4.7 / 1.1 | hurts |
| E5 | last six trials re-rate the top three | gaussian | −1% [−2, −1] | +4% [0.04, 9] | 0.1 / 0.8 | small |
| — | knowledge gradient (qKG) | gaussian | −4% [−23, 14] | −32% [−42, −24] | 2.7 / 3.6 | hurts |
| — | re-rate the incumbent every second trial | gaussian | −61% [−88, −33] | −5% [−12, 3] | 5.4 / 4.3 | hurts |
| E3 | noisy-input GP | unnoticed slip | −28% [−43, −13] | −19% [−31, −9] | −0.1 / −1.5 | mixed, net harmful |
| E5 | last six trials re-rate the top three | unnoticed slip | 0% [−0.3, 0.2] | −6% [−15, 1] | 0.1 / 0.8 | no effect |
| E2 | Student-t likelihood | misclick | −2% [−38, 31] | −13% [−50, 16] | 0.2 / 1.5 | no effect |
| E6 | first ten rated twice, fitted oracles | gaussian | −161% [−3504, −49] | −8% [−17, 6] | — | same direction |
| E7 | frag from a ten-trial pilot | — | Spearman 0.94 with exact frag; 0.85 with measured cost (exact: 0.84) | | | works |

**Why most changes fail.** Replication spends trials on precision. In a fifty-trial study a
design is worth more than halving one observation's variance, and the GP already pools neighbouring
observations. qKG does not replicate; why it loses with late error and on the deployed design is
untested. The noisy-input GP discounts observations in proportion to the local gradient, which is
largest on the flanks of a steep peak, plausibly the observations that locate it. None of the changes exploit the onset effect, the largest
structure in the data, without paying for it in designs.

**Still untested.** BO under input uncertainty for slips (unscented BO, Nogueira et al., 2016;
Oliveira et al., 2019; noisy-input entropy search, Fröhlich et al., 2020); a noisy-input GP that stops
inflating once the search is local; several raters per trial (Q3); S1 combined with A4; why S1 worsens
the deployed design; and whether replication pays at budgets well above fifty trials, where the onset
ratio is larger.

## 5. What this does not cover

Real raters were modelled by four error processes and two input-error processes, all stationary in
their parameters; learning effects, rater-specific scales and multi-rater disagreement were not run.
The extra-trials counts are censored at the budget and depend on the 1% tolerance. The adaptations
were run at 50 trials and five seeds; a change that fails at 50 trials may pay at larger budgets.
Every method recommendation is a within-arm comparison on twenty analytic landscapes; the human-oracle
arm tested only early replication, on three datasets.

## References to verify before citing

McHutchon & Rasmussen (2011), Gaussian process training with input noise, NeurIPS. Nogueira,
Martinez-Cantin, Bernardino & Jamone (2016), Unscented Bayesian optimization for safe robot grasping,
IROS. Beland & Nair (2017), Bayesian optimization under uncertainty, NeurIPS workshop. Oliveira, Ott &
Ramos (2019), Bayesian optimisation under uncertain inputs, AISTATS. Fröhlich, Klenske, Vinogradska,
Daniel & Zeilinger (2020), Noisy-input entropy search for efficient robust Bayesian optimization,
AISTATS. Martinez-Cantin, Tee & McCourt (2018), Practical Bayesian optimization in the presence of
outliers, AISTATS. Binois, Huang, Gramacy & Ludkovski (2019), Replication or exploration? Sequential
design for stochastic simulation experiments, Technometrics. Kirschner, Bogunovic, Jegelka & Krause
(2020), Distributionally robust Bayesian optimization, AISTATS. Letham, Karrer, Ottoni & Bakshy (2019),
Constrained Bayesian optimization with noisy experiments, Bayesian Analysis. Frazier, Powell & Dayanik
(2009), The knowledge-gradient policy for correlated normal beliefs, INFORMS J. Computing.
