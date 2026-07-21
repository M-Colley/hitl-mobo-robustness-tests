# Adaptive elicitation meta-policy via offline RL (sandbox prototype)

Prototype instantiation of Buçinca et al. (2026), *"Offline Reinforcement
Learning for Adaptive Support in AI-Assisted Decision-Making"* (TOCHI), on top
of this repo's oracle+noise simulator. Where the paper adapts the *assistance
type* shown to a human decision-maker, this prototype adapts the *elicitation
action* taken per iteration of a human-in-the-loop BO session.

## Correspondence to the paper

| Paper | This prototype |
|---|---|
| Participant answering 33 questions | Simulated session: 20 oracle queries (3 initial + 17 decisions), matching the eHMI study's ~20 evaluations/participant |
| 4 assistance types (no AI / explanation / SXAI / on-demand) | 4 meta-actions: `ei` (standard BO step), `exploit` (posterior-mean argmax), `replicate` (re-query the current recommendation), `probe` (random design) |
| State: NFC, concept, AI correctness, running accuracy, initial knowledge (64 states) | State: phase (early/late/endgame), surprise (recent \|obs − GP pred\| vs observed std), stall (recommendation improvement) — 12 states, observable quantities only |
| Reward: immediate accuracy / distal learning | Reward: per-step change in the *true* value of the recommended design (γ=1 over the finite horizon, so the return telescopes exactly to final inference quality — the repo's deployment metric) |
| Quasi-uniform exploratory policy, N=142 | Uniform-random behavior policy, 150 episodes × 4 conditions |
| Tabular Q-learning, greedy policy, coverage analysis | Same (12×4 table, unseen-action masking in both bootstrap and greedy extraction, coverage report) |
| Evaluation studies on fresh participants vs SXAI/random baselines | Fresh disjoint seed block; baselines: `always_ei` (the fixed-policy analogue of SXAI), `uniform_meta` (their random policy), `always_probe` (random floor) |

## Conditions

gaussian and AR(1) (ρ=0.8) error, jitter_std ∈ {0.3, 1.0} — roughly 0.4× and
1.4× the composite objective's SD (0.72), i.e. plausible and heavy human error.
Noise applies from the first observation (the repo's `jitter_iteration=0`
human-plausible condition). One policy is trained *pooled* across all
conditions: the noise level is not in the state, so the policy must infer the
regime from the surprise feature — that inference is the point of the exercise.

## Run

```bash
python sandbox/meta_policy_rl/meta_policy_experiment.py          # ~30 min
python sandbox/meta_policy_rl/meta_policy_experiment.py --quick  # smoke test
```

Outputs in `sandbox/meta_policy_rl/output/`:

- `learned_policy.csv` — Q-table, per-state chosen action, per-cell coverage
- `eval_results.csv` — per-episode final inference regret for every policy
- `summary.txt` — per-condition means ± SE and paired q_policy-vs-always_ei tests
- `results.png` — bar chart per condition
- `run_metadata.json`

## Results snapshot — v2, properly powered (2026-07-16)

v2 run (`output/`): 6 actions (adds `drop_worst`, `rest`), per-regime AND
pooled Q-tables, latent-σ EI baseline, n=200 paired eval seeds per condition
(~80% power for dz=0.2), BH-corrected primaries, 30,600 training transitions,
~9 min on 14 workers. Paired mean difference q_regime − always_ei on final
inference regret (negative favors the adaptive policy; BH-adjusted t p):

| jitter_std | gaussian | ar1 (ρ=0.8) |
|---|---|---|
| 0.3 | +0.023 (dz=+0.08, p=.37) | **+0.059 (dz=+0.26, p=.002, EI wins)** |
| 1.0 | +0.006 (dz=+0.02, p=.89) | **+0.075 (dz=+0.24, p=.003, EI wins)** |
| 2.5 | −0.004 (dz=−0.01, p=.89) | **+0.089 (dz=+0.23, p=.003, EI wins)** |

Headline: **a clean, well-powered null-to-negative result.** With a correctly
calibrated static baseline, the adaptive policy never beats always-EI, and the
per-regime variant is significantly *worse* under AR(1). v1's apparent win at
heavy gaussian noise (dz=−0.20 at n=40) did not replicate at n=200 (dz=−0.01)
— it was sampling noise, a weaker v1 baseline (EI σ included observation
noise), or both.

Why, mechanically:

- The exploratory log shows per-action mean rewards of only ±0.01–0.03 with
  large variance; EI has the best mean in 5/6 conditions. There is simply not
  enough state-conditional action-effect signal for a tabular Q-table to beat
  "always take the best average action".
- Per-regime specialization (the paper's per-NFC analogue) *hurts*: 5,100
  transitions per table vs 30,600 pooled means noisier Q estimates, and the
  degenerate choices show it — the ar1/2.5 regime policy learned to `rest` 14
  of 17 decisions (rest's reward is exactly 0 with zero variance, so it wins
  noisy argmaxes over small-positive-mean, high-variance actions). Bias–
  variance: with weak effects, variance dominates and pooling wins.
- Mirrors Bucinca et al.'s own hard case: their accuracy objective worked
  because assistance types had large, estimable effects (people copy shown
  recommendations); their learning objective — like our setting — lacked a
  dominant action and failed to robustly beat baselines.

Implication for the repo's research question: adaptive elicitation via
offline RL is **not** the low-hanging fruit here; the leverage is in better
static machinery (robust likelihoods, replication arms, noise-aware
surrogates). Note: v2 overwrote v1's main outputs in `output/`; only the v1
heavy-noise follow-up (4-action version) remains in `output_heavy/` and is
superseded by this run.

## Deliberate simplifications (read before trusting numbers)

1. **sklearn GP + candidate-pool acquisition** instead of BoTorch, for ~100×
   speed. Pool screening mirrors the main simulator's own candidate-pool
   approach, but absolute regret values are not comparable with the main sweep.
2. **Oracle fidelity ceiling**: the extra_trees oracle is fit on raw eHMI rows;
   policies learned in this simulator inherit its flaws. The paper itself makes
   this argument *against* simulator-trained policies — results here are
   hypothesis-generating for a real exploratory-policy study, not confirmatory.
3. **Paired noise streams**: within an eval seed, all policies share the same
   initial designs and the same per-step noise draws; only their design choices
   differ.
4. AR(1) noise starts from its stationary distribution (this deliberately fixes
   the cold-start issue the code review found in the main simulator).
