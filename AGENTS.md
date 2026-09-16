# Working in this repository

This file is for whoever picks the work up next, human or agent. `README.md`
explains what the project is; this explains how not to break it, and where the
traps are. Everything here was learned by hitting it.

## What the project claims, in one paragraph

Bayesian optimization driven by human ratings is measured on twenty analytic
landscapes with published optima, with four error processes injected at four
magnitudes and two onsets. The headline is not "error is expensive" but *where*
the expense is: the deployed design's regret splits into **search loss** (the run
never found anything better) and **selection loss** (it found something better and
shipped the wrong one), and selection is ~62% of what the error costs. Selection
is identically zero under exact observation, which is why a noiseless benchmark
cannot see it, and it is the term an acquisition function cannot reach. Remedies
are judged against that: acquisition-side changes recover nothing on the deployed
design, while spending trials on identification or modelling the fault explicitly
recovers 22–38%.

## Non-negotiables

**Score a process change against the STANDARD process, never against its own
clean twin.** Replicating ratings, re-rating before deploying, or changing the
acquisition also changes the *clean* run, so excess-over-own-twin measures the
arm against a handicapped reference. This is not a subtlety: it once made
replication look like it recovered a third of the cost when it recovers nothing.
`scripts/analyse_boba_adaptations.py` implements the right estimand
(`cost = ref_noisy - ref_clean`, `gain = ref_noisy - trt_noisy`,
`price = trt_clean - ref_clean`); import `paired_frame` and `summarise` from it
rather than writing a second copy.

**Never report an acquisition robustness ranking without naming the incumbent.**
The observed-max incumbent removes 15.6% of the measured cost on its own, −41%
for LogPI, and exactly 0.000 for UCB and qNEI (which never read `best_f` — that
is the control validating the pairing). Every ranking here ranks
acquisition-and-incumbent *pairs*.

**Do not re-raise the GP-noise confound.** Supplying the true injected variance
as `train_Yvar` removes 5.0% of the cost, nothing survives FDR, and the
onset ratio does not shrink. The natural mechanism was tested and rejected. The
onset effect remains unexplained; say so rather than explaining it.

**`random` and `sobol` must show exactly 0.000e+00 excess regret.** They ignore
observations, so any other number means the pairing is broken. The analyses fail
loudly on this; keep it that way.

**Aggregate as ratios of landscape means with a landscape bootstrap**, not as
means of per-run ratios. The suite's achievable improvement (`opt_z`) spans 45x,
so a per-run ratio lets a landscape with tiny regret outvote one with large.

## Traps that have cost real time

- **A run's variant lives only in its file name.** One output directory can hold
  four spike sizes or two ceiling modes; `run_metadata.json` is overwritten by
  the last invocation and the adaptation fields are not written per row.
  `evaluate_research_question.py` parses the suffix into a `variant` column and
  puts it in `CONDITION_COLS`. A variant that changes the CLEAN run needs its own
  `--output-dir`, and the simulator's own guard refuses anything else.
- **A baseline's oracle name can contain an underscore** (`extra_trees`). Parse
  it from the logged `oracle_model`, never by regex over the file name, or every
  fitted-oracle arm silently fragments.
- **Bash heredocs eat backslashes.** Use Write/Edit for LaTeX, regexes and
  Windows paths. A heredoc will quietly corrupt `\section` into `section`.
- **`nohup bash -c '...' &` does not survive** the tool call that started it;
  `nohup python ... &` does. Long jobs are resumable (`--resume`) for this reason.
- **Background jobs die when the machine sleeps.** Always `--resume`.
- **The interpreter is `C:\Users\markc\AppData\Local\Programs\Python\Python312`.**
  `torch` there is a CUDA build; do NOT `pip install -r requirements-eval.txt`,
  which would replace the wheel with the pinned CPU version.
- **Smoke-test a driver before an overnight run.**
  `run_boba_budget_neutral.ps1 -Smoke` runs every variant of every arm at toy
  size. It has already caught an arm silently skipped by a `--help` guard and an
  arm whose two variants collided in one directory — each would have cost hours.

## The analyses, and what each is for

| Script | Question |
|---|---|
| `bo_synthetic_error_simulation.py` | the known-function arm's simulator (20 analytic landscapes) |
| `bo_sensor_error_simulation.py` | the data-driven arm's simulator; both share the core |
| `evaluate_research_question.py` | pairs noisy runs with clean twins, per landscape |
| `analyse_boba_robustness.py` | cross-benchmark synthesis, rankings, descriptors |
| `analyse_boba_adaptations.py` | a process change against the standard process |
| `rescore_ship_rules.py` / `analyse_ship_rules.py` | what the same trials would have shipped under another rule |
| `decompose_regret.py` | deployed regret → search loss + selection loss |
| `replay_end_of_study.py` | tournaments and confirmation tests, replayed exactly |
| `replay_stopping.py` | onset detection + freeze; decision-stable stopping |
| `budget_split.py` | how many of T trials to spend identifying rather than searching |
| `changepoint_compare.py` | CUSUM vs GLR vs BOCPD on the same residual streams |
| `elicitation_compare.py` | a comparison loop against a rating loop at equal human cost |

Replays are exact, not approximations: the loop's proposal at trial *t* depends
only on trials 1..*t*-1, so a study that changes only its ending IS the logged run
truncated and continued. Every replay re-derives the standard process from the
prefix and aborts if it disagrees with the log.

## What is settled, and what is not

Settled, with the measurement: selection dominates the cost of error; no
acquisition-side change reaches the deployed design (augmented EI recovers 41% of
the trajectory cost and none of the deployed cost, which is the cleanest
statement of the mechanism); a comparative sitting over ~a third of the budget
recovers the most of any budget-neutral remedy; four confirmation trials cut
false improvement claims from 15.7% to 0.5%; a comparison loop is *exactly*
immune to drift and no help at all against a saturating scale; the CUSUM already
matches the classical optimal detector.

Not settled, and worth saying so in any write-up: **the human is simulated.** The
fitted oracle's held-out R² is 0.55 at best. Every remedy here is harmful below
0.25σ because it pays a fixed price whether or not error is present, and we have
no good way to tell a practitioner which regime they are in. The onset effect is
unexplained. A model-based budget rule *loses* to a fixed k, and we did not find
one that does better.

## Before you claim a result

1. `python -m pytest tests/ -q` — 650+ tests; they encode most of the above.
2. `python scripts/check_paper.py` — static checks on the paper's inputs, refs
   and table shapes. It does NOT verify that the document compiles.
3. Compile the paper (`pdflatex`, `bibtex`, `pdflatex` x2) and check the page
   count: ICLR allows **9 pages through Limitations**, and the main body is
   currently at 11. That overflow is known and unresolved.
