# Initial benchmark and Python 3.15 validation, 2026-10-21

These are software validation runs at deliberately small budgets. They do not
establish an acquisition ranking, a recovery claim, or a new paper result.
The full screening and held-out studies have not been executed.

## Environments

Both environments run on native Windows x86-64, on the CPU:

| Python | Torch | BoTorch | NumPy | SciPy | pandas |
|---|---|---|---|---|---|
| 3.13.12 | 2.13.0+cu132, CPU execution | 0.18.1 | 2.5.2 | 1.18.0 | 3.0.5 |
| 3.15.0 | 2.14.1+cpu | 0.18.1 | 2.5.4 | 1.18.1 | 3.0.6 |

Python 3.15 was installed in an isolated virtual environment. Torch's Python
3.15 wheel came from the [official CPU index](https://download.pytorch.org/whl/cpu).
The initial ordinary PyPI install correctly failed to find that wheel; the
Python 3.15 requirements now include its official index. A clean binary-only
dependency resolution was also checked. `pip check` reports no conflicts.

`requirements-feedback-py315.txt` installs the feedback runtime;
`requirements-tests-py315.txt` adds full-suite dependencies. Missing
XGBoost/LightGBM and a parquet engine caused test failures in the first lean
Python 3.15 environment; these test dependencies are now declared explicitly.
The historical `requirements-eval.txt` numerical lock is preserved.

## Actual smoke runs

| Path exercised | Python | Designs and conditions | Completed runs |
|---|---|---|---:|
| New feedback runner | 3.13 | Hartmann 3D/6D, Branin, Ackley; eight methods; eight judgments; Gaussian/drift; onset 0/4 | 160 |
| New feedback runner | 3.15 | Same configuration, two Windows spawn workers | 160 |
| Instrument and exposure costs | 3.15 | Hartmann 3D; eight methods; all eleven faults; six budget units; clipped/rounded scale; comparisons cost two units | 184 |
| Legacy analytic simulator | 3.13 | Hartmann 3D/6D and Branin; LogEI/random/Sobol; Gaussian/dropout; onset 0/4; eight trials | 45 |
| Legacy analytic simulator, final code | 3.15 | Same short legacy configuration, two Windows spawn workers | 45 |

The new runner's standard smoke uses one seed, 32 catalog designs and magnitude
1 landscape SD. The instrument smoke uses 16 catalog designs, magnitude .25,
rounding step .5 and bounds [-2,2] in landscape SDs. Its lapse magnitude is a
probability and its contrast magnitude is dimensionless, as labeled in the
outputs. It gives comparisons three judgments/six exposures and numeric
ratings six judgments/six exposures.

Checks on the saved outputs:

- Every required task completed and every final metric was finite.
- Random and Sobol had **0.000e+00 excess final search regret**, in landscape
  SDs, under response error, including the instrument smoke.
- Final deployed regret equaled final search regret plus final selection loss
  to floating-point precision: maximum absolute difference about 5.4e-15
  landscape SDs in the standard smoke and 4.5e-16 in the instrument smoke.
- Final selection losses were nonnegative to numerical precision.
- New-runner resume reused all 160 checkpoints on both Python versions, with
  no pending simulations.
- Legacy Python 3.15 resume preserved all 45 per-run CSV byte hashes,
  including run IDs. Changing the candidate pool from 32 to 33 was refused
  before any run completed; the old success marker was removed. Resuming the
  original configuration restored completion without changing those CSVs.

The audit unit tests additionally run the numerical bump extension for four
trials and check exact random-search pairing and safe resume. Nonfinite unused
landscape diagnostics no longer invalidate an otherwise finite objective.

## Corrections and performance

- Learned-noise dropout no longer crashes while logging an undefined additive
  variance. Known-noise dropout still fails explicitly.
- Anchor drift estimates center both trial indices and ratings within anchor
  identity. Deployment uses the corrected ratings too.
- New legacy logs carry initialization size and a full run fingerprint.
  Resume checks configurations, objective/data, code, Python, platform and
  numerical/oracle libraries. Unverified old logs require the explicit
  `--allow-legacy-resume` compatibility option.
- The first rating-dependent response accounts for initialization. Random and
  Sobol have no such response. Post-onset AUC windows remain anchored at the
  first affected observation.
- Failed executions cannot leave a stale `SWEEP_COMPLETE`; dry runs preserve
  an existing marker.
- Analysis-only commands do not start experiments when inputs are missing.
  Invalid settings and misspelled faults/options fail before execution.
- Checkpoint validation checks budgets, arity, exposure counts, feedback,
  recommendation eligibility, final ship rules and the regret decomposition.
- GP recommendations use requested marginal predictions and reuse cached
  moments for LogEI. Numerical tests compare against BoTorch's own analytic
  acquisition. Catalog objectives are cached per worker across fault cells;
  summaries discard traces as they read checkpoints; queued work is bounded.

No before/after throughput benchmark has been measured. These improvements
remove redundant computation and retained data; the per-run wall times should
be used to size the actual study on its target hardware.

## Reproduce

Final complete-suite results, after the corrections above:

| Python | Passed | Skipped | Failed | Wall time |
|---|---:|---:|---:|---:|
| 3.13.12 | 1,261 | 9 | 0 | 9m45s |
| 3.15.0 | 1,245 | 12 | 0 | 7m28s |

Some tests require optional packages or reference data/logs that are absent in
an environment; these account for the skips. The suites retain warnings from
the installed numerical libraries and deliberately degenerate fixtures.
Python 3.15 emits Torch's existing JIT warning on import; the exercised GP and
benchmark paths run successfully. Neither JIT nor `torch.compile` is required
by these runs. Static paper checks, Python 3.15 compilation, dependency checks
and Git whitespace checks also pass. The paper was not compiled with LaTeX.

`paper/COMMANDS.md`, section 12, records the exact initial execution commands.
Use fresh output directories after changing source or environment; strict
fingerprints intentionally prevent resuming logs generated by earlier code.

```
python -m pytest tests/ -o addopts= -q
.venv-feedback315/Scripts/python.exe -m pytest tests/ -o addopts= -q
python scripts/check_paper.py --paper paper
```

Raw local logs and environment freezes are kept under `validation-local/`
(ignored by Git). Benchmark output directories are ignored and were not added
to the commit. CI is configured to run the Python 3.15 full suite and Hartmann
smoke/resume on Windows and Ubuntu. Remote CI results are separate from this
local validation record.
