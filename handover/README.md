# Handover of the ICLR 2027 submission

**Delete this folder before building the submission or the supplementary zip.** It names people, accounts and machines on purpose, because it lists what must be scrubbed. Run `python handover/strip_handover_blocks.py` first. That removes the marked handover blocks from the root `README.md` and `AGENTS.md`. Then delete `handover/`.

Written 2026-09-16 after three audit passes.

| file | content |
|---|---|
| `README.md` | this entry point with deadlines, state, actions and checklists |
| `findings.md` | the single register of known problems with severity, location, fix and status |
| `readiness_review.md` | the paper as an ICLR reviewer sees it, the restructuring plan, compliance and the day plan |
| `strip_handover_blocks.py` | removes the marked handover blocks from the root files |
| `history/code_audit.md` | first audit pass, the evidence behind IDs A1 to C15 |
| `history/second_pass_findings.md` | second pass, the evidence behind IDs D1 to F9 |
| `history/third_pass_findings.md` | third pass with executed checks H1 to H6, findings I1 to I23 and the table of corrections |
| `history/_superseded_*_v1.md` | first versions of the review and the audit, kept for the record and partly wrong |
| `history/_audit_bundle_2026-09-16.tgz` | code snapshot the audits read, excluded from git by `handover/.gitignore` |

## Deadlines

The abstract and the final author list are due Friday 18 September 2026, 23:59 AoE. The paper and supplementary are due Friday 25 September 2026, 23:59 AoE. The formal requirements (9-page main text, 2027 style files, mandatory AI use statement, reciprocal reviewing, dual-submission rule, OpenReview profiles) are in `readiness_review.md` Section 3 with links to the official pages.

## Before pushing this folder

The remote is `github.com/M-Colley/hitl-mobo-robustness-tests`. The audit could not check its visibility. If the repository is public, keep `handover/` off the public branch, because it names the authors next to a description of the ICLR submission and its weaknesses, which reviewers could find during double-blind review. A private branch, a private fork or a local copy is safe. The tarball in `history/` is excluded by `handover/.gitignore`.

## What exists and where

The paper is `paper/main.tex`. It compiles to 28 pages with 9 main pages plus two lines, 19 appendices, 27 tables, no figures and four `\todo` markers. `scripts/make_boba_paper_tables.py` generates `paper/tables/`. Table 25 is the exception and is typed inside `main.tex`.

The main sweep `output-boba/` (79,200 runs) and several other ignored arms (`output-boba-knownnoise`, `-incumbent`, `-extensions`, `-ladder`, `-mo`, `-budget25`, `-budget100`, `-instrument`, `-robust`, the confirmatory runs, `output-fitted*`, `output/`) exist only on the Windows simulation machine. The 33 side arms tracked in git were checked and are intact.

Back up everything `make_boba_paper_tables.main` reads, today. That covers `output-boba/analysis/`, every `output-boba/*/evaluation/`, `output-boba/run_metadata.json`, the analysis and evaluation directories of every arm named in `make_boba_paper_tables.py:1088-1167`, and `output/noise_anchor.csv`, `output/noise_calibration.csv` and `output/best_oracle_models.json`.

The simulator uses the CPU only, with one single-threaded process per worker. The main sweep took 19.7 hours on 24 workers. That is about 21.5 worker-seconds per run and about 470 worker-hours in total, or about five days on four workers of the same speed. Keep simulations on the 24-worker machine and move analysis files between machines.

The code has 38 scripts (about 24,000 lines) and 29 test files (443 test functions). The PowerShell drivers read `$env:PYTHON` and otherwise use `$env:LOCALAPPDATA\Programs\Python\Python312`. torch on the simulation machine is a CUDA build, so do not install `requirements-eval.txt` there.

## State of the work

Executed or computed from real outputs. 98 tests pass offline, and no test fails on an assertion. The landscape statistics reproduce for 22 of 29 landscapes to 1.6e-13. Four input-error tables regenerate byte-identically. Two Table 5 cells, the input-error rows of Table 24 and four prices in Table 26 reproduce from raw outputs. The side arms show no optimiser fallbacks and no missing values, and the paired input-error arms share their slip draws.

Confirmed by reading. The simulator, the clean and noisy pairing, the four error processes, the standardisation, the decomposition arithmetic, the landscape bootstrap, the quasi-Poisson fit, the frag computation, the rescoring and replay machinery, and the separation of clean-run-changing variants into their own directories.

Not verified. About 345 tests that need torch, and every number that needs the main-sweep or fitted-arm outputs.

## The most important open items

The full list is `findings.md`. These decide acceptance.

| ID | item |
|---|---|
| A1, I1 | The headline metric is time-averaged search loss, and the shipped design loses two to three times as much. |
| C15, review | 2026 style, page spill, four `\todo` markers, a 524-word abstract and no figures. |
| C14, D3, I17 | Identifiers throughout the files that would go into the supplementary. |
| A2, A3 | A unit bug in the budget rule and an uncentred interaction behind a stated conclusion. |
| I2, E3, E4 | Selection is about half of the cost at realistic magnitudes, and several claims about it are too strong. |
| E2, I3, I6, I7 | Table 25 is hand-typed, lacks prices and definitions, and contains an invalid ratio. Two of its arms use a pooled reference. |
| C1, D7, C5, C6, C7, D9, I15 | Numbers with no recorded producer. |
| I8, D4, B3, D5 | The rater-noise anchor and the fitted-oracle claims rest on weak oracles and mixed denominators. |
| E1, E10, E12 | Missing citations and an AI use statement that does not yet meet the policy. |
| D1, D2 | The test suite runs nothing as shipped, and the parity tests only run on the author's machine. |

## Ordered actions

1. On the simulation machine, add a module-level `pytest.importorskip("numpyro")` to `tests/test_hierarchical_oracle.py`, run `python -m pytest tests/ -q`, and file the log in `handover/`.
2. Fix A2, A3, A4, B4 and C10, then rerun. For A2, delete the old `budget_split_derived.csv` and run `scripts/budget_split.py` without `--summary-only`, with the k-grid and sitting ρ that produced Section 9. The defaults (k-grid 2,3,5,8,12 and ρ 0.5) cannot produce k = 16 to 30, so recover the original command from the shell history or the outputs. For A3, rerun `scripts/analyse_boba_robustness.py` on `output-boba`. Then run `python scripts/make_boba_paper_tables.py --analysis output-boba/analysis`.
3. Read the 1σ, onset-0 row of `regret_decomposition.csv` and the `gp_kernel` column of `ship_rules_per_run.csv`.
4. Decide the metric framing (A1) and the title (I2). Both are author decisions.
5. Generate Table 25 by script with a price column and fault definitions, replace its missing-rating ratio, and rescore augmented EI and Thompson sampling against EI and LogEI.
6. Work through the text items in `findings.md` Section 1.
7. Optional reruns go into fresh output directories, because the drivers skip any directory that holds `SWEEP_COMPLETE` and every run resumes by file name. For B2, set `use_rbf_kernel=True` in `robust_gp.py:245` and point the `studentt` entry of `analyse_boba_adaptations.py` at the new directory. For provoice, write a driver that mirrors the fitted-arm sections of `run_boba_gaps_rest.ps1` and `run_boba_gaps2.ps1` with `--normalize-objective`, and remove the hard-coded `normalize=False` in `anchor_noise_scale.py:80`. `run_provoice_normalized.ps1` uses a different design and cannot reproduce the companion arm. For the confirmatory rerun, keep the 0.10 headroom screen, exclude Rosenbrock by name, and date the new preregistration before launching.
8. Restructure the paper as in `readiness_review.md` Section 4.
9. Build the supplementary with the checklist below.

## Numbers with no recorded producer

Each needs a committed script and command or removal from the paper. The AI policy treats an LLM-produced falsehood as a Code of Ethics violation, so an untraceable number is a real exposure. The list is Table 8's row set (n = 8, 15, 25, 50), the Appendix C ratio interval [1.1, 17], the elicitation arm's T = 20, the three MO-front sentences, Appendix P's freeze ceiling of 82% and 54%, the condition-averaged scalar Kendall's W of 0.52, Table 25, the k-sweep and budget-rule invocations behind Section 9, the producers of `noise_anchor.csv` and `noise_calibration.csv`, and the instrument spans and steps. Record every command in `paper/COMMANDS.md`.

## Pre-submission checklist

1. Run `python handover/strip_handover_blocks.py`, then delete `handover/`.
2. Decide whether `AGENTS.md` ships. After the strip it names nobody.
3. Export the repository without `.git`, `.github`, `sandbox/`, `external_datasets/`, `catboost_info/`, `paper/archive/`, LaTeX build files and `run_*.bat`.
4. Replace `LICENSE` with an anonymous licence notice.
5. Remove or anonymise the identifiers. The Windows username appears in `tests/test_boba_benchmarks.py:36`, `tests/test_mo_halo.py:37`, `tests/test_error_extensions.py:39`, `tests/test_acquisition_extensions.py:39`, `docs/known-function-arm-2026-09-06.md` and every `run_metadata.json`. GitHub account names appear in `datasets.json`, `datasets-ehmi.json`, `datasets-provoice.json`, `datasets-extended.json`, `tests/test_cli_smoke.py`, `tests/test_simulation_utils.py` and `docs/code-review-2026-06-12.md`. One data repository name contains a prior venue and year.
6. Search the export with `git grep -n -I -i -e markc -e colley -e susak -e chi25` (or `Select-String` on Windows) and expect no hits.
7. Update `scripts/check_paper.py:38` together with the style switch.
8. Include the analysis and evaluation files listed above, `boba_landscape_stats.json` and `boba_mo_stats.json`.

## Commands

```
python -m pytest tests/ -q
python scripts/check_paper.py --paper paper
python scripts/make_boba_paper_tables.py --analysis output-boba/analysis
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
