# ICLR 2027 submission

## Template

The style files here are the **official** ICLR 2027 ones, downloaded from
[ICLR/Master-Template](https://github.com/ICLR/Master-Template/tree/master/iclr2027):

| file | role |
|---|---|
| `iclr2027_conference.sty` | the conference style — do not edit |
| `iclr2027_conference.bst` | the bibliography style |
| `fancyhdr.sty`, `natbib.sty` | vendored dependencies, as upstream ships them |
| `math_commands.tex` | optional macros; not currently `\input` by `main.tex` |
| `iclr2027_conference_ORIGINAL.tex` | upstream's instructions file, kept for reference — **not part of the submission** |

`\iclrfinalcopy` stays commented out. Uncommenting it de-anonymises the paper,
and a non-anonymous submission is rejected without review; it goes in only for
camera-ready. The author block is a `\todo` for the same reason.

## Building

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The draft compiles under TeX Live 2026 (`pdflatex` at
`C:/texlive/2026/bin/windows`, not on the shell `PATH`) with no errors,
undefined references or overfull boxes. The main text through the Limitations
paragraph ends on page 9; the Reproducibility statement and the LLM-usage
section do not count toward ICLR's nine-page limit. The static checker catches
missing `\input` targets, dangling `\ref`s, unknown citation keys and malformed
generated tables without a compile:

```bash
python scripts/check_paper.py --paper paper
```

## Tables

**Do not edit `tables/*.tex` by hand.** Every number in them is read from the
sweep's analysis CSVs, so the paper cannot drift from the data:

```bash
python scripts/make_boba_paper_tables.py --analysis output-boba/analysis
```

`tables/benchmarks_wrapper.tex` is the one hand-written file in `tables/`: the
caption and float around the generated `benchmarks.tex`.

`known_noise.tex` and `incumbent.tex` appear once `run_boba_followups.ps1`
finishes and `compare_boba_arms.py` has run; until then the generator skips them
and `main.tex` refers to them in prose rather than `\input`ing them, so the
draft still builds.

## State

`\todo{...}` marks everything not yet backed by a completed run or not yet
written. Count them with the checker. The results sections quote the finished
10-seed sweep; the three ablation subsections are placeholders.
