# TMLR submission

## Template

The style files here are the **official** TMLR ones, downloaded on 2026-09-28
from [JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file):

| file | role |
|---|---|
| `tmlr.sty` | the journal style — do not edit |
| `tmlr.bst` | the bibliography style |
| `fancyhdr.sty` | vendored dependency, as upstream ships it |

The package option decides what the style prints. `\usepackage{tmlr}` is the
anonymous submission and must stay so while the paper is under review, since
TMLR is double blind. `\usepackage[preprint]{tmlr}` de-anonymises the paper and
drops the TMLR header (for arXiv); `\usepackage[accepted]{tmlr}` is the
camera-ready and needs `\month`, `\year` and `\openreview`, which `main.tex`
carries commented out. The real author block is `authors.tex`, git-ignored; the
style prints "Anonymous authors" until one of the two options is set.

The ICLR 2027 template this paper was first written for is kept in the
git-ignored `archive/iclr2027_template/`.

## Building

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && pdflatex main
```

From a clean directory (no `.aux`), two passes after `bibtex` are not enough:
the log still says "Label(s) may have changed. Rerun to get cross-references
right." A third pass clears it. Before using the PDF, check that `main.log`
carries no "Rerun" or "may have changed" line, and run `pdflatex main` again if
it does. `build_overleaf_bundle.py` does this itself: after `bibtex` it repeats
`pdflatex` until the log no longer asks for a rerun (at most four passes) and
fails if it still does.

The paper compiles under TeX Live 2026 (`pdflatex` at
`C:/texlive/2026/bin/windows`, not on the shell `PATH`) with no errors,
undefined references or overfull boxes. TMLR sets no page limit, but a paper's
length should be justified by its content and an unusually long main text
(appendices not counted) slows the review. `python paper/build_overleaf_bundle.py`
compiles the bundle in a clean directory and prints where the main text ends.
The static checker catches missing `\input` targets, dangling `\ref`s, unknown
citation keys and malformed generated tables without a compile:

```bash
python scripts/check_paper.py --paper paper
```

## Tables

**Do not edit `tables/*.tex` by hand.** Every number in them is read from the
sweep's analysis CSVs, so the paper cannot drift from the data:

```bash
python scripts/make_boba_paper_tables.py --analysis output-boba/analysis
```

One generated table has its own producer: `tables/sitting_by_magnitude.tex` is
written by `python scripts/sitting_by_magnitude.py` (together with its CSV and
JSON outputs under `output-boba/analysis/review/`), not by
`make_boba_paper_tables.py`, so rerun it after any change to the sitting's
inputs.

`tables/benchmarks_wrapper.tex` is the one hand-written file in `tables/`: the
caption and float around the generated `benchmarks.tex`. The tables written
directly in `main.tex` (among them the remedies table of Section 7 and the
oracle-isolation and held-out tables of the appendix) quote numbers whose
producers are recorded in `COMMANDS.md`, in the order they must run.
