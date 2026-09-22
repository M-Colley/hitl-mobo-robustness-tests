# Overleaf upload — what is in this bundle and what is still open

Upload the zip with Overleaf's **New Project → Upload Project**. Set `main.tex`
as the main document and the compiler to **pdfLaTeX**; Overleaf runs BibTeX by
itself. The build must finish with **zero errors and zero undefined references**
— it does on TeX Live 2026 with these exact files.

## Contents

| file | role |
|---|---|
| `main.tex` | the paper; nothing else is `\input` except the tables below |
| `references.bib` | 39 entries, 38 cited |
| `iclr2027_conference.sty`, `.bst` | the **official** ICLR 2027 style, byte-for-byte from `github.com/ICLR/Master-Template/iclr2027` — do not edit |
| `natbib.sty`, `fancyhdr.sty`, `math_commands.tex` | vendored with the style, as upstream ships them |
| `tables/*.tex` | generated tables; regenerate with `python scripts/make_boba_paper_tables.py`, never by hand |
| `figures/*.pdf` | four figures, drawn by `python scripts/make_paper_figures.py` from the analysis CSVs |
| `authors.tex` | the real author block; printed only once `\iclrfinalcopy` is uncommented |

**Source layout.** Every prose paragraph of `main.tex` is one source line, so
Overleaf's soft wrap shows a paragraph as a paragraph instead of breaking it
wherever an old hard wrap fell. Equations, tables and list items keep their own
lines. The reflow was checked by compiling before and after: the PDF text is
identical on every page.

Not included on purpose: `main.pdf` and the auxiliary files (Overleaf rebuilds
them), `iclr2027_conference_ORIGINAL.tex` (upstream's instructions, not part of a
submission), and `archive/`.

## Anonymity — read before you touch the preamble

`\iclrfinalcopy` is commented out and must stay so. Uncommenting it prints the
author block, and a non-anonymous ICLR submission is rejected without review.
The header reads "Under review as a conference paper at ICLR 2027" and the title
page reads "Anonymous authors", which is what it should read.

**Anonymity check before uploading the PDF.** The paper links
`anonymous.4open.science/r/hitl-mobo-robustness-tests-17E2`. On 2026-09-18 that
mirror still served `LICENSE` with the copyright holder's name and `datasets.json`
with the `github.com/M-Colley/...` data URLs unmasked, because Anonymous GitHub
only masks the terms listed in its settings. Add the names, GitHub handles, the
Windows user name in logged paths and the institutions there (or recreate the
mirror under a neutral id from a scrubbed export) and re-check those two files
through the mirror before submitting. ICLR desk-rejects a paper whose
supplementary material reveals the authors.

## Page limit

ICLR 2027 allows a *strict* **9 pages** of main text at submission (10 at
rebuttal). `python paper/build_overleaf_bundle.py` compiles the bundle in a clean
directory and prints where the main text ends and whether it FITS; on the
2026-09-22 build it ends 40% down page 9. Rebuild after every edit to the main
text and read that last line.

## Still open, and the authors' to decide

- **A DOI.** The reproducibility statement says an archived snapshot with a DOI
  will accompany the camera-ready version; that deposit is yours to make.
- **Two defects in the fitted-oracle companion's data** (Appendix B, "The error
  processes in the archival data"): `opticarvis` mixes two rating scales in 40 of
  586 rows, and `provoice` enters Predictability with the wrong sign. The paper
  states both and relies instead on the oracle-isolation experiment; fixing
  `datasets.json` and rerunning the companion arm is a few hours of compute.
- The references in `references.bib` added on 2026-09-22 were checked against
  publisher records; the older conference and arXiv entries were not all.

The full list of open items is `handover/findings.md` in the repository — it is
the single register, with severity and status columns.
