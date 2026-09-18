# Overleaf upload — what is in this bundle and what is still open

Upload the zip with Overleaf's **New Project → Upload Project**. Set `main.tex`
as the main document and the compiler to **pdfLaTeX**; Overleaf runs BibTeX by
itself. The build must finish with **zero errors and zero undefined references**
— it does on TeX Live 2026 with these exact files.

## Contents

| file | role |
|---|---|
| `main.tex` | the paper; nothing else is `\input` except the tables below |
| `references.bib` | 25 entries, 24 cited |
| `iclr2027_conference.sty`, `.bst` | the **official** ICLR 2027 style, byte-for-byte from `github.com/ICLR/Master-Template/iclr2027` — do not edit |
| `natbib.sty`, `fancyhdr.sty`, `math_commands.tex` | vendored with the style, as upstream ships them |
| `tables/*.tex` | 26 generated tables; regenerate with `python scripts/make_boba_paper_tables.py`, never by hand |

Not included on purpose: `main.pdf` and the auxiliary files (Overleaf rebuilds
them), `iclr2027_conference_ORIGINAL.tex` (upstream's instructions, not part of a
submission), and `archive/`.

## Anonymity — read before you touch the preamble

`\iclrfinalcopy` is commented out and must stay so. Uncommenting it prints the
author block, and a non-anonymous ICLR submission is rejected without review.
The header currently reads "Under review as a conference paper at ICLR 2027" and
the title page reads "Anonymous authors", which is what it should read.

## Open before submission — five `\todo` markers, all author decisions

Line numbers are as of this build; `grep -n 'todo{' main.tex` gives the current
positions.

| `main.tex` line | what |
|---|---|
| 38 | `\author{\todo{authors}}` — leave for review; fill in at camera-ready |
| 335 | cite the three human studies in the third person |
| 632 | repository URL and commit — at camera-ready only, it de-anonymises |
| 645 | **AI use statement**: the list of required-disclosure tasks for which no AI was used. The 2027 policy asks for this negative list explicitly; only the authors know it |
| 667 | **Ethics statement**: the original studies' ethics approval and consent terms, and that re-analysis is within them |

Once all five are resolved, delete the `\newcommand{\todo}` definition in the
preamble too, or any marker added later ships as red text.

Both the AI use statement and the Ethics statement follow the headings and the
structure of the 2027 template (`iclr2027_conference_ORIGINAL.tex`, lines
397–426). The AI statement is required; the ethics statement is recommended.

## Known blockers that are NOT fixed here

**Page limit — now met.** ICLR 2027 allows a *strict* **9 pages** of main text
at submission (10 at rebuttal). The main text through *Limitations* ends at the
bottom of **page 9**, and page 10 opens with the Reproducibility statement, which
does not count. That took: three sections to the appendix (fitted oracle,
one-shot loss, input error), the three controls to the appendix, the extra-trials
paragraph and its table to Appendix *Extra trials, by arm*, and a compression of
Setup, the seven-checks section and the Discussion. **The margin is about one
line.** Any addition to the main text — a sentence in the abstract counts — will
push the Limitations onto page 10; rebuild the bundle after every edit and read
its last line, which says FITS or OVER.

**Title and abstract.** The title names "the One Number That Predicts It" —
that is `frag`, whose analysis now lives in Appendix B. The abstract is 609
words; the readiness review flagged it at 524. Both are author decisions.

**No figure.** The paper has none. The readiness review (Section 4) calls this
out.

The full list of open items is `handover/findings.md` in the repository — it is
the single register, with severity and status columns.
