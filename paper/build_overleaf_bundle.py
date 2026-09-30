"""Build the zip that goes into Overleaf's "Upload Project", and prove it compiles first.

A bundle that compiles here but not there is worse than none, so this script
does not trust the working directory. It copies exactly the files a submission
needs into a fresh temporary directory, runs pdflatex -> bibtex -> pdflatex there
with the same toolchain Overleaf uses, repeating the last pdflatex until the log
no longer asks for a rerun (at most MAX_PASSES_AFTER_BIBTEX passes; from a clean
directory this paper needs three), and refuses to write the zip unless BibTeX
exits 0, LaTeX reports no error, and the log carries no undefined citation or
reference and no rerun request. What it then zips is that verified directory,
so the zip and the test are the same bytes.

It also measures where the main text ends and prints it. TMLR sets no page
limit, but a paper's length should be justified by its content and an unusually
long main text (appendices not counted) slows the review, so the number is worth
seeing without opening the PDF.

    python paper/build_overleaf_bundle.py
    python paper/build_overleaf_bundle.py --out paper/my-bundle.zip
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
DEFAULT_OUT = PAPER / "hitl-noisy-feedback-tmlr.zip"

# Exactly what a submission needs, nothing that Overleaf regenerates. Figures are
# collected from the sources in stage(). The three style files are the official
# ones from https://github.com/JmlrOrg/tmlr-style-file; tmlr.sty loads natbib.
STYLE_FILES = ["tmlr.sty", "tmlr.bst", "fancyhdr.sty"]
TOP_FILES = ["main.tex", "references.bib"] + STYLE_FILES
# Copied when present. README_OVERLEAF.md is git-ignored (it names a handle), so a
# fresh checkout lacks it and the build must not require it.
OPTIONAL_FILES = ["README_OVERLEAF.md"]

# The statements (AI use, broader impact, reproducibility) follow the main text;
# the first of them closes the main text that the length refers to.
FIRST_UNCOUNTED_HEADING = re.compile(r"AI\s*USE\s*STATEMENT")


def inputs_of(tex: Path) -> list[str]:
    """The \\input targets of one file, comments stripped."""
    source = "\n".join(
        line.split("%", 1)[0] for line in tex.read_text(encoding="utf-8").splitlines()
    )
    return sorted(set(re.findall(r"\\input\{([^}]+)\}", source)))


def graphics_of(tex: Path) -> list[str]:
    """The \\includegraphics targets of one file, comments stripped."""
    source = "\n".join(
        line.split("%", 1)[0] for line in tex.read_text(encoding="utf-8").splitlines()
    )
    return sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)))


def all_inputs(main_tex: Path) -> list[str]:
    """Every file reached by \\input from main.tex, transitively.

    A table wrapper can \\input the table it wraps (tables/benchmarks_wrapper.tex
    does), and the first version of this script followed only main.tex's own
    inputs. It compiled here, where the file happened to be on disk, and would
    have failed on Overleaf with "File tables/benchmarks.tex not found" -- the
    clean-room compile below is what caught it.
    """
    seen: list[str] = []
    queue = inputs_of(main_tex)
    while queue:
        rel = queue.pop(0)
        if rel in seen:
            continue
        seen.append(rel)
        src = PAPER / f"{rel}.tex"
        if not src.is_file():
            raise SystemExit(f"\\input{{{rel}}} is referenced but {src} does not exist")
        queue.extend(inputs_of(src))
    return sorted(seen)


def stage(dst: Path) -> None:
    for name in TOP_FILES:
        src = PAPER / name
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, dst / name)
    # The real author block, if this machine has it (git-ignored; see main.tex).
    # The zip is private to Overleaf, the repository mirror is not.
    if (PAPER / "authors.tex").is_file():
        shutil.copy2(PAPER / "authors.tex", dst / "authors.tex")
    for name in OPTIONAL_FILES:
        if (PAPER / name).is_file():
            shutil.copy2(PAPER / name, dst / name)
    for rel in all_inputs(PAPER / "main.tex"):
        target = dst / f"{rel}.tex"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER / f"{rel}.tex", target)
    # Figures: every \includegraphics target of main.tex and its inputs. The first
    # bundle with figures compiled here and failed in the clean room for want of
    # them; this is what the clean-room compile is for.
    tex_files = [PAPER / "main.tex"] + [PAPER / f"{rel}.tex" for rel in all_inputs(PAPER / "main.tex")]
    for tex in tex_files:
        for rel in graphics_of(tex):
            src = PAPER / rel
            if not src.is_file():
                raise SystemExit(f"\\includegraphics{{{rel}}} is referenced but {src} does not exist")
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


# pdflatex passes after bibtex. Two were not enough from a clean directory: the
# log still said "Label(s) may have changed", which a third pass clears.
MAX_PASSES_AFTER_BIBTEX = 4
# What LaTeX, natbib, hyperref and rerunfilecheck print when another pass would
# change the output.
RERUN_REQUEST = re.compile(r"Label\(s\) may have changed|Rerun to get|Please rerun LaTeX|"
                           r"rerunfilecheck Warning|Citation\(s\) may have changed")


def needs_rerun(log: str) -> bool:
    """Whether a pdflatex log asks for another pass."""
    return RERUN_REQUEST.search(log) is not None


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")


def compile_in(dst: Path) -> Path:
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    first = run(latex, dst)
    if first.returncode != 0:
        raise SystemExit(f"first pdflatex pass failed:\n{first.stdout[-3000:]}")
    bib = run(["bibtex", "main"], dst)
    if bib.returncode != 0:
        raise SystemExit(f"bibtex exited {bib.returncode}; Overleaf would show this as an error:\n{bib.stdout}")
    passes = 0
    while True:
        again = run(latex, dst)
        passes += 1
        if again.returncode != 0:
            raise SystemExit(f"pdflatex failed after bibtex:\n{again.stdout[-3000:]}")
        log = (dst / "main.log").read_text(encoding="utf-8", errors="replace")
        if not needs_rerun(log):
            break
        if passes >= MAX_PASSES_AFTER_BIBTEX:
            raise SystemExit(f"the log still asks for a rerun after {passes} pdflatex passes "
                             "following bibtex; the cross-references have not settled")
    print(f"pdflatex passes after bibtex: {passes}")
    errors = [ln for ln in log.splitlines() if ln.startswith("! ")]
    undefined = [ln for ln in log.splitlines()
                 if "Warning" in ln and ("undefined" in ln.lower())]
    if errors or undefined:
        raise SystemExit("the bundle does not compile cleanly:\n" + "\n".join(errors + undefined))
    return dst / "main.pdf"


def measure(pdf: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "pypdf not installed; page position not measured"
    reader = PdfReader(str(pdf))
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "")
        match = FIRST_UNCOUNTED_HEADING.search(text.upper())
        if not match:
            continue
        j = match.start()
        # What precedes the heading on its page, minus the running header. If that
        # is empty, the main text ended on the previous page and the heading merely
        # opens this one; reporting "ends on page i" in that case once read a
        # 9-page paper as a 10-page one.
        before = re.sub(r"^(\d+\s*)+", "", text[:j].strip())
        before = re.sub(r"^Under review as submission to TMLR\s*", "", before).strip()
        if not before:
            return (f"main text ends at the bottom of page {i - 1}; the statements open page {i} "
                    f"of {len(reader.pages)} (TMLR sets no page limit)")
        frac = j / max(1, len(text))
        return (f"main text ends {frac * 100:.0f}% down page {i} of {len(reader.pages)}, "
                f"about {i - 1 + frac:.1f} pages (TMLR sets no page limit)")
    return f"{len(reader.pages)} pages; the AI use statement heading was not found"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="overleaf-bundle-") as tmp:
        dst = Path(tmp) / "bundle"
        dst.mkdir()
        stage(dst)
        pdf = compile_in(dst)
        verdict = measure(pdf)
        # Zip the verified sources only -- not the PDF or the aux files.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(dst.rglob("*")):
                keep = path.suffix in {".tex", ".bib", ".sty", ".bst", ".md"} or (
                    path.suffix == ".pdf" and path.parent != dst)   # figures, not main.pdf
                if path.is_file() and keep:
                    zf.write(path, path.relative_to(dst).as_posix())
        names = zipfile.ZipFile(args.out).namelist()

    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB, {len(names)} files)")
    print("compiled clean in a fresh directory: bibtex 0, no LaTeX errors, no undefined references, "
          "no rerun request")
    print(verdict)


if __name__ == "__main__":
    main()
