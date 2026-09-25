"""Build the zip that goes into Overleaf's "Upload Project", and prove it compiles first.

A bundle that compiles here but not there is worse than none, so this script
does not trust the working directory. It copies exactly the files a submission
needs into a fresh temporary directory, runs pdflatex -> bibtex -> pdflatex x2
there with the same toolchain Overleaf uses, and refuses to write the zip unless
BibTeX exits 0, LaTeX reports no error, and the log carries no undefined
citation or reference. What it then zips is that verified directory, so the zip
and the test are the same bytes.

It also measures where the main text ends -- ICLR 2027 allows a strict 9 pages
at submission -- and prints it, because that number decides whether the bundle
is submittable and nobody should have to open the PDF to learn it.

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
DEFAULT_OUT = PAPER / "hitl-noisy-feedback-iclr2027.zip"

# Exactly what a submission needs, nothing that Overleaf regenerates. Figures are
# collected from the sources in stage().
STYLE_FILES = ["iclr2027_conference.sty", "iclr2027_conference.bst",
               "natbib.sty", "fancyhdr.sty", "math_commands.tex"]
TOP_FILES = ["main.tex", "references.bib", "README_OVERLEAF.md"] + STYLE_FILES

# The statements follow the template's order (AI use, Ethics, Reproducibility);
# the first of them closes the counted main text.
FIRST_UNCOUNTED_HEADING = "AIUSE STATEMENT"  # pypdf drops the space after the small-caps "AI"


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
    for _ in range(2):
        again = run(latex, dst)
        if again.returncode != 0:
            raise SystemExit(f"pdflatex failed after bibtex:\n{again.stdout[-3000:]}")
    log = (dst / "main.log").read_text(encoding="utf-8", errors="replace")
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
        j = text.upper().find(FIRST_UNCOUNTED_HEADING)
        if j < 0:
            continue
        # What precedes the heading on its page, minus the style's margin line
        # numbers and running header. If that is empty, the counted text ended on
        # the previous page and the heading merely opens this one -- which is
        # the layout the limit asks for, not an overflow. Reporting "ends on page
        # i" in that case once read a 9-page paper as a 10-page one.
        before = re.sub(r"^(\d+\s*)+", "", text[:j].strip())
        before = re.sub(r"^Under review.*?ICLR \d{4}\s*", "", before).strip()
        limit = 9
        if not before:
            counted = i - 1
            verdict = "FITS" if counted <= limit else f"OVER by {counted - limit} page(s)"
            return (f"main text ends at the bottom of page {counted}; the first uncounted "
                    f"section opens page {i} of {len(reader.pages)} -- {verdict} the strict "
                    f"{limit}-page ICLR 2027 submission limit")
        frac = j / max(1, len(text))
        verdict = "FITS" if i <= limit else f"OVER by about {i - limit - 1 + frac:.1f} page(s)"
        return (f"main text ends {frac * 100:.0f}% down page {i} of {len(reader.pages)} -- "
                f"{verdict} the strict {limit}-page ICLR 2027 submission limit")
    return f"{len(reader.pages)} pages; the '{FIRST_UNCOUNTED_HEADING}' heading was not found"


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
    print(f"compiled clean in a fresh directory: bibtex 0, no LaTeX errors, no undefined references")
    print(verdict)


if __name__ == "__main__":
    main()
