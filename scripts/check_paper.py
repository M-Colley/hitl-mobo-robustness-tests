"""Static checks on the paper source, for a machine with no LaTeX installed.

Not a substitute for compiling. It catches the failure modes that come from
generating tables programmatically and from editing a draft in pieces: an
``\\input`` whose file was never produced, a ``\\ref`` with no ``\\label``, a
``\\cite`` key missing from the bibliography, and a generated table whose braces
or column counts do not line up.

  python scripts/check_paper.py --paper paper
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def check(paper: Path) -> list[str]:
    problems: list[str] = []
    main = paper / "main.tex"
    if not main.exists():
        return [f"missing {main}"]
    source = _strip_comments(_read(main))

    # Style files the document depends on.
    for package in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", source):
        for name in (p.strip() for p in package.split(",")):
            local = paper / f"{name}.sty"
            # Any year's ICLR style: the name was hard-coded to 2026 once and went
            # stale the moment the venue moved.
            if re.fullmatch(r"iclr\d{4}_conference", name) and not local.exists():
                problems.append(f"\\usepackage{{{name}}} but {local.name} is not present")
    style = re.search(r"\\bibliographystyle\{([^}]+)\}", source)
    if style and not (paper / f"{style.group(1)}.bst").exists():
        problems.append(f"\\bibliographystyle{{{style.group(1)}}} but the .bst is not present")

    # \input targets, following one level of nesting.
    seen: set[str] = set()
    queue = re.findall(r"\\input\{([^}]+)\}", source)
    while queue:
        target = queue.pop()
        if target in seen:
            continue
        seen.add(target)
        path = paper / f"{target}.tex"
        if not path.exists():
            problems.append(f"\\input{{{target}}} -> missing {path}")
            continue
        nested = _strip_comments(_read(path))
        queue.extend(re.findall(r"\\input\{([^}]+)\}", nested))
        source += "\n" + nested

    # Cross-references.
    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    for ref in set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", source)):
        if ref not in labels:
            problems.append(f"\\ref{{{ref}}} has no matching \\label")

    # Citations.
    bib_files = re.search(r"\\bibliography\{([^}]+)\}", source)
    keys: set[str] = set()
    if bib_files:
        for name in (b.strip() for b in bib_files.group(1).split(",")):
            path = paper / f"{name}.bib"
            if not path.exists():
                problems.append(f"\\bibliography{{{name}}} -> missing {path}")
                continue
            keys |= set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", _read(path)))
    cited: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", source):
        cited |= {k.strip() for k in group.split(",")}
    for key in sorted(cited - keys):
        problems.append(f"\\cite{{{key}}} is not in the bibliography")

    # Generated tables: brace balance and a consistent column count.
    for table in sorted((paper / "tables").glob("*.tex")):
        body = _read(table)
        if body.count("{") != body.count("}"):
            problems.append(f"{table.name}: unbalanced braces")
        spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", body)
        if not spec:
            continue
        columns = len(re.findall(r"[lcrp]", spec.group(1)))
        for line in body.splitlines():
            line = line.strip()
            if not line.endswith(r"\\") or line.startswith("%"):
                continue
            # A \multicolumn{N}{..}{..} field occupies N columns, not one, so
            # count spans rather than ampersands. Without this a legitimate
            # spanning header reads as a short row.
            fields = line.rstrip("\\").split("&")
            cells = sum(
                int(m.group(1)) if (m := re.search(r"\\multicolumn\{(\d+)\}", f)) else 1
                for f in fields
            )
            if cells != columns:
                problems.append(
                    f"{table.name}: row has {cells} cells, tabular declares {columns}: "
                    f"{line[:70]}"
                )

    # An \input used twice renders the table twice and, when the generated file
    # carries its own \label, defines that label twice. LaTeX only warns, and
    # the second table is easy to miss in a long draft.
    inputs = re.findall(r"\\input\{([^}]+)\}", source)
    for name in sorted({n for n in inputs if inputs.count(n) > 1}):
        problems.append(f"\\input{{{name}}} appears {inputs.count(name)} times")

    # Same for labels. Count them in `source` ONLY: the \input resolution above
    # has already appended every input file's body to it, so re-scanning
    # tables/*.tex here would report every label in a generated table twice.
    all_labels = re.findall(r"\\label\{([^}]+)\}", source)
    for name in sorted({n for n in all_labels if all_labels.count(n) > 1}):
        problems.append(f"\\label{{{name}}} is defined {all_labels.count(name)} times")

    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args(argv)

    problems = check(args.paper)
    todos = len(re.findall(r"\\todo\{", _read(args.paper / "main.tex")))
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("Static checks passed (inputs, refs, citations, table shapes).")
    print(f"{todos} \\todo markers remain.")
    print("No LaTeX toolchain here: this does NOT verify that the document compiles.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
