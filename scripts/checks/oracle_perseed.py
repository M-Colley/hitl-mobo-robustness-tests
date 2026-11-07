"""The oracle-isolation result with each run seed's error grid scaled to its own oracle.

The published oracle-isolation arms (output-oracle-iso, -gaussian_process,
-mlp) scale every seed's error grid to the seed-7 oracle's sigma_f, while every
run seed refits its oracle; for the GP and MLP oracles the other seeds' sigma_f
are 0.57-1.98 times seed 7's (scripts/checks/oracle_sigmaf_seeds.py). The
per-seed arms (output-oracle-iso*-perseed, scripts/run_oracle_isolation_perseed.py)
rerun them with each seed's grid scaled to that seed's own sigma_f, so the
magnitude of a run is jitter_std / sigma_f(seed), which must fall on the grid
0.25/1/5 exactly (oracle_isolation.sigma_multiples). Their clean twins are the
published ones (copied after an identity check), and the exact arm is unchanged.

This script reuses the published pipeline, parameterised by directory:
oracle_isolation._analyse_one (the floor-gap files), oracle_achievable.main (the
achievable-improvement files under the four normalisers and the checks) and
oracle_families.main (the family comparison). It

  1. CONTROL: runs that pipeline on the published arms into a scratch directory
     outside the repository and requires every file it writes to be
     byte-identical to the published one (24 per family) and its printout to
     equal register_checks/oracle_achievable.txt and oracle_families.txt; it
     stops otherwise;
  2. checks that the per-seed arms are complete (every dataset evaluated, five
     seeds x six acquisitions x three magnitudes, each seed's runs on its own
     grid) and stops with the missing invocations otherwise;
  3. runs the pipeline on the per-seed arms, writing their analysis files into
     their own directories as the published arms hold theirs; and
  4. writes register_checks/oracle_perseed.txt: the quantities of tab:oracleiso
     (fitted and exact percentages, the ratio of landscape means with its
     landscape-bootstrap interval, landscapes on which the fitted design reports
     less, Wilcoxon p) for the trajectory search loss and the deployed design at
     0.25, 1 and 5 sigma, for both calibrations side by side, under the primary
     achievable-improvement normaliser, the three others of oracle_achievable.py
     (best_clean, sigma_f, both_best_clean) and the floor gap; then the
     family-contrast table of oracle_families.py for both calibrations, and the
     per-seed arms' full oracle_achievable and oracle_families printouts. The
     side-by-side numbers also go to review/oracle_perseed_comparison.csv.

In the per-seed printouts the estimator "sigma_f_seed7" keeps its name but no
longer describes the grid: there each seed's own sigma_f scales it, so
"sigma_f" is the estimator whose noise-to-normaliser ratio equals the exact
arm's. The landscapes share their random numbers by design; the bootstrap
treats them as independent.

    python scripts/checks/oracle_perseed.py                 # control, then the per-seed arms
    python scripts/checks/oracle_perseed.py --control-only  # the control alone
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import oracle_isolation as iso  # noqa: E402
import oracle_achievable as oa  # noqa: E402
import oracle_families as of  # noqa: E402
import run_oracle_isolation_perseed as drv  # noqa: E402

FAMILIES = ("tree", "gaussian_process", "mlp")  # the driver's names
OA_LABEL = dict(zip(FAMILIES, oa.ROOTS))         # tree, GP, MLP
OF_LABEL = dict(zip(FAMILIES, of.ROOTS))         # tree ensemble (selected), gaussian_process, mlp
SHORT = {"tree": "tree", "gaussian_process": "GP", "mlp": "MLP"}
PUBLISHED = {f: drv.published_root(f) for f in FAMILIES}
PERSEED = {f: drv.arm_root(f) for f in FAMILIES}
REGISTER = REPO / "output-boba" / "analysis" / "review" / "register_checks"
COMPARISON = REPO / "output-boba" / "analysis" / "review" / "oracle_perseed_comparison.csv"
# (normaliser label, file infix, summary suffix); the first is primary (tab:oracleiso's first two blocks)
NORMALISERS = (("achievable: oracle (primary)", "_achievable", ""),
               ("achievable: best_clean", "_achievable", "_best_clean"),
               ("achievable: sigma_f", "_achievable", "_sigma_f"),
               ("achievable: both_best_clean", "_achievable", "_both_best_clean"),
               ("floor gap: oracle", "", ""))
RESPONSES = (("", "trajectory search loss"), ("_deployed", "deployed design, final trial"))
SUMMARY_COLUMNS = ("frac_fitted", "frac_exact", "ratio_fitted_over_exact", "ratio_lo", "ratio_hi",
                   "n_lower", "n_landscapes", "wilcoxon_p", "median_ratio")
FAMILIES_SECTION = "== every achievable-unit estimator side by side"


# ---------------------------------------------------------------------------
# the published pipeline, parameterised by directory
# ---------------------------------------------------------------------------


def run_pipeline(roots: dict[str, Path], out_roots: dict[str, Path]) -> tuple[str, str]:
    """oracle_isolation._analyse_one (both responses) per family, then
    oracle_achievable.main and oracle_families.main on the written files.
    Returns the printouts of the last two. ``roots`` hold manifest.csv and runs/;
    ``out_roots`` receive the files and must hold a manifest.csv (the oracle
    description oracle_families reads); the selection files are the published arms'."""
    for f in FAMILIES:
        with contextlib.redirect_stdout(io.StringIO()):
            for suffix, response in iso.RESPONSES.items():
                iso._analyse_one(response, suffix, root=roots[f], out_dir=out_roots[f])
    achievable = io.StringIO()
    with contextlib.redirect_stdout(achievable):
        oa.main(roots={OA_LABEL[f]: roots[f] for f in FAMILIES},
                out_roots={OA_LABEL[f]: out_roots[f] for f in FAMILIES})
    families = io.StringIO()
    with contextlib.redirect_stdout(families):
        of.main(roots={OF_LABEL[f]: out_roots[f] for f in FAMILIES},
                selection_roots={OF_LABEL[f]: PUBLISHED[f] for f in FAMILIES})
    return achievable.getvalue(), families.getvalue()


def written_files(directory: Path) -> list[str]:
    return sorted(p.name for p in Path(directory).glob("oracle_isolation_*.csv")
                  if p.name.startswith(("oracle_isolation_per_landscape", "oracle_isolation_summary")))


def _numeric_gap(a: Path, b: Path) -> str:
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return "columns or rows differ"
    num = x.select_dtypes("number").columns
    gap = float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))) if len(num) else 0.0
    return f"max abs difference {gap:.3g}"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def control(scratch: Path) -> list[str]:
    """Run the pipeline on the published arms into ``scratch``; return the lines
    of the control report. Raises SystemExit if anything is not reproduced."""
    outs = {}
    for f in FAMILIES:
        outs[f] = scratch / f
        outs[f].mkdir(parents=True, exist_ok=True)
        shutil.copy2(PUBLISHED[f] / "manifest.csv", outs[f] / "manifest.csv")
    achievable, families = run_pipeline(PUBLISHED, outs)
    lines, failures = [], []
    for f in FAMILIES:
        names = written_files(outs[f])
        same = [n for n in names if (PUBLISHED[f] / n).is_file()
                and (outs[f] / n).read_bytes() == (PUBLISHED[f] / n).read_bytes()]
        for n in names:
            if n not in same:
                failures.append(f"{SHORT[f]} {n}: " + (_numeric_gap(outs[f] / n, PUBLISHED[f] / n)
                                                      if (PUBLISHED[f] / n).is_file() else "no published file"))
        if len(names) != 24:
            failures.append(f"{SHORT[f]}: {len(names)} files written, expected 24")
        lines.append(f"   {SHORT[f]:4s} {len(same)}/{len(names)} analysis files byte-identical to "
                     f"{PUBLISHED[f].relative_to(REPO).as_posix()}/")
    for name, text in (("oracle_achievable.txt", achievable), ("oracle_families.txt", families)):
        ok = text == _text(REGISTER / name)
        lines.append(f"   printout {'equals' if ok else 'DIFFERS FROM'} register_checks/{name} "
                     f"({len(text.splitlines())} lines)")
        if not ok:
            failures.append(f"printout differs from {name}")
    if failures:
        raise SystemExit("CONTROL FAILED: the pipeline does not reproduce the published arms:\n  "
                         + "\n  ".join(failures))
    return lines


# ---------------------------------------------------------------------------
# completeness of the per-seed arms
# ---------------------------------------------------------------------------


def missing_runs(families=FAMILIES) -> list[str]:
    """Everything that keeps the per-seed arms from being analysed, as readable lines."""
    problems = []
    for f in families:
        root = PERSEED[f]
        if not (root / iso.SEEDS_MANIFEST).is_file() or not (root / "manifest.csv").is_file():
            problems.append(f"{f}: no {iso.SEEDS_MANIFEST}/manifest.csv (run_oracle_isolation_perseed.py manifest)")
            continue
        manifest = drv.load_manifest(f)
        sigma_f = iso.seed_sigma_f(root)
        for landscape, block in manifest.groupby("landscape"):
            todo = [int(s) for s in block.seed if not (drv.run_dir(f, landscape) / f".done_seed{s}").is_file()]
            if todo:
                problems.append(f"{f} {landscape}: seeds {todo} not run")
                continue
            path = drv.run_dir(f, landscape) / "evaluation" / "paired_excess_metrics.csv"
            if not drv.done_marker(f, landscape).is_file() or not path.is_file():
                problems.append(f"{f} {landscape}: not evaluated")
                continue
            runs = pd.read_csv(path)
            runs = runs[(runs.jitter_iteration == 0) & runs.seed.isin(iso.SEEDS)]
            expected = len(iso.SEEDS) * len(drv.ACQ_LIST) * len(iso.SIGMA_GRID)
            if len(runs) != expected:
                problems.append(f"{f} {landscape}: {len(runs)} evaluated pairs, expected {expected}")
                continue
            try:
                mult = iso.sigma_multiples(runs, sigma_f.loc[landscape])
            except ValueError as exc:
                problems.append(f"{f} {landscape}: {exc}")
                continue
            counts = runs.assign(m=mult).groupby(["seed", "m"]).size()
            if len(counts) != len(iso.SEEDS) * len(iso.SIGMA_GRID) or (counts != len(drv.ACQ_LIST)).any():
                problems.append(f"{f} {landscape}: runs per (seed, magnitude) are not {len(drv.ACQ_LIST)} each")
    return problems


# ---------------------------------------------------------------------------
# the side-by-side table
# ---------------------------------------------------------------------------


def summary_file(directory: Path, unit: str, suffix: str, osuffix: str) -> Path:
    return Path(directory) / f"oracle_isolation_summary{unit}{suffix}{osuffix}.csv"


def side_by_side(arms: dict[str, dict[str, Path]]) -> pd.DataFrame:
    """One row per (response, normaliser, family, magnitude) with the summary
    columns of each arm (``arms`` maps an arm label to {family: directory})."""
    rows = []
    for suffix, response in RESPONSES:
        for label, unit, osuffix in NORMALISERS:
            for f in FAMILIES:
                frames = {arm: pd.read_csv(summary_file(dirs[f], unit, suffix, osuffix)).set_index("sigma_multiple")
                          for arm, dirs in arms.items()}
                sigmas = sorted(set.intersection(*(set(fr.index) for fr in frames.values())))
                for s in sigmas:
                    row = {"response": response, "normaliser": label, "family": SHORT[f], "sigma_multiple": s}
                    for arm, fr in frames.items():
                        for col in SUMMARY_COLUMNS:
                            row[f"{arm}_{col}"] = fr.loc[s, col]
                    rows.append(row)
    return pd.DataFrame(rows)


def _cell(r, arm: str) -> str:
    g = lambda c: r[f"{arm}_{c}"]  # noqa: E731
    return (f"{100 * g('frac_fitted'):5.1f}%  {g('ratio_fitted_over_exact'):6.3f} [{g('ratio_lo'):6.3f}, "
            f"{g('ratio_hi'):6.3f}]  {int(g('n_lower')):2d}/{int(g('n_landscapes'))}  p {g('wilcoxon_p'):<6.2g}")


def format_table(table: pd.DataFrame, arms: tuple[str, str]) -> list[str]:
    lines = []
    for (response, label), block in table.groupby(["response", "normaliser"], sort=False):
        lines.append(f"\n-- {response}; {label}")
        lines.append(f"   {'':4s} {'':5s} {'exact':>6s}   {arms[0] + ': fitted, ratio [95%], lower, p':<52s}"
                     f"   {arms[1] + ': fitted, ratio [95%], lower, p'}")
        for r in block.to_dict("records"):
            exact = r[f"{arms[0]}_frac_exact"]
            assert np.isclose(exact, r[f"{arms[1]}_frac_exact"], rtol=0, atol=1e-12), "the exact arm differs"
            lines.append(f"   {r['family']:4s} {r['sigma_multiple']:<5g} {100 * exact:5.1f}%   "
                         f"{_cell(r, arms[0]):<52s}   {_cell(r, arms[1])}")
    return lines


def families_section(text: str) -> str:
    start = text.find(FAMILIES_SECTION)
    if start < 0:
        raise ValueError("no family-contrast section in the oracle_families printout")
    return text[start:].rstrip("\n")


# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control-only", action="store_true", help="run the control on the published arms and stop")
    p.add_argument("--scratch", default=None, help="the control's output directory (default: a temporary one)")
    args = p.parse_args(argv)
    os.chdir(REPO)  # oracle_isolation reads output-boba/ relative to the working directory

    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="oracle_perseed_control_"))
    if scratch.resolve() == REPO or REPO in scratch.resolve().parents:
        raise SystemExit("the control writes outside the repository; choose another --scratch")
    print("== CONTROL: the pipeline on the published arms (seed-7 grid), written to a scratch directory")
    control_lines = control(scratch)
    print("\n".join(control_lines))
    if not args.scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    if args.control_only:
        return

    problems = missing_runs()
    if problems:
        print(f"\nINCOMPLETE: the per-seed runs are not finished ({len(problems)} problems); nothing written.",
              file=sys.stderr)
        for line in problems[:40]:
            print(f"   {line}", file=sys.stderr)
        if len(problems) > 40:
            print(f"   ... and {len(problems) - 40} more", file=sys.stderr)
        print("   python scripts/run_oracle_isolation_perseed.py status", file=sys.stderr)
        raise SystemExit(2)

    achievable, families = run_pipeline(PERSEED, PERSEED)
    arms = ("published", "per_seed")
    table = side_by_side({"published": PUBLISHED, "per_seed": PERSEED})
    COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON, index=False)
    published_families = _text(REGISTER / "oracle_families.txt")
    out = [
        "The oracle-isolation comparison with the error grid scaled per run seed (output-oracle-iso*-perseed)",
        "against the published arms (every seed's grid scaled to the seed-7 oracle). Fitted-oracle over exact",
        "ratio of landscape means [95% landscape bootstrap], landscapes on which the fitted design reports less,",
        "Wilcoxon p over the twenty; gaussian error from the first rating, seeds 7-11, LogEI/qNEI/UCB/EI.",
        "The exact arm (main sweep) and the clean twins are the same in both calibrations.",
        "",
        "== CONTROL: the pipeline on the published arms",
        *control_lines,
        "",
        "== the per-seed arms: complete (20 landscapes x 5 seeds x 6 acquisitions x 3 magnitudes per family, "
        "each seed on its own grid)",
        *format_table(table, arms),
        "",
        "######## family contrasts, published arms (seed-7 grid; register_checks/oracle_families.txt)",
        families_section(published_families),
        "",
        "######## family contrasts, per-seed arms",
        families_section(families),
        "",
        "######## the per-seed arms: scripts/checks/oracle_achievable.py's printout",
        achievable.rstrip("\n"),
        "",
        "######## the per-seed arms: scripts/checks/oracle_families.py's printout",
        families.rstrip("\n"),
    ]
    text = "\n".join(out) + "\n"
    (REGISTER / "oracle_perseed.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
