"""The oracle-isolation result by oracle family, on two normalisers.

The pipeline's model selection picks a tree ensemble on all twenty synthetic
archival datasets (output-oracle-iso). To test whether the understatement is a
property of piecewise-constant oracles, the same datasets were rerun against a
forced Gaussian process and a forced two-layer MLP, without jitter augmentation
(output-oracle-iso-gaussian_process, output-oracle-iso-mlp; see
scripts/oracle_isolation.py --family). Every run seed refits its oracle, so each
normaliser is formed within each seed's oracle and each landscape's fitted
fraction is the ratio of seed means, mean_s(excess_s) / mean_s(normaliser_s).
Two normalisers, each read from the per-landscape files of its producer:

  achievable improvement  (primary; the paper's headline unit) exact excess / opt_z
                          against the fitted mean_s(excess_s) / mean_s(y_opt_s -
                          mean_f,s); scripts/review_checks/oracle_achievable.py,
                          files *_achievable*.csv
  floor gap               (the unit of tab:fitted) excess over the optimum less a
                          same-budget random/Sobol floor; scripts/oracle_isolation.py

The first normaliser cannot approach zero; the second does on Branin, Powell and
Rosenbrock for the exact objective, and it is larger on a tree-ensemble oracle
than on the exact landscape because the oracle's estimated optimum lies a median
17% of its achievable improvement above the best value any clean run reaches
(the floor-share lines of oracle_achievable.py). In both the primary optimum is
each seed's logged y_opt and the sensitivity each seed's best clean value. The
achievable unit has two more sensitivities, because each family sets its own
achievable improvement and the families set it differently (a tree oracle's
y_opt_s - mean_f,s is a median 1.27 times sigma_f,s * opt_z, a GP's or MLP's
about 1):

  sigma_f          the fitted A_s = sigma_f,s * opt_z, the landscape's achievable
                   improvement in the SD units of seed s's oracle, which is also
                   the unit of the injected error
  both_best_clean  each arm's optimum replaced by the best value any clean run of
                   the seed reached (the exact arm over the same six acquisitions)

Compact checks follow (the box mean from the floor runs' designs; y_opt raised
to the best value any run visited; the seed-7 sigma_f in place of each seed's),
and a closing table sets the tree ensembles' ratio and the two family contrasts
side by side under every achievable-unit estimator, so that a claim can be
checked against all of them.

For each family this prints the oracle's fidelity (correlation with the true
landscape at fresh points, grouped cross-validated R2) and how it compares with
the tree ensembles landscape by landscape. Then, per normaliser, optimum
estimator, response (trajectory: post-onset per-iteration excess search regret;
deployed: the shipped design's excess at the final trial) and magnitude:

  ratio      the ratio of landscape means of the fitted over the exact fraction,
             with the analysis's 95% landscape-bootstrap interval
  median     the median per-landscape ratio
  w/o small  the ratio without the landscapes whose exact floor gap is below
             oracle_isolation.SMALL_GAP of opt_z (named), with its interval
  lower      landscapes on which the fitted design reports less, and the
             Wilcoxon signed-rank p over the twenty
  rho        Spearman correlation of the per-landscape costs, fitted vs exact,
             and of fidelity with the per-landscape ratio

and the paired family contrasts GP minus tree and MLP minus tree: the difference
of the ratios of landscape means on the same landscape resamples (the exact
fraction is the same in all three families), and landscape by landscape, the
number of landscapes on which the family's fitted fraction exceeds the tree
ensemble's with the Wilcoxon p. It closes, per normaliser, with the six
human-performance laws alone and with the smooth families restricted to the
landscapes on which their correlation with the truth reaches the tree
ensembles' minimum. The landscapes share their random numbers by design; the
bootstrap treats them as independent. Every interval printed here is checked
against the family's summary file.

    python scripts/review_checks/oracle_achievable.py   # the achievable-unit files
    python scripts/run_review_checks.py --only oracle_families
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import oracle_isolation as iso  # noqa: E402

ROOTS = {
    "tree ensemble (selected)": REPO / "output-oracle-iso",
    "gaussian_process": REPO / "output-oracle-iso-gaussian_process",
    "mlp": REPO / "output-oracle-iso-mlp",
}
SHORT = {"tree ensemble (selected)": "tree", "gaussian_process": "GP", "mlp": "MLP"}
METRICS = (("", "trajectory"), ("_deployed", "deployed"))
# (file infix, label); the first is primary
UNITS = (("_achievable", "share of the achievable improvement (primary: the paper's headline unit)"),
         ("", "fraction of the floor gap (the unit of tab:fitted)"))
# The estimators printed in full, per unit: name -> (fitted column, exact column,
# summary-file suffix, heading); the first of a unit is primary. The checks are compact.
_OPTIMA = {"oracle": ("frac_fitted", "frac_exact", "", "optimum per seed: oracle (primary)"),
           "best_clean": ("frac_fitted_best_clean", "frac_exact", "_best_clean",
                          "optimum per seed: best_clean (sensitivity)")}
FULL = {
    "_achievable": {**_OPTIMA,
                    "sigma_f": ("frac_fitted_sigma_f", "frac_exact", "_sigma_f",
                                "fitted normaliser sigma_f,s * opt_z per seed (sensitivity)"),
                    "both_best_clean": ("frac_fitted_best_clean", "frac_exact_best_clean", "_both_best_clean",
                                        "both arms' optima their best clean value per seed (sensitivity)")},
    "": dict(_OPTIMA),
}
# Every achievable-unit estimator, for the closing table (the full ones and the checks).
ROBUSTNESS = {**{k: v[:3] for k, v in FULL["_achievable"].items()},
              "empirical": ("frac_fitted_empirical", "frac_exact", "_empirical"),
              "visited": ("frac_fitted_visited", "frac_exact", "_visited"),
              "sigma_f_seed7": ("frac_fitted_sigma_f_seed7", "frac_exact", "_sigma_f_seed7")}


def summary_path(root: Path, unit: str, suffix: str, osuffix: str) -> Path:
    return root / f"oracle_isolation_summary{unit}{suffix}{osuffix}.csv"


def compact_check(per: dict[tuple[str, str, str], pd.DataFrame], unit: str, column: str, title: str) -> None:
    """The primary against a check estimator ``column``: ratio, median and count lower."""
    print(f"\n== {title}")
    for f in ROOTS:
        for suffix, label in METRICS:
            b = per[(f, unit, suffix)]
            cells = []
            for s, c in b.groupby("sigma_multiple"):
                r0, r1 = c.frac_fitted.mean() / c.frac_exact.mean(), c[column].mean() / c.frac_exact.mean()
                m0, m1 = np.median(c.frac_fitted / c.frac_exact), np.median(c[column] / c.frac_exact)
                n0, n1 = int((c.frac_fitted < c.frac_exact).sum()), int((c[column] < c.frac_exact).sum())
                cells.append(f"{s:g}s ratio {r0:.3f}->{r1:.3f} median {m0:.3f}->{m1:.3f} lower {n0}->{n1}")
            print(f"   {SHORT[f]:4s} {label:10s} " + "; ".join(cells))


def supremum_check(per: dict[tuple[str, str, str], pd.DataFrame]) -> None:
    """The logged y_opt is a random-search estimate a run can beat. Raising it to
    the best value any run of the seed visited ("visited") against the primary,
    in floor-gap units."""
    for f in ROOTS:
        seeds = per[(f, "", "")].drop_duplicates("landscape")
        print(f"   {SHORT[f]:4s} landscape-seeds on which a run beat y_opt: {int(seeds.n_seeds_above_y_opt.sum())}/"
              f"{len(seeds) * len(iso.SEEDS)}, by at most {seeds.max_visited_over_y_opt.max():.3f} "
              f"(on {seeds.loc[seeds.max_visited_over_y_opt.idxmax(), 'landscape']})")
    compact_check(per, "", "frac_fitted_visited",
                  "floor gap: the logged y_opt against the best value any run visited (optimum 'visited')")


def cv_r2(root: Path) -> pd.Series:
    values = {}
    for path in sorted((root / "selection").glob("iso_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for dataset in payload["datasets"]:
            entry = dataset["objectives"]["composite"]
            values[dataset["name"].replace("iso_", "", 1)] = float(entry["scores"][entry["best_model"]])
    return pd.Series(values, dtype=float)


def _pct(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def fidelity(manifests: dict[str, pd.DataFrame], cvs: dict[str, pd.Series]) -> None:
    tree = next(iter(manifests))
    corr_t = manifests[tree].corr_oracle_truth
    cv_t = cvs[tree]
    floor = float(corr_t.min())
    print(f"\n== fidelity against the tree ensembles (tree minimum correlation {floor:.3f}, "
          f"on {corr_t.idxmin()})")
    for family, m in manifests.items():
        if family == tree:
            continue
        corr = m.corr_oracle_truth.reindex(corr_t.index)
        below = corr[corr < floor].sort_values()
        cv = cvs[family].reindex(cv_t.index)
        print(f"   {SHORT[family]:4s} correlation with the truth: median {corr.median():.3f} (tree {corr_t.median():.3f}); "
              f"below the tree minimum on {len(below)}/20 "
              f"({', '.join(f'{k} {v:.3f}' for k, v in below.items())}); lower than the tree's on "
              f"{int((corr < corr_t).sum())}/20, higher on {int((corr > corr_t).sum())}/20")
        print(f"        grouped CV R2: median {cv.median():.3f} (tree {cv_t.median():.3f}); beats the tree on "
              f"{int((cv > cv_t).sum())}/20; median difference {float((cv - cv_t).median()):+.3f}; "
              f"below 0.1 on {int((cv < 0.1).sum())} ({', '.join(f'{k} {v:.3f}' for k, v in cv[cv < 0.1].sort_values().items())})")


def load_per(root: Path, unit: str, suffix: str) -> pd.DataFrame:
    return pd.read_csv(root / f"oracle_isolation_per_landscape{unit}{suffix}.csv").sort_values(["sigma_multiple", "landscape"])


def family_block(per: dict[str, pd.DataFrame], column: str, unit: str, osuffix: str, suffix: str, label: str,
                 exact_column: str = "frac_exact") -> None:
    fams = list(per)
    sigmas = sorted(per[fams[0]].sigma_multiple.unique())
    blocks = {f: [per[f][per[f].sigma_multiple == s].reset_index(drop=True) for s in sigmas] for f in fams}
    # The same resamples as every family's summary (oracle_isolation.summarise).
    indices = iso.bootstrap_indices([len(b) for b in blocks[fams[0]]])
    keeps = [(b.gap_exact_over_opt_z >= iso.SMALL_GAP).to_numpy() for b in blocks[fams[0]]]
    sub_indices = iso.bootstrap_indices([int(k.sum()) for k in keeps], seed=iso.DATA_SEED + 1)
    print(f"\n   -- {label}")
    for k, s in enumerate(sigmas):
        ref = blocks[fams[0]][k]
        for f in fams[1:]:
            assert blocks[f][k].landscape.tolist() == ref.landscape.tolist()
            assert np.allclose(blocks[f][k][exact_column], ref[exact_column], rtol=0, atol=1e-12), "exact arms differ"
        keep = keeps[k]
        idx, sub = indices[k], sub_indices[k]
        ex = ref[exact_column].to_numpy()
        boot, boot_sub, fit_all = {}, {}, {}
        print(f"      {s:g} sigma  (exact arm {100 * ex.mean():.1f}, mean over landscapes; without "
              f"{', '.join(ref.landscape[~keep])}: exact floor gap below {iso.SMALL_GAP:.0%} of opt_z)")
        for f in fams:
            b = blocks[f][k]
            fit = b[column].to_numpy()
            fit_all[f] = fit
            boot[f] = fit[idx].mean(axis=1) / ex[idx].mean(axis=1)
            fk, ek = fit[keep], ex[keep]
            boot_sub[f] = fk[sub].mean(axis=1) / ek[sub].mean(axis=1)
            ratio = fit.mean() / ex.mean()
            lo, hi = _pct(boot[f])
            slo, shi = _pct(boot_sub[f])
            summary = pd.read_csv(summary_path(ROOTS[f], unit, suffix, osuffix)).set_index("sigma_multiple").loc[s]
            assert max(abs(summary.ratio_fitted_over_exact - ratio), abs(summary.ratio_lo - lo),
                       abs(summary.ratio_hi - hi), abs(summary.ratio_wo_small_gap_lo - slo),
                       abs(summary.ratio_wo_small_gap_hi - shi)) < 1e-12, (f, s, "the summary is not reproduced")
            rho_cost = spearmanr(fit, ex).statistic
            rho_fid = spearmanr(b.corr_oracle_truth, fit / ex).statistic
            print(f"        {SHORT[f]:4s} fitted {100 * fit.mean():6.1f}  ratio {ratio:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"median {np.median(fit / ex):.3f}  w/o small {fk.mean() / ek.mean():.3f} [{slo:.3f}, {shi:.3f}]  "
                  f"lower {int((fit < ex).sum())}/20 (w/o small {int((fk < ek).sum())}/{len(fk)})  "
                  f"p {wilcoxon(fit, ex).pvalue:.2g}  rho(cost) {rho_cost:+.2f}  rho(fidelity, ratio) {rho_fid:+.2f}")
        tree = fams[0]
        for f in fams[1:]:
            d = boot[f] - boot[tree]
            ds = boot_sub[f] - boot_sub[tree]
            point = fit_all[f].mean() / ex.mean() - fit_all[tree].mean() / ex.mean()
            point_s = fit_all[f][keep].mean() / ex[keep].mean() - fit_all[tree][keep].mean() / ex[keep].mean()
            higher = int((fit_all[f] > fit_all[tree]).sum())
            p = wilcoxon(fit_all[f], fit_all[tree]).pvalue
            lo, hi = _pct(d)
            slo, shi = _pct(ds)
            print(f"        {SHORT[f]} - tree: ratio difference {point:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"w/o small {point_s:+.3f} [{slo:+.3f}, {shi:+.3f}]  "
                  f"fitted fraction higher than the tree's on {higher}/20 landscapes (Wilcoxon p {p:.2g})")


def six_laws(per: dict[tuple[str, str, str], pd.DataFrame], unit: str) -> None:
    """The six human-performance laws alone: fitted and exact fraction (means over the six)."""
    from smooth_subset_and_gp_diagnostics import HUMAN_LAWS
    print(f"\n   the six human-performance laws ({', '.join(HUMAN_LAWS)}): mean fitted / mean exact fraction, "
          f"and their ratio")
    for how, (column, exact_column, _, _) in FULL[unit].items():
        for suffix, label in METRICS:
            parts = []
            for f in ROOTS:
                b = per[(f, unit, suffix)]
                b = b[b.landscape.isin(HUMAN_LAWS)]
                assert b.landscape.nunique() == len(HUMAN_LAWS)
                cells = b.groupby("sigma_multiple")[[column, exact_column]].mean()
                parts.append(f"{SHORT[f]} " + ", ".join(
                    f"{s:g}s {r.iloc[0]:.3f}/{r.iloc[1]:.3f} ({r.iloc[0] / r.iloc[1]:.2f})" for s, r in cells.iterrows()))
            print(f"   {how:10s} {label:10s} " + " | ".join(parts))


def fidelity_matched(per: dict[tuple[str, str, str], pd.DataFrame], unit: str, floor: float) -> None:
    """The smooth families on the landscapes where they are at least as faithful as the worst tree."""
    print(f"\n   smooth families restricted to landscapes with correlation >= {floor:.3f} (the tree minimum), "
          f"primary optimum, 1 sigma")
    for suffix, label in METRICS:
        for f in list(ROOTS)[1:]:
            b = per[(f, unit, suffix)]
            b = b[(b.sigma_multiple == 1.0) & (b.corr_oracle_truth >= floor)]
            print(f"   {label:10s} {SHORT[f]:4s} {len(b)} landscapes: ratio {b.frac_fitted.mean() / b.frac_exact.mean():.3f}, "
                  f"median {np.median(b.frac_fitted / b.frac_exact):.3f}, lower {int((b.frac_fitted < b.frac_exact).sum())}/{len(b)}")


def main() -> None:
    manifests, cvs = {}, {}
    for family, root in ROOTS.items():
        manifest = root / "manifest.csv"
        if not manifest.is_file():
            print(f"== {family}: no manifest yet")
            return
        m = pd.read_csv(manifest).set_index("landscape")
        manifests[family], cvs[family] = m, cv_r2(root)
        r2 = cvs[family]
        print(f"== {family}: {len(m)} landscapes; models {m.oracle_model.value_counts().to_dict()}")
        print(f"   fidelity median {m.corr_oracle_truth.median():.3f} range [{m.corr_oracle_truth.min():.3f}, "
              f"{m.corr_oracle_truth.max():.3f}]; sigma_f (seed 7) range [{m.sigma_f_fitted.min():.2f}, "
              f"{m.sigma_f_fitted.max():.2f}]; grouped CV R2 median {r2.median():.3f} range [{r2.min():.3f}, {r2.max():.3f}]")
    fidelity(manifests, cvs)
    tree_floor = float(manifests[next(iter(ROOTS))].corr_oracle_truth.min())

    per = {(f, unit, suffix): load_per(root, unit, suffix)
           for f, root in ROOTS.items() for unit, _ in UNITS for suffix, _ in METRICS}
    own_achievable(per)
    for unit, unit_label in UNITS:
        print(f"\n######## {unit_label}")
        for how, (column, exact_column, osuffix, heading) in FULL[unit].items():
            print(f"\n== {unit_label.split(' (')[0]}; {heading}")
            for suffix, label in METRICS:
                family_block({f: per[(f, unit, suffix)] for f in ROOTS}, column,
                             unit, osuffix, suffix, label, exact_column)
        six_laws(per, unit)
        fidelity_matched(per, unit, tree_floor)
    compact_check(per, "_achievable", "frac_fitted_empirical",
                  "achievable improvement: the oracle mean over 20,000 points against the mean over the "
                  "100 clean floor designs per seed (optimum 'empirical')")
    compact_check(per, "_achievable", "frac_fitted_visited",
                  "achievable improvement: the logged y_opt against the best value any run visited (optimum 'visited')")
    compact_check(per, "_achievable", "frac_fitted_sigma_f_seed7",
                  "achievable improvement: the primary against sigma_f * opt_z with the seed-7 sigma_f, the one "
                  "that scales every seed's error grid ('sigma_f_seed7')")
    print("\n== floor gap: runs above the logged y_opt")
    supremum_check(per)
    robustness(per)


def own_achievable(per: dict[tuple[str, str, str], pd.DataFrame]) -> None:
    """How each family sets its own achievable improvement, against the landscape's in its sigma_f units."""
    print("\n== each family's own achievable improvement (y_opt_s - mean_f,s, mean over seeds) over "
          "sigma_f,s * opt_z (the landscape's achievable improvement in the oracle's SD units)")
    for f in ROOTS:
        b = per[(f, "_achievable", "")]
        b = b[b.sigma_multiple == 1.0]
        q = b.achievable_fitted_over_sigma_f_opt_z
        bc = b.achievable_fitted_best_clean / b.achievable_fitted_sigma_f
        print(f"   {SHORT[f]:4s} median {q.median():.3f}, range [{q.min():.3f}, {q.max():.3f}], above 1 on "
              f"{int((q > 1).sum())}/{len(q)}; with the best clean value as the optimum median {bc.median():.3f}")
    b = per[(next(iter(ROOTS)), "_achievable", "")]
    b = b[b.sigma_multiple == 1.0]
    q = b.achievable_exact_best_clean / b.opt_z
    print(f"   exact arm: best clean value over opt_z median {q.median():.3f}, range [{q.min():.3f}, {q.max():.3f}]")


def robustness(per: dict[tuple[str, str, str], pd.DataFrame]) -> None:
    """The tree ensembles' ratio and the paired family contrasts under every
    achievable-unit estimator, on the resamples of the summaries."""
    fams = list(ROOTS)
    tree = fams[0]
    print("\n== every achievable-unit estimator side by side: ratio of landscape means [95% landscape-bootstrap "
          "interval]; family contrasts are differences of ratios on the same resamples")
    for suffix, label in METRICS:
        frames = {f: per[(f, "_achievable", suffix)] for f in fams}
        sigmas = sorted(frames[tree].sigma_multiple.unique())
        blocks = {f: [frames[f][frames[f].sigma_multiple == s].reset_index(drop=True) for s in sigmas] for f in fams}
        indices = iso.bootstrap_indices([len(b) for b in blocks[tree]])
        for k, s in enumerate(sigmas):
            print(f"   {label} {s:g} sigma")
            spans = {key: [] for key in ("tree", "GP", "MLP", "GP - tree", "MLP - tree")}
            for how, (column, exact_column, osuffix) in ROBUSTNESS.items():
                ex = blocks[tree][k][exact_column].to_numpy()
                idx = indices[k]
                boot, point, cells = {}, {}, []
                for f in fams:
                    assert blocks[f][k].landscape.tolist() == blocks[tree][k].landscape.tolist()
                    fit = blocks[f][k][column].to_numpy()
                    boot[f] = fit[idx].mean(axis=1) / ex[idx].mean(axis=1)
                    point[f] = fit.mean() / ex.mean()
                    lo, hi = _pct(boot[f])
                    summary = pd.read_csv(summary_path(ROOTS[f], "_achievable", suffix, osuffix)).set_index(
                        "sigma_multiple").loc[s]
                    assert max(abs(summary.ratio_fitted_over_exact - point[f]), abs(summary.ratio_lo - lo),
                               abs(summary.ratio_hi - hi)) < 1e-12, (f, s, how, "the summary is not reproduced")
                    cells.append(f"{SHORT[f]} {point[f]:.3f} [{lo:.3f}, {hi:.3f}]")
                    if how in FULL["_achievable"]:
                        spans[SHORT[f]].append((point[f], lo, hi))
                for f in fams[1:]:
                    lo, hi = _pct(boot[f] - boot[tree])
                    d = point[f] - point[tree]
                    cells.append(f"{SHORT[f]}-tree {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                    if how in FULL["_achievable"]:
                        spans[f"{SHORT[f]} - tree"].append((d, lo, hi))
                print(f"      {how:15s} " + "  ".join(cells))
            n = len(FULL["_achievable"])
            parts = []
            for key, vals in spans.items():
                pts = [v[0] for v in vals]
                if key in ("tree", "GP", "MLP"):
                    below = sum(v[2] < 1 for v in vals)
                    above = sum(v[1] > 1 for v in vals)
                    parts.append(f"{key} {min(pts):.2f} to {max(pts):.2f} (interval below 1 in {below}/{n}, "
                                 f"above 1 in {above}/{n})")
                else:
                    pos = sum(v[1] > 0 for v in vals)
                    neg = sum(v[2] < 0 for v in vals)
                    parts.append(f"{key} {min(pts):+.2f} to {max(pts):+.2f} (interval above 0 in {pos}/{n}, "
                                 f"below 0 in {neg}/{n})")
            print(f"      over the {n} estimators printed in full ({', '.join(FULL['_achievable'])}): " + "; ".join(parts))


if __name__ == "__main__":
    main()
