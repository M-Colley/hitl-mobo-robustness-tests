"""sigma_f of the oracle every run seed of the oracle-isolation arms actually used.

The simulator refits the oracle for every run seed (build_oracle(seed=seed)),
but oracle_isolation.py calibrate() computes sigma_f, the unit of the error
grid, and the fidelity figure corr_oracle_truth on the seed-7 oracle only, and
run_oracle_isolation.ps1 passes that seed-7 grid to seeds 7-11. This refits
every seed's oracle with the committed code (oracle_isolation.oracle_stats, the
function calibrate() uses; no BO runs), checks that seed 7 reproduces the
manifest, and reports per family:

  - the range of sigma_f(seed) / sigma_f(seed 7) over seeds 8-11,
  - the share of the 80 landscape-seeds (20 landscapes x seeds 8-11) outside
    [0.8, 1.25], and the landscapes they fall on,
  - the magnitude the nominal 0.25/1/5 sigma grid actually had on those seeds
    (nominal multiple / ratio), and
  - the per-seed correlation of the oracle with the true landscape.

It writes the per-seed table to output-oracle-iso/oracle_sigmaf_seeds.csv, with
each oracle's mean over the same points (mean_f), which
scripts/review_checks/oracle_achievable.py uses as the fitted arm's analogue of
the landscape mean behind opt_z.

Each fit runs in a temporary working directory (oracle_isolation.fit_outside_repo):
the two CatBoost tree oracles would otherwise rewrite the tracked catboost_info/
folder at the repository root. The numbers do not depend on it.

    python scripts/review_checks/oracle_sigmaf_seeds.py [--workers 6]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker process

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

FAMILIES = {  # label -> (analysis root, augmentation the arm was run with)
    "tree": ("output-oracle-iso", "jitter"),
    "gaussian_process": ("output-oracle-iso-gaussian_process", "none"),
    "mlp": ("output-oracle-iso-mlp", "none"),
}
BAND = (0.8, 1.25)


def _one_landscape(name: str, models: dict[str, str]) -> list[dict]:
    os.chdir(REPO)
    import oracle_isolation as iso
    rows = []
    for family, (root, augmentation) in FAMILIES.items():
        for seed in iso.SEEDS:
            with contextlib.redirect_stdout(io.StringIO()):
                sigma_f, corr, mean_f = iso.oracle_stats(name, models[family], augmentation, seed=seed)
            rows.append({"family": family, "landscape": name, "oracle_model": models[family],
                         "seed": seed, "sigma_f": sigma_f, "corr_oracle_truth": corr, "mean_f": mean_f})
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    os.chdir(REPO)
    manifests = {f: pd.read_csv(REPO / root / "manifest.csv").set_index("landscape") for f, (root, _) in FAMILIES.items()}
    names = list(manifests["tree"].index)
    tasks = {name: {f: m.loc[name, "oracle_model"] for f, m in manifests.items()} for name in names}
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_one_landscape, tasks, tasks.values()):
            rows.extend(part)
    d = pd.DataFrame(rows).sort_values(["family", "landscape", "seed"]).reset_index(drop=True)
    d["sigma_f_manifest"] = [manifests[f].loc[n, "sigma_f_fitted"] for f, n in zip(d.family, d.landscape)]
    d["ratio_to_seed7"] = d.sigma_f / d.groupby(["family", "landscape"]).sigma_f.transform(lambda s: s.iloc[0])
    d.to_csv(REPO / "output-oracle-iso" / "oracle_sigmaf_seeds.csv", index=False)

    seven = d[d.seed == 7]
    corr_m = [manifests[f].loc[n, "corr_oracle_truth"] for f, n in zip(seven.family, seven.landscape)]
    print(f"seed 7 against the manifests: max |sigma_f - manifest| {float((seven.sigma_f - seven.sigma_f_manifest).abs().max()):.1e}, "
          f"max |corr - manifest| {float(np.max(np.abs(seven.corr_oracle_truth.to_numpy() - np.asarray(corr_m)))):.1e}")
    for family in FAMILIES:
        b = d[(d.family == family) & (d.seed != 7)]
        out = b[(b.ratio_to_seed7 < BAND[0]) | (b.ratio_to_seed7 > BAND[1])]
        where = out.groupby("landscape").ratio_to_seed7.apply(lambda r: "/".join(f"{v:.2f}" for v in r))
        eff = 1.0 / b.ratio_to_seed7
        corr = d[d.family == family].groupby("landscape").corr_oracle_truth
        print(f"\n== {family}: sigma_f(seed)/sigma_f(seed 7) over seeds 8-11 in [{b.ratio_to_seed7.min():.3f}, "
              f"{b.ratio_to_seed7.max():.3f}] (median {b.ratio_to_seed7.median():.3f}); outside [{BAND[0]}, {BAND[1]}] on "
              f"{len(out)}/{len(b)} landscape-seeds ({100 * len(out) / len(b):.1f}%), on {out.landscape.nunique()} landscapes")
        for name, ratios in where.items():
            print(f"     {name:<20s} seeds with ratio outside the band: {ratios}")
        print(f"   the nominal 1 sigma was {eff.min():.2f} to {eff.max():.2f} sigma_f of the oracle actually used "
              f"(so 0.25 sigma was {0.25 * eff.min():.2f}-{0.25 * eff.max():.2f} and 5 sigma {5 * eff.min():.2f}-{5 * eff.max():.2f})")
        span = (corr.max() - corr.min())
        print(f"   corr(oracle, truth) per seed: range over all seeds [{d[d.family == family].corr_oracle_truth.min():.3f}, "
              f"{d[d.family == family].corr_oracle_truth.max():.3f}]; largest within-landscape spread {span.max():.3f} "
              f"({span.idxmax()}); median of the per-landscape seed means {corr.mean().median():.3f}")


if __name__ == "__main__":
    main()
