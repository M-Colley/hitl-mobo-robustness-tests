"""The one-shot selection loss beyond the mediation table (Appendix C.2).

Two numbers of the mediation appendix sit outside tables/mediation.tex:

1. where the one-shot predictor falls short. The ``mediator_only`` model of
   analyse_boba_robustness.mediator_model (frag at the cell's magnitude plus the
   onset and error-process indicators, fitted on landscape x condition cell
   means of the post-onset excess in landscape SDs) leaves residuals; their
   correlation across the cells with log opt_z and with log tail weight (the
   descriptors' ``log_tail_ratio``) says on which landscapes fifty noisy choices
   cost more than one predicts;
2. how well it predicts out of sample. Each landscape is left out in turn, the
   model is refitted on the other nineteen, and the left-out landscape's cells
   are predicted; R^2 is 1 - SSE / SST over all held-out cells, SST about the
   grand mean. The same is done for the descriptor model with the magnitude
   (``descriptors_only``: log sigma_e and the four primary descriptors).

Both reuse analyse_boba_robustness's loading, cell means and design (predictors
z-scored within each fit, first indicator level dropped), so the in-sample R^2
printed beside them equals the mediation table's.

    python scripts/review_checks/frag_residuals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as smapi
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import analyse_boba_robustness as abr  # noqa: E402
import boba_benchmarks as bb  # noqa: E402

MODELS = {
    "mediator_only": ["frag_at_c"],
    "descriptors_only": ["log_noise"] + abr.PRIMARY_DESCRIPTORS,
    "magnitude_only": ["log_noise"],
    "indicators_only": [],
}


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """The cells of analyse_boba_robustness.mediator_model."""
    learners = df[~df["acquisition"].isin(abr.MODEL_FREE)].copy()
    return (
        learners.dropna(subset=["frag_at_c"])
        .groupby(["dataset", "error_model", "jitter_std", "jitter_iteration"])
        .agg(excess_sd=("excess_sd", "mean"),
             frag_at_c=("frag_at_c", "first"),
             log_noise=("log_noise", "first"),
             **{name: (name, "first") for name in abr.PRIMARY_DESCRIPTORS})
        .reset_index()
    )


def design(cells: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """z-scored terms and the onset / error-process indicators, as mediator_model builds them."""
    dummies = pd.get_dummies(
        pd.DataFrame({"error_model": cells["error_model"].to_numpy(),
                      "onset": cells["jitter_iteration"].astype(str).to_numpy()}),
        drop_first=True, dtype=float,
    )
    if terms:
        block = cells[terms].astype(float)
        block = (block - block.mean()) / block.std(ddof=0).replace(0.0, 1.0)
        out = pd.concat([block.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    else:
        out = dummies.reset_index(drop=True)
    return smapi.add_constant(out, has_constant="add")


def fit(cells: pd.DataFrame, terms: list[str]):
    X = design(cells, terms)
    return smapi.OLS(cells["excess_sd"].astype(float).to_numpy(), X).fit(), X


def loo_r2(cells: pd.DataFrame, terms: list[str]) -> float:
    """Leave-one-landscape-out R^2; the z-scoring is refitted on the training landscapes."""
    y = cells["excess_sd"].to_numpy(dtype=float)
    pred = np.full(len(cells), np.nan)
    for name in sorted(cells["dataset"].unique()):
        test = (cells["dataset"] == name).to_numpy()
        train_cells = cells[~test].reset_index(drop=True)
        test_cells = cells[test].reset_index(drop=True)
        model, _ = fit(train_cells, terms)
        # standardise the held-out cells with the TRAINING moments
        dummies = pd.get_dummies(
            pd.DataFrame({"error_model": pd.Categorical(test_cells["error_model"],
                                                        categories=sorted(cells["error_model"].unique())),
                          "onset": pd.Categorical(test_cells["jitter_iteration"].astype(str),
                                                  categories=sorted(cells["jitter_iteration"].astype(str).unique()))}),
            drop_first=True, dtype=float,
        )
        if terms:
            mu = train_cells[terms].astype(float).mean()
            sd = train_cells[terms].astype(float).std(ddof=0).replace(0.0, 1.0)
            block = (test_cells[terms].astype(float) - mu) / sd
            Xt = pd.concat([block.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        else:
            Xt = dummies.reset_index(drop=True)
        Xt = smapi.add_constant(Xt, has_constant="add")[model.model.exog_names]
        pred[test] = Xt.to_numpy(dtype=float) @ model.params
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / sst


def main() -> None:
    stats = bb.load_stats(bb.DEFAULT_STATS_PATH)
    df = abr.attach_landscape(abr.load_paired(REPO / "output-boba"), stats)
    cells = cell_means(df)
    print(f"{len(cells)} landscape x condition cells over {cells['dataset'].nunique()} landscapes; "
          "response: post-onset excess in landscape SDs (cell means over the ten model-based acquisitions)")

    print("\n1. residuals of mediator_only (frag + indicators), correlated across the cells")
    model, _ = fit(cells, MODELS["mediator_only"])
    resid = model.resid
    for label, column in (("log opt_z", "log_opt_z"), ("log tail weight", "log_tail_ratio"),
                          ("dimension", "dim"), ("ruggedness", "ruggedness")):
        x = cells[column].to_numpy(dtype=float)
        r, rp = pearsonr(x, resid)
        s, sp = spearmanr(x, resid)
        print(f"   {label:16s} Pearson {r:+.2f} (p {rp:.2g})   Spearman {s:+.2f} (p {sp:.2g})")
    per = pd.DataFrame({"dataset": cells["dataset"], "resid": resid}).groupby("dataset")["resid"].mean()
    land = cells.groupby("dataset")[["log_opt_z", "log_tail_ratio"]].first().loc[per.index]
    for label, column in (("log opt_z", "log_opt_z"), ("log tail weight", "log_tail_ratio")):
        s, sp = spearmanr(land[column], per)
        print(f"   per-landscape mean residual vs {label:16s} Spearman {s:+.2f} (p {sp:.2g}, 20 landscapes)")

    print("\n2. in-sample and leave-one-landscape-out R^2")
    for name, terms in MODELS.items():
        m, _ = fit(cells, terms)
        print(f"   {name:18s} in-sample R^2 {m.rsquared:.3f}   leave-one-landscape-out R^2 {loo_r2(cells, terms):.3f}")


if __name__ == "__main__":
    main()
