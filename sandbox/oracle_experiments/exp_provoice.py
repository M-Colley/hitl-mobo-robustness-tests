"""provoice probe: diagnose why cold r2 ~ 0 / warm rmse 0.85, then test fixes.

DIAGNOSIS (pandas, see findings in output header):
  D1. Group bug: 38 session files (19 participants x 2 conditions) are merged
      into 19 groups by infer_oracle_groups (User_ID). Worse, rglob returns
      'Condition 2' files BEFORE 'Condition 1' files, so within each merged
      group the row order is [cond2 rows..., cond1 rows...] while timestamps
      show condition 1 was run first. The warm "temporal" split therefore
      trains on the participant's SECOND session plus the early first session
      and tests on the late FIRST session (test rows are 0% condition 2).
  D2. Scale bug: composite = mean(Predictability, 'Percieved Usefulness',
      -Mental Demand) without scale normalization. Pred/Useful are 1-5;
      Mental Demand is 1-17 observed (0-20 NASA-TLX-style scale). Signed
      variances 1.08 / 0.89 / 8.24 -> -Mental Demand is 80.7% of composite
      variance (r = 0.965 with composite). The composite is effectively
      negative mental demand.
  D3. Shared initial design: 304/532 rows (57%) are 'sampling' phase with only
      9 distinct X configs (7 identical across all 38 files). 64.8% of
      composite variance lies WITHIN identical-X rows -> params-only pooled
      R^2 ceiling ~0.35 cold. Between-user share 24.6%, between-session
      (user x condition) share 33.7%.
  D4. Params all vary fine within session (13-18 unique values per group per
      param); LevelOfAutonomy is continuous in [0,1] (85 unique values), not
      discrete. Ratings drift upward with iteration (mean per-group
      corr(iter, y) = 0.33) -- BO convergence, so late-session test rows are
      distribution-shifted.

FIXES EVALUATED THROUGH THE HARNESS (this file):
  F1 (warm). Session groups: redefine groups as user x condition (38 sessions)
      so the warm split is genuinely early->late within one session. Valid for
      warm only -- the oracle's deployed job is within-session prediction.
      NOT applied to cold (same participant would appear in train+test folds).
  F2 (warm). Condition one-hot feature: within a session the condition is
      known context at prediction time, so it is legitimate warm input.
      Not used for cold (unseen-user protocol keeps X = design params only,
      per the experiment brief).
  F3 (cold+warm). Per-construct models averaged into the composite: one
      ExtraTrees per signed construct, predictions averaged. Implemented as a
      wrapper estimator that receives a row-index column (column 0 of X) to
      look up the train rows' construct targets; only train-row targets are
      ever used in fit, and the index column is never used as a feature.
  F4 (cold, exploratory). Scale-normalized composite target:
      mean((P-1)/4, (U-1)/4, 1 - MD/20). RMSE not comparable to baseline /
      noise floor (different units); r2 shows whether de-biasing the
      MD dominance makes the target more learnable.

The only hyperparameter selection (min_samples_leaf for one cold variant) is
nested INSIDE the training folds via an inner GroupKFold over training users
(GroupTunedET below); everything else uses the fixed deployed ExtraTrees
family or a fixed-alpha Ridge sanity check.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import harness

SEED = harness.SEED


def et_factory():
    return ExtraTreesRegressor(
        n_estimators=600, random_state=SEED, min_samples_leaf=2, n_jobs=1
    )


class ConstructAverager:
    """Predict the composite as the mean of per-construct model predictions.

    X must carry a row-index in column 0 (features are columns 1:). fit()
    uses the index only to look up the TRAIN rows' construct targets from the
    precomputed (row-wise) construct matrix; predict() never touches it.
    """

    def __init__(self, construct_Y: np.ndarray, base_factory):
        self.construct_Y = construct_Y
        self.base_factory = base_factory
        self.models_: list | None = None

    def fit(self, X, y=None):
        idx = np.asarray(X[:, 0], dtype=int)
        F = X[:, 1:]
        self.models_ = []
        for j in range(self.construct_Y.shape[1]):
            m = self.base_factory()
            m.fit(F, self.construct_Y[idx, j])
            self.models_.append(m)
        return self

    def predict(self, X):
        F = X[:, 1:]
        preds = np.column_stack([m.predict(F) for m in self.models_])
        return preds.mean(axis=1)


class GroupTunedET:
    """ExtraTrees with min_samples_leaf selected by an inner GroupKFold over
    the TRAINING users only (nested selection; test rows/users never seen).

    X must carry a row-index in column 0 so the wrapper can look up the train
    rows' group labels; the index column is never used as a feature.
    """

    def __init__(self, groups: np.ndarray, leaf_grid=(2, 8, 32, 64), n_inner: int = 3):
        self.groups = groups
        self.leaf_grid = leaf_grid
        self.n_inner = n_inner
        self.model_ = None
        self.best_leaf_ = None

    def _make(self, leaf: int) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=600, random_state=SEED, min_samples_leaf=leaf, n_jobs=1
        )

    def fit(self, X, y):
        idx = np.asarray(X[:, 0], dtype=int)
        F = X[:, 1:]
        g = self.groups[idx]
        splits = min(self.n_inner, len(np.unique(g)))
        gkf = GroupKFold(n_splits=splits)
        best_rmse, best_leaf = np.inf, self.leaf_grid[0]
        for leaf in self.leaf_grid:
            errs = []
            for tr, va in gkf.split(F, y, groups=g):
                m = self._make(leaf)
                m.fit(F[tr], y[tr])
                errs.append(mean_squared_error(y[va], m.predict(F[va])))
            rmse = float(np.sqrt(np.mean(errs)))
            if rmse < best_rmse:
                best_rmse, best_leaf = rmse, leaf
        self.best_leaf_ = best_leaf
        self.model_ = self._make(best_leaf)
        self.model_.fit(F, y)
        return self

    def predict(self, X):
        return self.model_.predict(X[:, 1:])


def main() -> None:
    data = harness.load_dataset("provoice")
    df = data.df
    n = len(data.y)
    assert np.allclose(data.y, data.construct_Y.mean(axis=1)), "composite != mean of signed constructs"

    cond = df["Condition_ID"].to_numpy()
    cond2 = (cond == 2).astype(float)
    iteration = df["Iteration"].to_numpy(dtype=float)
    row_idx = np.arange(n, dtype=float)

    # Feature variants (all row-wise / closed-form, no fitted state)
    X_base = data.X
    X_idx = np.column_stack([row_idx, X_base])
    X_cond = np.column_stack([X_base, cond2])
    X_idx_cond = np.column_stack([row_idx, X_base, cond2])
    X_iter = np.column_stack([X_base, iteration])
    X_idx_iter = np.column_stack([row_idx, X_base, iteration])

    # F1: groups = one BO session per source file (38 sessions, 13-15 rows
    # each). NOTE: user x Condition_ID would give only 37 sessions because
    # P17's 'Condition 2' file (17_2) carries a mislabeled Condition_ID of 1
    # in the raw CSV -- the file path is the reliable session identifier.
    session_groups = df["__source_file"].astype(str).to_numpy()
    data_sess = dataclasses.replace(data, groups=session_groups)

    # F4: scale-normalized constructs/composite (theoretical ranges:
    # Pred/Useful 1-5 -> (x-1)/4; Mental Demand 0-20 NASA-TLX -> x/20, negated)
    P = df["Predictability"].to_numpy(dtype=float)
    U = df["Percieved Usefulness"].to_numpy(dtype=float)
    MD = df["Mental Demand"].to_numpy(dtype=float)
    cons_rescaled = np.column_stack([(P - 1) / 4.0, (U - 1) / 4.0, 1.0 - MD / 20.0])
    y_rescaled = cons_rescaled.mean(axis=1)

    rows: list[dict] = []

    def add(model_name: str, res: dict, note: str = "") -> None:
        row = {"dataset": "provoice", "model": model_name, **res}
        if note:
            row["note"] = note
        rows.append(row)
        print(f"[done] {model_name} | {res.get('protocol')} | "
              f"r2={res.get('r2_mean_fold', res.get('r2_pooled')):.3f} rmse={res['rmse_pooled']:.3f}")

    # ---------------- COLD ----------------
    add("extra_trees_baseline", harness.evaluate_cold(data, harness.baseline_factory),
        "deployed oracle, reference")
    add("ridge_sanity", harness.evaluate_cold(
        data, lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "fixed alpha, scaler inside pipeline (train-fold fit only)")
    add("construct_avg_extra_trees", harness.evaluate_cold(
        data, lambda: ConstructAverager(data.construct_Y, et_factory), X=X_idx),
        "F3: one ET per signed construct, predictions averaged into composite")
    add("extra_trees_nested_leaf_tuned", harness.evaluate_cold(
        data, lambda: GroupTunedET(data.groups), X=X_idx),
        "min_samples_leaf in {2,8,32,64} selected by inner GroupKFold over training users only")

    # per-construct learnability (diagnostic): same protocol, single-construct targets
    for j, cname in enumerate(data.objective_columns):
        add(f"extra_trees_target={cname}", harness.evaluate_cold(
            data, harness.baseline_factory, y=data.construct_Y[:, j]),
            "diagnostic: single signed construct as target; rmse in construct units")

    add("extra_trees_rescaled_composite", harness.evaluate_cold(
        data, harness.baseline_factory, y=y_rescaled),
        "F4: scale-normalized composite target; rmse NOT comparable to baseline/floor")
    add("construct_avg_rescaled_composite", harness.evaluate_cold(
        data, lambda: ConstructAverager(cons_rescaled, et_factory), X=X_idx, y=y_rescaled),
        "F3+F4: per-construct models on rescaled constructs; rmse NOT comparable")

    # ---------------- WARM ----------------
    add("extra_trees_baseline", harness.evaluate_warm_per_user(data, harness.baseline_factory),
        "reference; merged user groups order cond2-rows before cond1-rows (D1)")
    add("extra_trees_session_groups", harness.evaluate_warm_per_user(data_sess, harness.baseline_factory),
        "F1: groups = user x condition session (38 sessions), true early->late split")
    add("extra_trees_cond_feature", harness.evaluate_warm_per_user(
        data, harness.baseline_factory, X=X_cond),
        "F2: merged groups + condition one-hot feature")
    add("construct_avg_session_groups", harness.evaluate_warm_per_user(
        data_sess, lambda: ConstructAverager(data.construct_Y, et_factory), X=X_idx),
        "F1+F3")
    add("extra_trees_session_groups_iter", harness.evaluate_warm_per_user(
        data_sess, harness.baseline_factory, X=X_iter),
        "F1 + iteration index as known-context feature")
    add("construct_avg_session_groups_iter", harness.evaluate_warm_per_user(
        data_sess, lambda: ConstructAverager(data.construct_Y, et_factory), X=X_idx_iter),
        "F1+F3 + iteration feature")

    add("extra_trees_pooled_onehot", harness.evaluate_warm_pooled(
        data, harness.baseline_factory, user_onehot=True),
        "reference pooled warm, merged user groups")
    add("extra_trees_pooled_onehot_cond", harness.evaluate_warm_pooled(
        data, harness.baseline_factory, user_onehot=True, X=X_cond),
        "F2 pooled: user one-hot + condition feature")
    add("extra_trees_pooled_session_onehot", harness.evaluate_warm_pooled(
        data_sess, harness.baseline_factory, user_onehot=True),
        "F1 pooled: session groups, session one-hot")
    add("extra_trees_pooled_session_onehot_iter", harness.evaluate_warm_pooled(
        data_sess, harness.baseline_factory, user_onehot=True, X=X_iter),
        "F1 pooled + iteration feature (one-hot appended after iteration by the harness)")

    harness.print_results("provoice-probe", rows)


if __name__ == "__main__":
    main()
