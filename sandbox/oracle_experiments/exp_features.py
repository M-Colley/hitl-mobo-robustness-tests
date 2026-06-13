"""Feature-engineering experiment: row-wise/closed-form perceptual transforms.

All engineered features are row-wise closed-form transforms of the design
parameters (no fitted state), so they are precomputed on the full matrix and
passed via X=. Scalers live inside the per-fit pipelines. Ridge alpha is
selected with an inner GroupKFold over the TRAINING users only (cold); in the
warm per-user protocol each fit sees a single user, so alpha falls back to
efficient leave-one-out RidgeCV on that user's TRAINING rows only.

The group id needed for the nested inner CV is smuggled in as the LAST column
of X for the ridge wrapper (stripped before any fitting/scaling; never used
as a feature). extra_trees is evaluated on the feature matrix WITHOUT that
column.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import harness

ALPHAS = np.logspace(-3, 4, 15)


# --------------------------------------------------------------------------
# Ridge with nested (training-only) alpha selection
# --------------------------------------------------------------------------
class NestedGroupRidge(BaseEstimator, RegressorMixin):
    """Last column of X = numeric group id (stripped, never a feature).

    fit(): if the training rows span >=2 groups, pick alpha by inner
    GroupKFold over the training groups (pooled inner-test RMSE), then refit
    a StandardScaler+Ridge pipeline on all training rows. With a single
    group (warm per-user), use leave-one-out RidgeCV on the training rows.
    Test rows/users are never touched during selection.
    """

    def __init__(self, alphas=ALPHAS):
        self.alphas = alphas

    @staticmethod
    def _pipe(alpha):
        return Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=alpha))])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        g = X[:, -1]
        Xf = X[:, :-1]
        n_groups = len(np.unique(g))
        if n_groups >= 2:
            n_splits = min(3, n_groups)
            gkf = GroupKFold(n_splits=n_splits)
            folds = list(gkf.split(Xf, y, groups=g))
            best_alpha, best_rmse = None, np.inf
            for a in self.alphas:
                sq_errs = []
                for tr, te in folds:
                    m = self._pipe(a).fit(Xf[tr], y[tr])
                    sq_errs.append((y[te] - m.predict(Xf[te])) ** 2)
                rmse = float(np.sqrt(np.mean(np.concatenate(sq_errs))))
                if rmse < best_rmse:
                    best_rmse, best_alpha = rmse, float(a)
            self.alpha_ = best_alpha
            self.model_ = self._pipe(best_alpha).fit(Xf, y)
        else:
            self.model_ = Pipeline(
                [("sc", StandardScaler()), ("ridge", RidgeCV(alphas=self.alphas))]
            ).fit(Xf, y)
            self.alpha_ = float(self.model_.named_steps["ridge"].alpha_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.model_.predict(X[:, :-1])


def ridge_factory():
    return NestedGroupRidge()


# --------------------------------------------------------------------------
# Row-wise closed-form feature builders
# --------------------------------------------------------------------------
def _rgb_to_hsv(r, g, b):
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 0
    rm = mask & (mx == r)
    gm = mask & (mx == g) & ~rm
    bm = mask & ~rm & ~gm
    h[rm] = ((g - b)[rm] / diff[rm]) % 6.0
    h[gm] = (b - r)[gm] / diff[gm] + 2.0
    h[bm] = (r - g)[bm] / diff[bm] + 4.0
    h = h / 6.0
    s = np.where(mx > 0, diff / np.where(mx > 0, mx, 1.0), 0.0)
    return h, s, mx


def col(data, name):
    return data.X[:, data.param_columns.index(name)]


def ehmi_features(data):
    r, g, b, a = (col(data, c) for c in ["r", "g", "b", "a"])
    vw, hw = col(data, "verticalWidth"), col(data, "horizontalWidth")
    blink, vol = col(data, "blinkFrequency"), col(data, "volume")
    h, s, v = _rgb_to_hsv(r, g, b)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    extra = np.column_stack(
        [
            h,
            np.sin(2 * np.pi * h),  # hue is circular
            np.cos(2 * np.pi * h),
            s,
            v,
            lum,
            chroma,
            a * lum,  # alpha-weighted luminance
            vw * hw,  # area
            blink * vol,  # salience interaction
        ]
    )
    percep = np.hstack([data.X, extra])
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly2i = poly.fit_transform(data.X)  # closed-form, no data-dependent state
    return {"percep": percep, "poly2i": poly2i}


OPTI_PAIRS = [
    ("Trajectory", "TrajectoryAlpha"),
    ("Trajectory", "TrajectorySize"),
    ("EgoTrajectory", "EgoTrajectoryAlpha"),
    ("EgoTrajectory", "EgoTrajectorySize"),
    ("PedestrianIntention", "PedestrianIntentionSize"),
    ("SemanticSegmentation", "SemanticSegmentationAlpha"),
    ("CarStatus", "CarStatusAlpha"),
    ("CoveredArea", "CoveredAreaAlpha"),
    ("CoveredArea", "CoveredAreaSize"),
]
OPTI_ELEMENTS = [
    "Trajectory",
    "EgoTrajectory",
    "PedestrianIntention",
    "SemanticSegmentation",
    "CarStatus",
    "CoveredArea",
    "OccludedCars",
]


def opticarvis_features(data):
    prods = np.column_stack([col(data, e) * col(data, m) for e, m in OPTI_PAIRS])
    n_enabled = np.sum(
        np.column_stack([col(data, e) > 0.5 for e in OPTI_ELEMENTS]), axis=1
    ).astype(float)
    vis = np.hstack([data.X, prods, n_enabled[:, None]])
    return {"vis": vis}


def provoice_features(data):
    L = col(data, "InterventionLightingTransparency")
    V = col(data, "AuditoryAlertVolume")  # min 0.1 -> safe denominator
    S = col(data, "InterventionSymbolTransparency")  # min 0.1 -> safe denominator
    loa = col(data, "LevelOfAutonomy")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly2 = poly.fit_transform(data.X)
    ratio = np.hstack(
        [
            data.X,
            np.column_stack(
                [L * V, L * S, V * S, L * V * S, L / V, L / S, V / S, S / V]
            ),
        ]
    )
    # LevelOfAutonomy discreteness check: 85 unique values in 532 rows
    # (continuous BO suggestions clustered at ~15 repeated points) -> NOT a
    # small categorical; exact one-hot skipped. Fallback: fixed 5-bin one-hot.
    n_uniq_loa = len(np.unique(loa))
    edges = np.array([0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(loa, edges)
    onehot = (bins[:, None] == np.arange(5)[None, :]).astype(float)
    poly2_loabin = np.hstack([poly2, onehot])
    print(f"provoice LevelOfAutonomy unique values: {n_uniq_loa} (treated as continuous)")
    return {"poly2": poly2, "ratio": ratio, "poly2_loa5bin": poly2_loabin}


FEATURE_BUILDERS = {
    "ehmi": ehmi_features,
    "opticarvis": opticarvis_features,
    "provoice": provoice_features,
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    rows = []
    for name in harness.DATASET_NAMES:
        data = harness.load_dataset(name)
        # numeric group codes for the ridge wrapper's nested inner CV
        _, gcodes = np.unique(data.groups, return_inverse=True)
        gcol = gcodes.astype(float)[:, None]
        feats = {"raw": data.X}
        feats.update(FEATURE_BUILDERS[name](data))
        for feat_name, Xf in feats.items():
            Xg = np.hstack([Xf, gcol])
            for model_name, factory, Xuse in [
                ("extra_trees", harness.baseline_factory, Xf),
                ("ridge", ridge_factory, Xg),
            ]:
                tag = f"{model_name}+{feat_name}"
                cold = harness.evaluate_cold(data, factory, X=Xuse)
                warm = harness.evaluate_warm_per_user(data, factory, X=Xuse)
                rows.append(
                    {"dataset": name, "model": tag, "features": feat_name,
                     "n_features": int(Xf.shape[1]), **cold}
                )
                rows.append(
                    {"dataset": name, "model": tag, "features": feat_name,
                     "n_features": int(Xf.shape[1]), **warm}
                )
                print(
                    f"done {name} {tag}: cold r2={cold['r2_mean_fold']:.3f} "
                    f"rmse={cold['rmse_pooled']:.3f} | warm rmse={warm['rmse_pooled']:.3f}",
                    flush=True,
                )
    harness.print_results("feature-engineering", rows)


if __name__ == "__main__":
    main()
