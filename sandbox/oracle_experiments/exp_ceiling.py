"""exp_ceiling: formalize ACHIEVABLE oracle fidelity per dataset.

For each dataset:
  1. Variance decomposition of the composite y (total / between-user / within-user).
  2. Intra-rater noise SD from output/noise_calibration.csv (preferred_sd logic,
     composite rows): repeat SD if repeat_dof>=10, else close-NN SD if
     n_close_pairs>=30, else all-NN SD (upper bound).
  3. Ceilings:
       cold  max R2 ~= 1 - (sigma_user_intercept^2 + sigma_noise^2) / sigma_total^2
       warm  max R2 ~= 1 - sigma_noise^2 / sigma_within_test^2
              (sigma_within_test = y_test_std from the baseline warm run)
       min RMSE ~= sigma_noise
  4. Empirical anchors through the harness:
       dummy      cold: DummyRegressor (sanity, r2 ~ 0 by construction)
       user_mean  warm: per-user DummyRegressor = predict the mean of the
                  user's TRAIN rows (how much of warm accuracy is just
                  knowing the user's level?)
  5. Baseline numbers from CONTEXT expressed as % of their ceiling.

No hyperparameter selection anywhere; all estimators n_jobs=1, seeded.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import harness

NOISE_CSV = harness.REPO_ROOT / "output" / "noise_calibration.csv"

# Baselines from CONTEXT (deployed extra_trees oracle).
CONTEXT_BASELINES = {
    "ehmi": {"cold_r2": 0.152, "cold_rmse": 0.631, "warm_r2": 0.635, "warm_rmse": 0.416},
    "opticarvis": {"cold_r2": 0.651, "cold_rmse": 0.739, "warm_r2": None, "warm_rmse": 0.922},
    "provoice": {"cold_r2": -0.008, "cold_rmse": 0.955, "warm_r2": -0.321, "warm_rmse": 0.851},
}


def preferred_sd(row: pd.Series) -> tuple[float, str]:
    """Mirror calibrate_noise_from_data.preferred_sd."""
    if row["repeat_dof"] >= 10 and np.isfinite(row["sd_repeat"]):
        return float(row["sd_repeat"]), "exact re-presentations"
    if row["n_close_pairs"] >= 30 and np.isfinite(row["sd_nn_close"]):
        return float(row["sd_nn_close"]), "close NN pairs"
    return float(row["sd_nn_all"]), "all NN pairs (upper bound)"


def noise_sd_for(name: str) -> tuple[float, str]:
    tab = pd.read_csv(NOISE_CSV)
    row = tab[(tab["dataset"] == name) & (tab["objective"] == "composite")].iloc[0]
    return preferred_sd(row)


def variance_decomposition(y: np.ndarray, groups: np.ndarray) -> dict:
    """Total / between-user (bias-corrected) / within-user variance of y."""
    users = np.unique(groups)
    user_means, user_vars, user_ns = [], [], []
    for u in users:
        yu = y[groups == u]
        user_means.append(float(np.mean(yu)))
        user_ns.append(len(yu))
        if len(yu) >= 2:
            user_vars.append(float(np.var(yu, ddof=1)))
    total_var = float(np.var(y, ddof=1))
    within_var = float(np.mean(user_vars))  # mean within-user variance
    mean_n = float(np.mean(user_ns))
    between_raw = float(np.var(user_means, ddof=1))
    # Bias correction: each user mean carries sampling error ~ within_var / n.
    between_corr = max(0.0, between_raw - within_var / mean_n)
    return {
        "n_users": int(len(users)),
        "mean_n_per_user": round(mean_n, 2),
        "total_var": round(total_var, 4),
        "total_sd": round(np.sqrt(total_var), 4),
        "between_user_var_raw": round(between_raw, 4),
        "between_user_var_corrected": round(between_corr, 4),
        "within_user_var": round(within_var, 4),
        "between_share_of_total": round(between_corr / total_var, 4),
    }


def dummy_factory():
    return DummyRegressor(strategy="mean")


def pct(x, ceiling):
    if x is None or ceiling is None or not np.isfinite(ceiling) or ceiling <= 0:
        return None
    return round(100.0 * x / ceiling, 1)


def main() -> None:
    rows = []
    for name in harness.DATASET_NAMES:
        data = harness.load_dataset(name)
        sigma_noise, noise_source = noise_sd_for(name)
        noise_var = sigma_noise**2
        dec = variance_decomposition(data.y, data.groups)

        # --- empirical anchors through the harness ---
        dummy_cold = harness.evaluate_cold(data, dummy_factory)
        # per-user DummyRegressor == predict the mean of the user's TRAIN rows
        user_mean_warm = harness.evaluate_warm_per_user(data, dummy_factory)
        base_cold = harness.evaluate_cold(data, harness.baseline_factory)
        base_warm = harness.evaluate_warm_per_user(data, harness.baseline_factory)

        # --- ceilings ---
        cold_ceiling_r2 = 1.0 - (dec["between_user_var_corrected"] + noise_var) / dec["total_var"]
        warm_test_var = base_warm["y_test_std"] ** 2
        warm_ceiling_r2 = 1.0 - noise_var / warm_test_var

        cb = CONTEXT_BASELINES[name]
        cold_notes = (
            f"sigma_noise={sigma_noise:.3f} ({noise_source}); decomposition: {dec}; "
            f"cold_ceiling_r2={cold_ceiling_r2:.3f}; min_rmse=sigma_noise={sigma_noise:.3f}; "
            f"baseline cold r2 {cb['cold_r2']} = {pct(cb['cold_r2'], cold_ceiling_r2)}% of ceiling; "
            f"baseline cold rmse {cb['cold_rmse']} = {round(100 * sigma_noise / cb['cold_rmse'], 1)}% "
            f"floor/rmse (100%=at noise floor, >100%=below floor)"
        )
        warm_notes = (
            f"sigma_noise={sigma_noise:.3f} ({noise_source}); "
            f"y_test_std(baseline warm)={base_warm['y_test_std']:.3f}; "
            f"warm_ceiling_r2={warm_ceiling_r2:.3f}; min_rmse={sigma_noise:.3f}; "
            f"baseline warm r2 {cb['warm_r2']} = {pct(cb['warm_r2'], warm_ceiling_r2)}% of ceiling; "
            f"baseline warm rmse {cb['warm_rmse']} = {round(100 * sigma_noise / cb['warm_rmse'], 1)}% "
            f"floor/rmse; user_mean warm rmse={user_mean_warm['rmse_pooled']:.3f} "
            f"(baseline improvement over user_mean: "
            f"{round(100 * (1 - base_warm['rmse_pooled'] / user_mean_warm['rmse_pooled']), 1)}% rmse)"
        )

        rows.append({
            "dataset": name, "model": "ceiling", "protocol": "cold",
            "r2_mean_fold": round(cold_ceiling_r2, 4), "rmse_pooled": round(sigma_noise, 4),
            "y_std": dec["total_sd"], "notes": cold_notes,
        })
        rows.append({
            "dataset": name, "model": "ceiling", "protocol": "warm_per_user",
            "r2_pooled": round(warm_ceiling_r2, 4), "rmse_pooled": round(sigma_noise, 4),
            "y_test_std": base_warm["y_test_std"], "notes": warm_notes,
        })
        rows.append({
            "dataset": name, "model": "dummy", **dummy_cold,
            "notes": "pooled-mean DummyRegressor; cold r2 ~ 0 sanity check",
        })
        rows.append({
            "dataset": name, "model": "user_mean", **user_mean_warm,
            "notes": "per-user DummyRegressor = predicts mean of the user's TRAIN rows",
        })
        rows.append({"dataset": name, "model": "extra_trees_rerun", **base_cold})
        rows.append({"dataset": name, "model": "extra_trees_rerun", **base_warm})

    harness.print_results("noise-ceiling", rows)


if __name__ == "__main__":
    main()
