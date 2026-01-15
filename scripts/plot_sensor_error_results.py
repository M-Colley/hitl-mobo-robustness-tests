"""Improved statistical evaluation with ANOVA and proper post-hoc tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/plots"))
    return parser.parse_args()


def load_iteration_logs(input_dir: Path) -> pd.DataFrame:
    files = list(input_dir.glob("bo_sensor_error_*_seed*_*.csv"))
    if not files:
        raise FileNotFoundError("No per-iteration logs found in input-dir.")
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def summarize_final_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = (
        df.sort_values("iteration")
        .groupby("run_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    final_rows["baseline"] = final_rows["error_model"] == "none"
    return final_rows



def evaluate_final_outcomes_improved(final_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Comprehensive statistical analysis using ANOVA and appropriate post-hoc tests.
    
    Returns dictionary with:
    - anova_results: Mixed ANOVA results
    - posthoc_results: Tukey HSD post-hoc comparisons
    - effect_sizes: Effect size metrics
    - descriptive_stats: Descriptive statistics by group
    - regret_tests: Paired t-tests for regret metrics
    """
    baseline = final_df[final_df["baseline"]].copy()
    jittered = final_df[~final_df["baseline"]].copy()
    
    merge_keys = ["objective", "acquisition", "seed", "oracle_model"]
    if "dataset" in final_df.columns:
        merge_keys.insert(0, "dataset")

    regret_cols = ["simple_regret_true", "regret_cum_true", "regret_avg_true"]
    baseline_cols = ["objective_true", "objective_observed", *regret_cols]
    merged = jittered.merge(
        baseline[merge_keys + baseline_cols],
        on=merge_keys,
        how="inner",
        suffixes=("_jitter", "_baseline"),
    )
    
    if merged.empty:
        return {}
    
    # Calculate differences for repeated measures
    merged["true_diff"] = merged["objective_true_jitter"] - merged["objective_true_baseline"]
    merged["obs_diff"] = merged["objective_observed_jitter"] - merged["objective_observed_baseline"]
    
    results = {}
    
    # ============================================================================
    # 1. MIXED ANOVA (treating seed as random effect)
    # ============================================================================
    print("\n" + "="*80)
    print("MIXED ANOVA ANALYSIS")
    print("="*80)
    
    for metric in ["true_diff", "obs_diff"]:
        metric_name = "True Objective" if metric == "true_diff" else "Observed Objective"
        print(f"\n{metric_name} Difference:")
        print("-" * 80)
        
        # Check which factors have multiple levels
        factors = {
            'dataset': merged['dataset'].nunique() if 'dataset' in merged.columns else 1,
            'objective': merged['objective'].nunique(),
            'acquisition': merged['acquisition'].nunique(),
            'error_model': merged['error_model'].nunique(),
            'jitter_std': merged['jitter_std'].nunique(),
            'jitter_iteration': merged['jitter_iteration'].nunique(),
            'oracle_model': merged['oracle_model'].nunique(),
        }
        
        print(f"Factor levels: {factors}")
        print(f"Total observations: {len(merged)}")
        
        # Build formula dynamically based on available factors
        valid_factors = [f for f, n in factors.items() if n >= 2]
        
        if len(valid_factors) == 0:
            print("Warning: No factors with 2+ levels. Cannot perform ANOVA.")
            continue
        
        # Check if we have enough observations for interaction terms
        # Rule of thumb: need at least 20 observations per parameter
        total_combinations = np.prod([factors[f] for f in valid_factors])
        min_obs_needed = total_combinations * 2  # At least 2 obs per cell
        
        print(f"Unique factor combinations: {total_combinations}")
        print(f"Minimum observations needed: {min_obs_needed}")
        
        # Start with full model and fall back to simpler models if needed
        models_to_try = []
        
        if len(valid_factors) >= 3 and len(merged) >= min_obs_needed:
            # Try full interaction model
            models_to_try.append(
                (f"{metric} ~ " + " * ".join([f"C({f})" for f in valid_factors]), "full interaction")
            )
        
        if len(valid_factors) >= 2:
            # Try additive model (main effects only)
            models_to_try.append(
                (f"{metric} ~ " + " + ".join([f"C({f})" for f in valid_factors]), "main effects only")
            )
            
            # Try two-way interactions if we have enough data
            if len(valid_factors) == 2 and len(merged) >= min_obs_needed:
                models_to_try.append(
                    (f"{metric} ~ C({valid_factors[0]}) * C({valid_factors[1]})", "two-way interaction")
                )
        
        if len(valid_factors) == 1:
            models_to_try.append(
                (f"{metric} ~ C({valid_factors[0]})", "single factor")
            )
        
        # Try models in order until one works
        anova_success = False
        for formula, description in models_to_try:
            print(f"\nTrying ANOVA: {description}")
            print(f"Formula: {formula}")
            
            try:
                model = ols(formula, data=merged).fit()
                
                # Check for infinite or NaN values
                if not np.all(np.isfinite(model.params)):
                    print("  Warning: Model parameters contain infinite or NaN values. Trying simpler model...")
                    continue
                
                anova_table = anova_lm(model, typ=2)
                
                # Check if ANOVA table is valid
                if anova_table['sum_sq'].isna().all() or not np.all(np.isfinite(anova_table['sum_sq'])):
                    print("  Warning: ANOVA table contains invalid values. Trying simpler model...")
                    continue
                
                # Calculate effect sizes (eta-squared)
                total_ss = anova_table['sum_sq'].sum()
                if total_ss > 0:
                    anova_table['eta_sq'] = anova_table['sum_sq'] / total_ss
                    anova_table['omega_sq'] = (
                        (anova_table['sum_sq'] - anova_table['df'] * model.mse_resid) / 
                        (total_ss + model.mse_resid)
                    )
                else:
                    anova_table['eta_sq'] = np.nan
                    anova_table['omega_sq'] = np.nan
                
                print("\n" + anova_table.to_string())
                
                results[f'anova_{metric}'] = anova_table
                results[f'anova_{metric}_model'] = description
                
                # Interpret main effects
                print("\nEffect Size Interpretation (eta-squared):")
                for idx, row in anova_table.iterrows():
                    if pd.isna(row['PR(>F)']) or pd.isna(row['eta_sq']):
                        continue
                    
                    if row['eta_sq'] >= 0.14:
                        size = "LARGE"
                    elif row['eta_sq'] >= 0.06:
                        size = "MEDIUM"
                    elif row['eta_sq'] >= 0.01:
                        size = "SMALL"
                    else:
                        size = "negligible"
                    
                    if row['PR(>F)'] < 0.001:
                        sig = "***"
                    elif row['PR(>F)'] < 0.01:
                        sig = "**"
                    elif row['PR(>F)'] < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    
                    print(f"  {idx}: η² = {row['eta_sq']:.4f} ({size}) {sig}")
                
                anova_success = True
                break
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if not anova_success:
            print(f"\nCould not fit any ANOVA model for {metric_name}.")
            print("Falling back to separate one-way ANOVAs for each factor...")
            
            # Perform separate one-way ANOVAs for each factor
            oneway_results = []
            for factor in valid_factors:
                print(f"\n  One-way ANOVA for {factor}:")
                groups = [group[metric].dropna().values for name, group in merged.groupby(factor)]
                
                if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        
                        # Calculate eta-squared
                        grand_mean = merged[metric].mean()
                        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
                        ss_total = sum((merged[metric] - grand_mean)**2)
                        eta_sq = ss_between / ss_total if ss_total > 0 else 0
                        
                        if eta_sq >= 0.14:
                            size = "LARGE"
                        elif eta_sq >= 0.06:
                            size = "MEDIUM"
                        elif eta_sq >= 0.01:
                            size = "SMALL"
                        else:
                            size = "negligible"
                        
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        
                        print(f"    F({len(groups)-1}, {len(merged)-len(groups)}) = {f_stat:.4f}, p = {p_val:.4f} {sig}")
                        print(f"    η² = {eta_sq:.4f} ({size})")
                        
                        oneway_results.append({
                            'metric': metric,
                            'factor': factor,
                            'F': f_stat,
                            'p_value': p_val,
                            'eta_sq': eta_sq,
                            'interpretation': size
                        })
                    except Exception as e:
                        print(f"    Error: {e}")
                else:
                    print(f"    Insufficient data for comparison")
            
            if oneway_results:
                results[f'oneway_anova_{metric}'] = pd.DataFrame(oneway_results)
    
    # ============================================================================
    # 2. POST-HOC TESTS (Tukey HSD for pairwise comparisons)
    # ============================================================================
    print("\n" + "="*80)
    print("POST-HOC ANALYSIS (Tukey HSD)")
    print("="*80)
    
    for metric in ["true_diff", "obs_diff"]:
        metric_name = "True Objective" if metric == "true_diff" else "Observed Objective"
        
        # Post-hoc for acquisition strategies
        n_acquisitions = merged['acquisition'].nunique()
        if n_acquisitions >= 2:
            print(f"\n{metric_name} - Pairwise Acquisition Comparisons:")
            print("-" * 80)
            try:
                tukey_acq = pairwise_tukeyhsd(merged[metric], merged['acquisition'])
                print(tukey_acq)
                results[f'tukey_acq_{metric}'] = tukey_acq
            except Exception as e:
                print(f"Could not perform Tukey HSD for acquisitions: {e}")
        else:
            print(f"\n{metric_name} - Skipping acquisition comparisons (only {n_acquisitions} group(s))")
        
        # Post-hoc for error models
        n_error_models = merged['error_model'].nunique()
        if n_error_models >= 2:
            print(f"\n{metric_name} - Pairwise Error Model Comparisons:")
            print("-" * 80)
            try:
                tukey_error = pairwise_tukeyhsd(merged[metric], merged['error_model'])
                print(tukey_error)
                results[f'tukey_error_{metric}'] = tukey_error
            except Exception as e:
                print(f"Could not perform Tukey HSD for error models: {e}")
        else:
            print(f"\n{metric_name} - Skipping error model comparisons (only {n_error_models} group(s))")
        
        # Post-hoc for jitter_std if there are multiple levels
        n_jitter_stds = merged['jitter_std'].nunique()
        if n_jitter_stds >= 2:
            print(f"\n{metric_name} - Pairwise Jitter Std Comparisons:")
            print("-" * 80)
            try:
                tukey_jitter = pairwise_tukeyhsd(merged[metric], merged['jitter_std'])
                print(tukey_jitter)
                results[f'tukey_jitter_{metric}'] = tukey_jitter
            except Exception as e:
                print(f"Could not perform Tukey HSD for jitter_std: {e}")
        else:
            print(f"\n{metric_name} - Skipping jitter_std comparisons (only {n_jitter_stds} level(s))")
    
    # ============================================================================
    # 3. EFFECT SIZES (Cohen's d) for key comparisons
    # ============================================================================
    print("\n" + "="*80)
    print("EFFECT SIZES (Cohen's d)")
    print("="*80)
    
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size."""
        diff = group1 - group2
        pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
        return diff.mean() / (pooled_std + 1e-10)
    
    effect_sizes = []
    for (objective, acq, error_model, jitter_std, jitter_iter, oracle), group in merged.groupby(
        ['objective', 'acquisition', 'error_model', 'jitter_std', 'jitter_iteration', 'oracle_model']
    ):
        d_true = cohens_d(group['objective_true_jitter'], group['objective_true_baseline'])
        d_obs = cohens_d(group['objective_observed_jitter'], group['objective_observed_baseline'])
        
        effect_sizes.append({
            'objective': objective,
            'acquisition': acq,
            'error_model': error_model,
            'jitter_std': jitter_std,
            'jitter_iteration': jitter_iter,
            'oracle_model': oracle,
            'cohens_d_true': d_true,
            'cohens_d_obs': d_obs,
            'n': len(group),
        })
    
    effect_df = pd.DataFrame(effect_sizes)
    
    # Interpret effect sizes
    def interpret_d(d):
        abs_d = abs(d)
        if abs_d >= 0.8:
            return "LARGE"
        elif abs_d >= 0.5:
            return "MEDIUM"
        elif abs_d >= 0.2:
            return "SMALL"
        else:
            return "negligible"
    
    effect_df['interpretation_true'] = effect_df['cohens_d_true'].apply(interpret_d)
    effect_df['interpretation_obs'] = effect_df['cohens_d_obs'].apply(interpret_d)
    
    print("\nEffect Sizes by Condition:")
    print(effect_df.to_string(index=False))
    
    results['effect_sizes'] = effect_df
    
    # ============================================================================
    # 4. DESCRIPTIVE STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    
    desc_stats = merged.groupby(['objective', 'acquisition', 'error_model', 'jitter_std']).agg({
        'true_diff': ['mean', 'std', 'sem', 'count'],
        'obs_diff': ['mean', 'std', 'sem', 'count'],
    }).round(4)
    
    print("\nMean Differences by Condition:")
    print(desc_stats.to_string())
    
    results['descriptive_stats'] = desc_stats
    
    # ============================================================================
    # 5. REGRET METRICS (Paired t-tests)
    # ============================================================================
    print("\n" + "="*80)
    print("REGRET METRICS (Paired t-tests)")
    print("="*80)

    regret_tests = []
    for metric in regret_cols:
        jitter_col = f"{metric}_jitter"
        baseline_col = f"{metric}_baseline"
        if jitter_col not in merged.columns or baseline_col not in merged.columns:
            continue
        paired = merged[[jitter_col, baseline_col]].dropna()
        if len(paired) < 2:
            print(f"Skipping {metric}: insufficient paired samples")
            continue
        t_stat, p_val = ttest_rel(paired[jitter_col], paired[baseline_col])
        mean_diff = float((paired[jitter_col] - paired[baseline_col]).mean())
        regret_tests.append({
            "metric": metric,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "mean_diff": mean_diff,
            "n": int(len(paired)),
        })
        print(f"{metric}: t={t_stat:.4f}, p={p_val:.4f}, mean diff={mean_diff:.4f}")

    if regret_tests:
        results["regret_tests"] = pd.DataFrame(regret_tests)

    # ============================================================================
    # 6. SAVE RESULTS
    # ============================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ANOVA tables
    for key, value in results.items():
        if 'anova' in key and isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{key}.csv")
        elif 'tukey' in key and hasattr(value, 'summary'):
            # Save Tukey results as DataFrame
            summary_df = pd.DataFrame(data=value.summary().data[1:], columns=value.summary().data[0])
            summary_df.to_csv(output_dir / f"{key}.csv", index=False)
    
    # Save effect sizes
    if 'effect_sizes' in results:
        effect_df = results['effect_sizes']
        effect_df.to_csv(output_dir / "effect_sizes_cohens_d.csv", index=False)
    
    # Save descriptive stats
    if 'descriptive_stats' in results:
        desc_stats = results['descriptive_stats']
        desc_stats.to_csv(output_dir / "descriptive_statistics.csv")

    if "regret_tests" in results:
        results["regret_tests"].to_csv(output_dir / "regret_paired_tests.csv", index=False)
    
    # Save one-way ANOVA results if available
    for key in list(results.keys()):
        if 'oneway_anova' in key and isinstance(results[key], pd.DataFrame):
            results[key].to_csv(output_dir / f"{key}.csv", index=False)
    
    # ============================================================================
    # 7. SIMPLE EFFECTS ANALYSIS (if interaction is significant)
    # ============================================================================
    print("\n" + "="*80)
    print("SIMPLE EFFECTS ANALYSIS")
    print("="*80)
    
    # Only perform if we have multiple error models and jitter_std levels
    if merged['error_model'].nunique() >= 2 and merged['jitter_std'].nunique() >= 2:
        # Test effect of error_model at each level of jitter_std
        for jitter_std_val in sorted(merged['jitter_std'].unique()):
            subset = merged[merged['jitter_std'] == jitter_std_val]
            print(f"\nEffect of error_model at jitter_std={jitter_std_val}:")
            
            for metric in ['true_diff', 'obs_diff']:
                groups = [group[metric].dropna().values for name, group in subset.groupby('error_model')]
                # Only perform if we have at least 2 groups with data
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"  {metric}: F={f_stat:.4f}, p={p_val:.4f} {sig}")
                    except Exception as e:
                        print(f"  {metric}: Could not compute F-test ({e})")
                else:
                    print(f"  {metric}: Insufficient groups for comparison")
    else:
        print("\nSkipping simple effects analysis - need multiple error models and jitter_std levels")
    
    print("\n" + "="*80)
    print("Analysis complete. Results saved to:", output_dir)
    print("="*80)
    
    return results


# Additional helper function for reporting
def generate_statistical_report(results: dict, output_dir: Path) -> None:
    """Generate a human-readable statistical report."""
    report_path = output_dir / "statistical_report.txt"
    
    # Use UTF-8 encoding to support Greek letters and other Unicode characters
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Mixed ANOVA with Type II sums of squares (or one-way ANOVAs if data limited)\n")
        f.write("2. Post-hoc pairwise comparisons using Tukey HSD (when applicable)\n")
        f.write("3. Effect sizes: η² (eta-squared) and Cohen's d\n")
        f.write("4. Significance level: α = 0.05\n\n")
        
        f.write("INTERPRETATION GUIDELINES:\n")
        f.write("-" * 80 + "\n")
        f.write("Effect Size (Cohen's d):\n")
        f.write(" - Small: 0.2 ≤ |d| < 0.5\n")
        f.write(" - Medium: 0.5 ≤ |d| < 0.8\n")
        f.write(" - Large: |d| ≥ 0.8\n\n")
        f.write("Effect Size (η²):\n")
        f.write(" - Small: 0.01 ≤ η² < 0.06\n")
        f.write(" - Medium: 0.06 ≤ η² < 0.14\n")
        f.write(" - Large: η² ≥ 0.14\n\n")
        
        # Summarize key findings
        if 'effect_sizes' in results and not results['effect_sizes'].empty:
            effect_df = results['effect_sizes']
            large_effects = effect_df[
                (effect_df['interpretation_true'] == 'LARGE') | 
                (effect_df['interpretation_obs'] == 'LARGE')
            ]
            
            f.write("KEY FINDINGS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total conditions tested: {len(effect_df)}\n")
            f.write(f"Conditions with LARGE effects: {len(large_effects)}\n\n")
            
            if not large_effects.empty:
                f.write("Conditions with largest effects:\n")
                top_effects = large_effects.nlargest(10, 'cohens_d_true')
                f.write(top_effects.to_string(index=False))
                f.write("\n\n")
        else:
            f.write("KEY FINDINGS:\n")
            f.write("-" * 80 + "\n")
            f.write("No effect size data available.\n\n")
        
        # Summary of ANOVA results
        for key in results.keys():
            if 'anova_true_diff' == key or 'anova_obs_diff' == key:
                metric = 'True Objective' if 'true' in key else 'Observed Objective'
                f.write(f"\nANOVA RESULTS - {metric}:\n")
                f.write("-" * 80 + "\n")
                anova_table = results[key]
                f.write(anova_table.to_string())
                f.write("\n")
                
                # Model type used
                model_key = f"{key}_model"
                if model_key in results:
                    f.write(f"\nModel type: {results[model_key]}\n")
        
        # Summary of one-way ANOVAs if used
        for key in results.keys():
            if 'oneway_anova' in key:
                metric = 'True Objective' if 'true' in key else 'Observed Objective'
                f.write(f"\nONE-WAY ANOVA RESULTS - {metric}:\n")
                f.write("-" * 80 + "\n")
                f.write(results[key].to_string(index=False))
                f.write("\n\n")
        
        # Summary of post-hoc tests
        tukey_found = False
        for key in results.keys():
            if 'tukey' in key:
                tukey_found = True
                break
        
        if tukey_found:
            f.write("\nPOST-HOC TEST RESULTS:\n")
            f.write("-" * 80 + "\n")
            f.write("See individual CSV files for detailed Tukey HSD comparisons.\n\n")
        
        # Descriptive statistics summary
        if 'descriptive_stats' in results:
            f.write("\nDESCRIPTIVE STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write("See descriptive_statistics.csv for detailed summaries.\n\n")
        
        # Data quality notes
        f.write("\nDATA QUALITY NOTES:\n")
        f.write("-" * 80 + "\n")
        if 'effect_sizes' in results and not results['effect_sizes'].empty:
            effect_df = results['effect_sizes']
            total_n = effect_df['n'].sum()
            min_n = effect_df['n'].min()
            max_n = effect_df['n'].max()
            mean_n = effect_df['n'].mean()
            
            f.write(f"Total observations: {total_n}\n")
            f.write(f"Sample size per condition: min={min_n}, max={max_n}, mean={mean_n:.1f}\n")
            
            # Check for small sample sizes
            small_n = effect_df[effect_df['n'] < 5]
            if not small_n.empty:
                f.write(f"\nWarning: {len(small_n)} condition(s) have fewer than 5 observations.\n")
                f.write("Results for these conditions should be interpreted with caution.\n")
        else:
            f.write("No data quality information available.\n")
    
    print(f"\nStatistical report saved to: {report_path}")
    
    
    
    
def evaluate_final_outcomes(final_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    baseline = final_df[final_df["baseline"]].copy()
    jittered = final_df[~final_df["baseline"]].copy()
    merged = jittered.merge(
        baseline[
            [
                "objective",
                "acquisition",
                "seed",
                "oracle_model",
                "objective_true",
                "objective_observed",
            ]
        ],
        on=["objective", "acquisition", "seed", "oracle_model"],
        how="inner",
        suffixes=("_jitter", "_baseline"),
    )
    if merged.empty:
        return pd.DataFrame()

    rows = []
    group_cols = [
        "objective",
        "acquisition",
        "error_model",
        "jitter_iteration",
        "jitter_std",
        "oracle_model",
    ]
    for keys, group in merged.groupby(group_cols):
        objective, acquisition, error_model, jitter_iteration, jitter_std, oracle_model = keys
        n_runs = len(group)
        mean_true_diff = float(
            (group["objective_true_jitter"] - group["objective_true_baseline"]).mean()
        )
        mean_obs_diff = float(
            (group["objective_observed_jitter"] - group["objective_observed_baseline"]).mean()
        )
        
        # Calculate Cohen's d effect sizes
        def cohens_d(group1, group2):
            diff = group1 - group2
            return diff.mean() / (diff.std() + 1e-10)  # Add small constant to avoid division by zero
        
        cohens_d_true = float(cohens_d(group["objective_true_jitter"], group["objective_true_baseline"]))
        cohens_d_obs = float(cohens_d(group["objective_observed_jitter"], group["objective_observed_baseline"]))
        
        true_p = float("nan")
        obs_p = float("nan")
        if n_runs >= 2:
            true_test = ttest_rel(
                group["objective_true_jitter"],
                group["objective_true_baseline"],
                nan_policy="omit",
            )
            obs_test = ttest_rel(
                group["objective_observed_jitter"],
                group["objective_observed_baseline"],
                nan_policy="omit",
            )
            true_p = float(true_test.pvalue)
            obs_p = float(obs_test.pvalue)
        rows.append(
            {
                "objective": objective,
                "acquisition": acquisition,
                "error_model": error_model,
                "jitter_iteration": jitter_iteration,
                "jitter_std": jitter_std,
                "oracle_model": oracle_model,
                "runs": n_runs,
                "mean_true_diff": mean_true_diff,
                "mean_observed_diff": mean_obs_diff,
                "cohens_d_true": cohens_d_true,
                "cohens_d_observed": cohens_d_obs,
                "p_value_true": true_p,
                "p_value_observed": obs_p,
            }
        )

    stats = pd.DataFrame(rows)
    
    # Apply Bonferroni correction
    n_tests = len(stats) * 2  # multiply by 2 because we test both true and observed
    stats["p_value_true_bonferroni"] = stats["p_value_true"] * n_tests
    stats["p_value_observed_bonferroni"] = stats["p_value_observed"] * n_tests
    # Cap corrected p-values at 1.0
    stats["p_value_true_bonferroni"] = stats["p_value_true_bonferroni"].clip(upper=1.0)
    stats["p_value_observed_bonferroni"] = stats["p_value_observed_bonferroni"].clip(upper=1.0)
    
    # Add significance flags (alpha = 0.05)
    stats["significant_true_uncorrected"] = stats["p_value_true"] < 0.05
    stats["significant_observed_uncorrected"] = stats["p_value_observed"] < 0.05
    stats["significant_true_bonferroni"] = stats["p_value_true_bonferroni"] < 0.05
    stats["significant_observed_bonferroni"] = stats["p_value_observed_bonferroni"] < 0.05
    
    stats_path = output_dir / "final_outcome_significance.csv"
    stats.to_csv(stats_path, index=False)
    
    # Print summary
    print("\nStatistical Testing Summary:")
    print(f"Total number of comparisons: {len(stats)}")
    print(f"Total number of tests: {n_tests}")
    print(f"Bonferroni-corrected alpha: {0.05 / n_tests:.6f}")
    print(f"\nSignificant results (uncorrected, alpha=0.05):")
    print(f"  True objective: {stats['significant_true_uncorrected'].sum()}/{len(stats)}")
    print(f"  Observed objective: {stats['significant_observed_uncorrected'].sum()}/{len(stats)}")
    print(f"\nSignificant results (Bonferroni-corrected, alpha=0.05):")
    print(f"  True objective: {stats['significant_true_bonferroni'].sum()}/{len(stats)}")
    print(f"  Observed objective: {stats['significant_observed_bonferroni'].sum()}/{len(stats)}")
    
    return stats


def plot_final_outcome_significance(results: dict, output_dir: Path) -> None:
    """Plot significance results from the statistical analysis."""
    if not results or 'effect_sizes' not in results:
        print("No statistical results to plot")
        return
    
    stats = results['effect_sizes']
    if stats.empty:
        print("Effect sizes DataFrame is empty")
        return
    
    # Plot 1: Effect sizes (Cohen's d) heatmap
    for metric_suffix in ['true', 'obs']:
        d_col = f'cohens_d_{metric_suffix}'
        if d_col not in stats.columns:
            print(f"Column {d_col} not found in stats")
            continue

        for objective, obj_data in stats.groupby('objective'):
            # Group by key factors
            for (error_model, oracle_model), data in obj_data.groupby(['error_model', 'oracle_model']):
                if len(data) < 2:
                    print(f"Insufficient data for {objective}, {error_model}, {oracle_model} ({len(data)} rows)")
                    continue

                # Create pivot table for heatmap
                pivot_data = data.pivot_table(
                    values=d_col,
                    index='acquisition',
                    columns='jitter_std',
                    aggfunc='mean'
                )

                # Check if pivot_data is empty or all NaN
                if pivot_data.empty or pivot_data.isna().all().all():
                    print(f"No valid data for heatmap: {objective}, {error_model}, {oracle_model}, {metric_suffix}")
                    continue

                # Check if there's at least some non-NaN data
                if pivot_data.notna().sum().sum() == 0:
                    print(f"All NaN values in pivot table: {objective}, {error_model}, {oracle_model}, {metric_suffix}")
                    continue

                plt.figure(figsize=(12, 6))

                try:
                    sns.heatmap(
                        pivot_data,
                        annot=True,
                        fmt='.3f',
                        cmap='RdYlGn',
                        center=0,
                        cbar_kws={'label': "Cohen's d"},
                        mask=pivot_data.isna()  # Mask NaN values
                    )

                    metric_name = 'True Objective' if metric_suffix == 'true' else 'Observed Objective'
                    plt.title(f"Effect Sizes ({metric_name}) - {objective}, {error_model}, {oracle_model}")
                    plt.ylabel("Acquisition Strategy")
                    plt.xlabel("Jitter Std")
                    plt.tight_layout()

                    filename = f"effect_sizes_{metric_suffix}_{objective}_{error_model}_{oracle_model}.png"
                    plt.savefig(output_dir / filename, dpi=200)
                    print(f"Saved heatmap: {filename}")
                except Exception as e:
                    print(
                        f"Failed to create heatmap for {objective}, {error_model}, {oracle_model}, "
                        f"{metric_suffix}: {e}"
                    )
                finally:
                    plt.close()
    
    # Plot 2: Effect size distributions
    for objective, obj_data in stats.groupby('objective'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        plots_created = False
        for idx, (metric_suffix, ax) in enumerate(zip(['true', 'obs'], axes)):
            d_col = f'cohens_d_{metric_suffix}'
            if d_col not in obj_data.columns:
                print(f"Column {d_col} not found for distribution plot")
                continue

            # Check if we have valid data
            valid_data = obj_data[obj_data[d_col].notna()]
            if valid_data.empty:
                print(f"No valid data for distribution plot: {objective}, {metric_suffix}")
                continue

            try:
                sns.boxplot(data=valid_data, x='acquisition', y=d_col, hue='error_model', ax=ax)

                # Add reference lines for effect size thresholds
                ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(-0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(-0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(-0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

                metric_name = 'True Objective' if metric_suffix == 'true' else 'Observed Objective'
                ax.set_title(f"Effect Size Distribution - {objective} ({metric_name})")
                ax.set_ylabel("Cohen's d")
                ax.set_xlabel("Acquisition Strategy")
                ax.tick_params(axis='x', rotation=45)

                plots_created = True
            except Exception as e:
                print(f"Failed to create distribution plot for {objective}, {metric_suffix}: {e}")

        if plots_created:
            plt.tight_layout()
            filename = f"effect_sizes_distribution_{objective}.png"
            plt.savefig(output_dir / filename, dpi=200)
            print(f"Saved distribution plot: {filename}")
        else:
            print(f"No distribution plots created - insufficient valid data for {objective}")

        plt.close(fig)
    
    
def plot_objectives(df: pd.DataFrame, output_dir: Path) -> None:
    group_cols = [
        "objective",
        "acquisition",
        "error_model",
        "jitter_std",
        "jitter_iteration",
        "oracle_model",
        "iteration",
    ]
    grouped = (
        df.groupby(group_cols)[["objective_true", "objective_observed"]].mean().reset_index()
    )
    for (objective, acq, error_model, jitter_std, jitter_iteration, oracle_model), data in grouped.groupby(
        ["objective", "acquisition", "error_model", "jitter_std", "jitter_iteration", "oracle_model"]
    ):
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=data, x="iteration", y="objective_true", label="Objective (true)")
        sns.lineplot(data=data, x="iteration", y="objective_observed", label="Objective (observed)")
        plt.title(
            "Objective trajectory "
            f"({objective}, {acq}, {error_model}, {oracle_model}, jitter={jitter_iteration}, std={jitter_std})"
        )
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.tight_layout()
        filename = (
            "objective_trajectory_"
            f"{objective}_{acq}_{error_model}_{oracle_model}_jit{jitter_iteration}_std{jitter_std}.png"
        )
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def plot_adjustments(input_dir: Path, output_dir: Path) -> None:
    summary_stats = input_dir / "bo_sensor_error_summary_stats.csv"
    if not summary_stats.exists():
        return
    stats = pd.read_csv(summary_stats)
    for (objective, error_model, jitter_iteration), data in stats.groupby(
        ["objective", "error_model", "jitter_iteration"]
    ):
        plot = sns.catplot(
            data=data,
            x="acquisition",
            y="delta_l2_mean",
            hue="baseline",
            col="jitter_std",
            kind="bar",
            height=4,
            aspect=1.1,
        )
        plot.fig.suptitle(
            f"Mean parameter adjustment (L2 norm) - {objective} / {error_model} (jitter={jitter_iteration})"
        )
        plot.set_axis_labels("acquisition", "Mean delta L2")
        plot.tight_layout()
        filename = f"delta_l2_mean_{objective}_{error_model}_jit{jitter_iteration}.png"
        plot.savefig(output_dir / filename, dpi=200)
        plt.close(plot.fig)


def plot_excess_adjustments(input_dir: Path, output_dir: Path) -> None:
    excess_path = input_dir / "bo_sensor_error_excess_summary.csv"
    if not excess_path.exists():
        return
    excess = pd.read_csv(excess_path)
    summary = (
        excess.groupby(["objective", "acquisition", "error_model", "jitter_iteration", "jitter_std"])
        .agg(
            delta_excess_l2_mean=("delta_excess_l2_norm", "mean"),
            delta_excess_l2_std=("delta_excess_l2_norm", "std"),
            runs=("delta_excess_l2_norm", "count"),
        )
        .reset_index()
    )
    for (objective, jitter_iteration, jitter_std), data in summary.groupby(
        ["objective", "jitter_iteration", "jitter_std"]
    ):
        plt.figure(figsize=(8, 4))
        sns.barplot(
            data=data,
            x="acquisition",
            y="delta_excess_l2_mean",
            hue="error_model",
        )
        plt.title(
            "Mean excess adjustment (L2 norm) "
            f"- {objective} jitter={jitter_iteration}, std={jitter_std:.2f}"
        )
        plt.ylabel("Mean delta excess L2")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"delta_excess_l2_mean_{objective}_jit{jitter_iteration}_std{jitter_std}.png",
            dpi=200,
        )
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = load_iteration_logs(args.input_dir)
    plot_objectives(logs, args.output_dir)
    plot_adjustments(args.input_dir, args.output_dir)
    plot_excess_adjustments(args.input_dir, args.output_dir)
    final_outcomes = summarize_final_outcomes(logs)
    results_dict = evaluate_final_outcomes_improved(final_outcomes, args.output_dir)
    plot_final_outcome_significance(results_dict, args.output_dir)
    generate_statistical_report(results_dict, args.output_dir)
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
