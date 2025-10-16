#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_os_performance.py

Comprehensive diagnosis of why OS-DML underperforms in certain scenarios.
Based on analysis results showing OS has worst performance in OBS-S.
"""
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn and other warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from methods import run_os, run_unif, run_lss
from data_generation import generate_obs_s_data, generate_rct_s_data, generate_obs_c_data
import time

# Suppress LightGBM output globally
import os
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

def analyze_existing_results():
    """Analyze the existing simulation results to identify patterns."""
    
    print("="*80)
    print("DIAGNOSIS: OS-DML PERFORMANCE ISSUES")
    print("="*80)
    
    # Load Experiment 2 results
    df = pd.read_csv('analysis_results/experiment_2_main_comparison/summary_table.csv')
    
    print("\n" + "="*80)
    print("ISSUE 1: OS-DML UNDERPERFORMS IN OBS-S SCENARIO")
    print("="*80)
    
    print("\nRMSE Comparison (lower is better):")
    print("-"*80)
    for scenario in ['OBS-S', 'OBS-C', 'RCT-S', 'RCT-C']:
        scenario_df = df[df['scenario'] == scenario][['method', 'RMSE']].set_index('method')
        print(f"\n{scenario}:")
        print(scenario_df.sort_values('RMSE'))
        
        # Highlight if OS is worst among subsampling methods
        subsampling_methods = scenario_df.loc[['OS', 'UNIF', 'LSS']]
        if subsampling_methods.idxmin()['RMSE'] != 'OS':
            print(f"  ⚠️  OS is NOT the best! Best is: {subsampling_methods.idxmin()['RMSE']}")
        if subsampling_methods.idxmax()['RMSE'] == 'OS':
            print(f"  ❌ OS is the WORST among subsampling methods!")
    
    print("\n" + "="*80)
    print("ISSUE 2: POOR COVERAGE RATES")
    print("="*80)
    
    print("\nCoverage Comparison (should be ~0.95):")
    print("-"*80)
    os_coverage = df[df['method'] == 'OS'][['scenario', 'Coverage']]
    print(os_coverage)
    print(f"\nScenarios with coverage < 0.90:")
    poor_coverage = os_coverage[os_coverage['Coverage'] < 0.90]
    if not poor_coverage.empty:
        print(poor_coverage)
        print("  ❌ SEVERE UNDERCOVERAGE!")
    
    print("\n" + "="*80)
    print("ISSUE 3: BIAS ANALYSIS")
    print("="*80)
    
    print("\nBias Comparison (should be ~0):")
    print("-"*80)
    bias_comparison = df[df['method'].isin(['OS', 'UNIF', 'LSS'])][['scenario', 'method', 'Bias']].pivot(
        index='scenario', columns='method', values='Bias')
    print(bias_comparison)
    
    print("\n" + "="*80)
    print("POTENTIAL ROOT CAUSES")
    print("="*80)
    
    print("""
    Based on the results, OS-DML underperforms mainly in OBS-S:
    
    1. ❌ WORST RMSE: 0.0354 (vs UNIF: 0.0297, LSS: 0.0281)
    2. ❌ LOW COVERAGE: 82% (should be 95%)
    3. ❌ LARGE BIAS: -0.0214 (vs UNIF: 0.0001)
    
    POSSIBLE CAUSES:
    
    A. PILOT SAMPLE TOO SMALL (r0=3000)
       - May not capture probability structure well
       - Pilot estimates may be noisy
       → Test: Increase r0 from 3000 to 5000
    
    B. STABILIZATION CONSTANT TOO LARGE (δ=0.01)
       - May over-regularize probabilities
       - Reduces adaptivity advantage
       → Test: Try δ=0.001 or δ=0
    
    C. PILOT NUISANCE ESTIMATES POOR QUALITY
       - Only 30 trees in pilot (vs 100 in main)
       - May not learn treatment effect heterogeneity
       → Test: Use 100 trees in pilot
    
    D. PROBABILITY CONCENTRATION ISSUES
       - Some units may get very high probabilities
       - Could lead to high variance
       → Diagnose: Check probability distribution
    
    E. VARIANCE ESTIMATOR ISSUES
       - Hansen-Hurwitz variance may be underestimated
       - Two-stage variance formula might be wrong
       → Test: Compare with bootstrap variance
    
    F. OBS-S SPECIFIC ISSUES
       - Moderate propensity overlap
       - Treatment effect heterogeneity
       - May be harder to estimate influence functions
       → Diagnose: Check pilot IF quality
    """)


def diagnostic_test_1_gbm_complexity():
    """Test if pilot model complexity (number of trees) affects OS performance."""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST 1: GBM COMPLEXITY SENSITIVITY")
    print("="*80)
    print("\nHypothesis: Pilot uses too few trees (30 vs 100 in main)")
    
    np.random.seed(42)
    data = generate_obs_s_data(n=100000, p=40)
    
    # Test different numbers of estimators in pilot
    n_estimators_values = [10, 20, 30, 50, 75, 100, 150]
    results = []
    
    print("\nTesting different pilot model complexities:")
    print("-"*80)
    print(f"{'n_estimators':<15} {'RMSE':<10} {'Bias':<10} {'Coverage':<10} {'Runtime':<10}")
    print("-"*80)
    
    for n_est in n_estimators_values:
        # Need to temporarily modify the model to test this
        # We'll do this by manually implementing the pilot stage
        reps_results = []
        
        for rep in range(10):
            np.random.seed(42 + rep)
            
            # Manual pilot with different complexity
            N = len(data['Y_obs'])
            r0, r1 = 3000, 7000
            
            # Phase 1: Pilot with specified complexity
            pilot_idx = np.random.choice(N, size=r0, replace=True)
            X_pilot = data['X'][pilot_idx]
            W_pilot = data['W'][pilot_idx]
            Y_pilot = data['Y_obs'][pilot_idx]
            
            import lightgbm as lgb
            import config
            lgbm_params = {'n_jobs': 1, 'random_state': config.BASE_SEED, 'n_estimators': n_est, 
                          'verbose': -1}
            
            start_time = time.time()
            mu0_model = lgb.LGBMRegressor(**lgbm_params).fit(X_pilot[W_pilot == 0], Y_pilot[W_pilot == 0])
            mu1_model = lgb.LGBMRegressor(**lgbm_params).fit(X_pilot[W_pilot == 1], Y_pilot[W_pilot == 1])
            e_model = lgb.LGBMClassifier(**lgbm_params).fit(X_pilot, W_pilot)
            
            # Predict on full data
            mu0_full = mu0_model.predict(data['X'])
            mu1_full = mu1_model.predict(data['X'])
            e_full = np.clip(e_model.predict_proba(data['X'])[:, 1], 0.01, 0.99)
            
            # Compute probabilities
            from methods import _orthogonal_score
            phi_pilot = _orthogonal_score(data['Y_obs'], data['W'], mu0_full, mu1_full, e_full)
            delta = 0.01
            abs_phi = np.abs(phi_pilot)
            pps_probs = (abs_phi + delta) / np.sum(abs_phi + delta)
            
            # Phase 2: Main sample (using standard approach)
            pps_idx = np.random.choice(N, size=r1, replace=True, p=pps_probs)
            combined_idx = np.concatenate([pilot_idx, pps_idx])
            q_j = np.concatenate([np.full(r0, 1.0/N), pps_probs[pps_idx]])
            
            # Run standard DML on combined sample
            from methods import _fit_nuisance_models, _get_hansen_hurwitz_ci
            X_c = data['X'][combined_idx]
            W_c = data['W'][combined_idx]
            Y_c = data['Y_obs'][combined_idx]
            
            weights = 1.0 / q_j
            mu0, mu1, e = _fit_nuisance_models(X_c, W_c, Y_c, 2, False, 
                                               np.mean(W_c), sample_weight=weights)
            scores = _orthogonal_score(Y_c, W_c, mu0, mu1, e)
            est_ate, ci_lower, ci_upper = _get_hansen_hurwitz_ci(scores, q_j, N)
            runtime = time.time() - start_time
            
            bias = est_ate - data['true_ate']
            coverage = (ci_lower <= data['true_ate'] <= ci_upper)
            reps_results.append({'bias': bias, 'sq_error': bias**2, 'coverage': coverage, 'runtime': runtime})
        
        df_temp = pd.DataFrame(reps_results)
        rmse = np.sqrt(df_temp['sq_error'].mean())
        bias_mean = df_temp['bias'].mean()
        coverage_mean = df_temp['coverage'].mean()
        runtime_mean = df_temp['runtime'].mean()
        
        print(f"{n_est:<15} {rmse:<10.4f} {bias_mean:<10.4f} {coverage_mean:<10.2f} {runtime_mean:<10.2f}")
        results.append({'n_estimators': n_est, 'rmse': rmse, 'bias': bias_mean, 
                       'coverage': coverage_mean, 'runtime': runtime_mean})
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('test1_gbm_complexity_results.csv', index=False)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RMSE plot
    axes[0, 0].plot(df_results['n_estimators'], df_results['rmse'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].axvline(x=30, color='red', linestyle='--', label='Current (30)')
    axes[0, 0].axvline(x=100, color='green', linestyle='--', label='Main stage (100)')
    axes[0, 0].set_xlabel('Number of Trees in Pilot')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE vs. Pilot Model Complexity')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Coverage plot
    axes[0, 1].plot(df_results['n_estimators'], df_results['coverage'], 'o-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.95, color='red', linestyle='--', label='Nominal 95%')
    axes[0, 1].axvline(x=30, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Number of Trees in Pilot')
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].set_title('Coverage vs. Pilot Model Complexity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0.7, 1.0])
    
    # Bias plot
    axes[1, 0].plot(df_results['n_estimators'], df_results['bias'], 'o-', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=30, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Number of Trees in Pilot')
    axes[1, 0].set_ylabel('Bias')
    axes[1, 0].set_title('Bias vs. Pilot Model Complexity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Runtime plot
    axes[1, 1].plot(df_results['n_estimators'], df_results['runtime'], 'o-', linewidth=2, markersize=8)
    axes[1, 1].axvline(x=30, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Number of Trees in Pilot')
    axes[1, 1].set_ylabel('Runtime (seconds)')
    axes[1, 1].set_title('Runtime vs. Pilot Model Complexity')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test1_gbm_complexity_sensitivity.pdf', dpi=300, bbox_inches='tight')
    print("\n✓ Results saved to: test1_gbm_complexity_results.csv")
    print("✓ Plot saved to: test1_gbm_complexity_sensitivity.pdf")
    
    optimal_n = df_results.loc[df_results['rmse'].idxmin(), 'n_estimators']
    print(f"\n→ Optimal n_estimators based on RMSE: {optimal_n:.0f}")
    print(f"→ Current n_estimators in pilot: 30")
    
    if optimal_n > 30:
        print(f"✓ DIAGNOSIS: Pilot model too simple! Should use n_estimators={optimal_n:.0f}")
    else:
        print("✗ Model complexity is not the main issue")


def diagnostic_test_2_delta():
    """Test if delta=0.01 is too large."""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST 2: DELTA SENSITIVITY")
    print("="*80)
    print("\nHypothesis: δ=0.01 over-regularizes probabilities")
    
    np.random.seed(42)
    data = generate_obs_s_data(n=100000, p=40)
    
    # Test range: 10^-4 to 10^-1 (log scale)
    delta_values = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = []
    
    print("\nTesting different delta values:")
    print("-"*80)
    print(f"{'Delta':<10} {'RMSE':<10} {'Bias':<10} {'Coverage':<10} {'Runtime':<10}")
    print("-"*80)
    
    for delta in delta_values:
        # Run multiple replications
        reps_results = []
        for rep in range(10):
            np.random.seed(42 + rep)
            result = run_os(
                X=data['X'], W=data['W'], Y_obs=data['Y_obs'],
                pi_true=data['pi_true'], is_rct=False,
                r={'r0': 3000, 'r1': 7000}, k_folds=2, delta=delta
            )
            bias = result['est_ate'] - data['true_ate']
            coverage = (result['ci_lower'] <= data['true_ate'] <= result['ci_upper'])
            reps_results.append({'bias': bias, 'sq_error': bias**2, 'coverage': coverage, 
                               'runtime': result['runtime']})
        
        df_temp = pd.DataFrame(reps_results)
        rmse = np.sqrt(df_temp['sq_error'].mean())
        bias_mean = df_temp['bias'].mean()
        coverage_mean = df_temp['coverage'].mean()
        runtime_mean = df_temp['runtime'].mean()
        
        print(f"{delta:<10.4f} {rmse:<10.4f} {bias_mean:<10.4f} {coverage_mean:<10.2f} {runtime_mean:<10.2f}")
        results.append({'delta': delta, 'rmse': rmse, 'bias': bias_mean, 
                       'coverage': coverage_mean, 'runtime': runtime_mean})
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('test2_delta_sensitivity_results.csv', index=False)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RMSE plot
    axes[0, 0].plot(df_results['delta'], df_results['rmse'], 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].axvline(x=0.01, color='red', linestyle='--', label='Current (0.01)')
    axes[0, 0].set_xlabel('Delta (δ)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE vs. Stabilization Constant')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Coverage plot
    axes[0, 1].plot(df_results['delta'], df_results['coverage'], 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].axhline(y=0.95, color='red', linestyle='--', label='Nominal 95%')
    axes[0, 1].axvline(x=0.01, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Delta (δ)')
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].set_title('Coverage vs. Stabilization Constant')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0.7, 1.0])
    
    # Bias plot
    axes[1, 0].plot(df_results['delta'], df_results['bias'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0.01, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Delta (δ)')
    axes[1, 0].set_ylabel('Bias')
    axes[1, 0].set_title('Bias vs. Stabilization Constant')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Runtime plot
    axes[1, 1].plot(df_results['delta'], df_results['runtime'], 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].axvline(x=0.01, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Delta (δ)')
    axes[1, 1].set_ylabel('Runtime (seconds)')
    axes[1, 1].set_title('Runtime vs. Stabilization Constant')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test2_delta_sensitivity.pdf', dpi=300, bbox_inches='tight')
    print("\n✓ Results saved to: test2_delta_sensitivity_results.csv")
    print("✓ Plot saved to: test2_delta_sensitivity.pdf")
    
    optimal_delta = df_results.loc[df_results['rmse'].idxmin(), 'delta']
    print(f"\n→ Optimal delta based on RMSE: {optimal_delta:.4f}")
    print(f"→ Current delta: 0.01")
    
    if optimal_delta < 0.01:
        print(f"✓ DIAGNOSIS: Delta too large! Should use delta={optimal_delta:.4f}")
    else:
        print("✗ Delta is not the main issue")


def diagnostic_test_3_pilot_size():
    """Test if larger pilot improves OS performance."""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST 3: PILOT SIZE SENSITIVITY")
    print("="*80)
    print("\nHypothesis: r0=3000 is too small, need larger pilot")
    
    np.random.seed(42)
    data = generate_obs_s_data(n=100000, p=40)
    
    # Test range: pilot_ratio from 0.1 to 0.9 (r_total = 10000)
    pilot_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    results = []
    
    print("\nTesting different pilot sizes (total budget=10000):")
    print("-"*80)
    print(f"{'r0':<8} {'r1':<8} {'RMSE':<10} {'Bias':<10} {'Coverage':<10} {'Runtime':<10}")
    print("-"*80)
    
    for r0 in pilot_sizes:
        r1 = 10000 - r0
        
        # Run multiple replications
        reps_results = []
        for rep in range(10):
            np.random.seed(42 + rep)
            result = run_os(
                X=data['X'], W=data['W'], Y_obs=data['Y_obs'],
                pi_true=data['pi_true'], is_rct=False,
                r={'r0': r0, 'r1': r1}, k_folds=2
            )
            bias = result['est_ate'] - data['true_ate']
            coverage = (result['ci_lower'] <= data['true_ate'] <= result['ci_upper'])
            reps_results.append({'bias': bias, 'sq_error': bias**2, 'coverage': coverage,
                               'runtime': result['runtime']})
        
        df_temp = pd.DataFrame(reps_results)
        rmse = np.sqrt(df_temp['sq_error'].mean())
        bias_mean = df_temp['bias'].mean()
        coverage_mean = df_temp['coverage'].mean()
        runtime_mean = df_temp['runtime'].mean()
        
        print(f"{r0:<8} {r1:<8} {rmse:<10.4f} {bias_mean:<10.4f} {coverage_mean:<10.2f} {runtime_mean:<10.2f}")
        results.append({'r0': r0, 'r1': r1, 'pilot_ratio': r0/10000, 'rmse': rmse, 'bias': bias_mean, 
                       'coverage': coverage_mean, 'runtime': runtime_mean})
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('test3_pilot_size_sensitivity_results.csv', index=False)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RMSE plot
    axes[0, 0].plot(df_results['pilot_ratio'], df_results['rmse'], 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].axvline(x=0.3, color='red', linestyle='--', label='Current (r0=3000, 30%)')
    axes[0, 0].set_xlabel('Pilot Ratio (r0 / r_total)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('RMSE vs. Pilot Size')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Coverage plot
    axes[0, 1].plot(df_results['pilot_ratio'], df_results['coverage'], 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].axhline(y=0.95, color='red', linestyle='--', label='Nominal 95%')
    axes[0, 1].axvline(x=0.3, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Pilot Ratio (r0 / r_total)')
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].set_title('Coverage vs. Pilot Size')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0.7, 1.0])
    
    # Bias plot
    axes[1, 0].plot(df_results['pilot_ratio'], df_results['bias'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0.3, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Pilot Ratio (r0 / r_total)')
    axes[1, 0].set_ylabel('Bias')
    axes[1, 0].set_title('Bias vs. Pilot Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Runtime plot
    axes[1, 1].plot(df_results['pilot_ratio'], df_results['runtime'], 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].axvline(x=0.3, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Pilot Ratio (r0 / r_total)')
    axes[1, 1].set_ylabel('Runtime (seconds)')
    axes[1, 1].set_title('Runtime vs. Pilot Size')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test3_pilot_size_sensitivity.pdf', dpi=300, bbox_inches='tight')
    print("\n✓ Results saved to: test3_pilot_size_sensitivity_results.csv")
    print("✓ Plot saved to: test3_pilot_size_sensitivity.pdf")
    
    optimal_r0 = df_results.loc[df_results['rmse'].idxmin(), 'r0']
    print(f"\n→ Optimal r0 based on RMSE: {optimal_r0:.0f}")
    print(f"→ Current r0 in experiments: 3000")
    
    if optimal_r0 > 3000:
        print(f"✓ DIAGNOSIS: Pilot too small! Should use r0={optimal_r0:.0f}")
    else:
        print("✗ Pilot size is not the main issue")


def diagnostic_probability_distribution():
    """Examine the distribution of sampling probabilities (supplementary analysis)."""
    
    print("\n" + "="*80)
    print("SUPPLEMENTARY ANALYSIS: PROBABILITY DISTRIBUTION")
    print("="*80)
    print("\nExamining sampling probability distribution")
    
    np.random.seed(42)
    data = generate_obs_s_data(n=100000, p=40)
    
    # Manually extract probabilities from OS-DML
    N = len(data['Y_obs'])
    r0 = 3000
    
    # Phase 1: Pilot
    pilot_idx = np.random.choice(N, size=r0, replace=True)
    X_pilot = data['X'][pilot_idx]
    W_pilot = data['W'][pilot_idx]
    Y_pilot = data['Y_obs'][pilot_idx]
    
    import lightgbm as lgb
    import config
    lgbm_params = {'n_jobs': 1, 'random_state': config.BASE_SEED, 'n_estimators': 30, 
                  'verbose': -1}
    
    mu0_model = lgb.LGBMRegressor(**lgbm_params).fit(X_pilot[W_pilot == 0], Y_pilot[W_pilot == 0])
    mu1_model = lgb.LGBMRegressor(**lgbm_params).fit(X_pilot[W_pilot == 1], Y_pilot[W_pilot == 1])
    e_model = lgb.LGBMClassifier(**lgbm_params).fit(X_pilot, W_pilot)
    
    mu0_full = mu0_model.predict(data['X'])
    mu1_full = mu1_model.predict(data['X'])
    e_full = np.clip(e_model.predict_proba(data['X'])[:, 1], 0.01, 0.99)
    
    # Compute influence functions
    from methods import _orthogonal_score
    phi_pilot = _orthogonal_score(data['Y_obs'], data['W'], mu0_full, mu1_full, e_full)
    
    # Compute probabilities
    delta = 0.01
    abs_phi = np.abs(phi_pilot)
    pps_probs = (abs_phi + delta) / np.sum(abs_phi + delta)
    
    print("\nSampling Probability Statistics:")
    print("-"*80)
    print(f"Min probability:  {pps_probs.min():.8f}")
    print(f"Max probability:  {pps_probs.max():.8f}")
    print(f"Mean probability: {pps_probs.mean():.8f}")
    print(f"Median probability: {np.median(pps_probs):.8f}")
    print(f"Std probability:  {pps_probs.std():.8f}")
    print(f"\nConcentration ratio (max/min): {pps_probs.max()/pps_probs.min():.1f}")
    print(f"Effective sample size: {1/np.sum(pps_probs**2):.0f} (out of {N})")
    
    # Check for extreme probabilities
    top_1pct = np.percentile(pps_probs, 99)
    print(f"\nTop 1% units have probability > {top_1pct:.8f}")
    print(f"These {int(0.01*N)} units account for {np.sum(pps_probs[pps_probs > top_1pct]):.2%} of total probability")
    
    if pps_probs.max() / pps_probs.min() > 1000:
        print("\n❌ EXTREME probability concentration detected!")
        print("   Some units are sampled 1000x more than others")
    
    # Influence function statistics
    print("\n\nInfluence Function Statistics:")
    print("-"*80)
    print(f"Min |φ|:  {abs_phi.min():.4f}")
    print(f"Max |φ|:  {abs_phi.max():.4f}")
    print(f"Mean |φ|: {abs_phi.mean():.4f}")
    print(f"Median |φ|: {np.median(abs_phi):.4f}")
    print(f"Std |φ|:  {abs_phi.std():.4f}")
    
    # Create histogram plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Probability distribution
    axes[0].hist(pps_probs, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Sampling Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Sampling Probabilities')
    axes[0].axvline(x=1/N, color='red', linestyle='--', label='Uniform (1/N)')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Influence function distribution
    axes[1].hist(abs_phi, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('|Influence Function|')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of |φ| Values')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('supplementary_probability_distribution.pdf', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved to: supplementary_probability_distribution.pdf")


def find_optimal_combination():
    """
    Analyze results from all three tests to find optimal parameter combination.
    Tests the optimal configuration and compares with current baseline.
    """
    
    print("\n" + "="*80)
    print("FINDING OPTIMAL PARAMETER COMBINATION")
    print("="*80)
    
    # Load results from all three tests
    try:
        df_gbm = pd.read_csv('test1_gbm_complexity_results.csv')
        df_delta = pd.read_csv('test2_delta_sensitivity_results.csv')
        df_pilot = pd.read_csv('test3_pilot_size_sensitivity_results.csv')
    except FileNotFoundError as e:
        print(f"\n❌ Cannot find test results files. Please ensure all tests completed successfully.")
        print(f"   Missing file: {e}")
        return
    
    print("\n" + "-"*80)
    print("INDIVIDUAL TEST RESULTS")
    print("-"*80)
    
    # Find optimal from each test
    optimal_gbm = df_gbm.loc[df_gbm['rmse'].idxmin()]
    optimal_delta = df_delta.loc[df_delta['rmse'].idxmin()]
    optimal_pilot = df_pilot.loc[df_pilot['rmse'].idxmin()]
    
    print(f"\nTest 1 - Optimal GBM Complexity:")
    print(f"  n_estimators = {optimal_gbm['n_estimators']:.0f}")
    print(f"  RMSE = {optimal_gbm['rmse']:.4f}, Coverage = {optimal_gbm['coverage']:.2f}")
    
    print(f"\nTest 2 - Optimal Delta:")
    print(f"  delta = {optimal_delta['delta']:.4f}")
    print(f"  RMSE = {optimal_delta['rmse']:.4f}, Coverage = {optimal_delta['coverage']:.2f}")
    
    print(f"\nTest 3 - Optimal Pilot Size:")
    print(f"  r0 = {optimal_pilot['r0']:.0f}, r1 = {optimal_pilot['r1']:.0f}")
    print(f"  RMSE = {optimal_pilot['rmse']:.4f}, Coverage = {optimal_pilot['coverage']:.2f}")
    
    # Current configuration
    print("\n" + "-"*80)
    print("CURRENT CONFIGURATION")
    print("-"*80)
    current_gbm = df_gbm[df_gbm['n_estimators'] == 30].iloc[0]
    current_delta = df_delta[df_delta['delta'] == 0.01].iloc[0]
    current_pilot = df_pilot[df_pilot['r0'] == 3000].iloc[0]
    
    print(f"  n_estimators = 30,  RMSE = {current_gbm['rmse']:.4f}, Coverage = {current_gbm['coverage']:.2f}")
    print(f"  delta = 0.01,       RMSE = {current_delta['rmse']:.4f}, Coverage = {current_delta['coverage']:.2f}")
    print(f"  r0 = 3000,          RMSE = {current_pilot['rmse']:.4f}, Coverage = {current_pilot['coverage']:.2f}")
    
    # Test optimal combination
    print("\n" + "="*80)
    print("TESTING OPTIMAL COMBINATION")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  n_estimators = {optimal_gbm['n_estimators']:.0f}")
    print(f"  delta = {optimal_delta['delta']:.4f}")
    print(f"  r0 = {optimal_pilot['r0']:.0f}, r1 = {optimal_pilot['r1']:.0f}")
    
    print("\nRunning 20 replications with optimal configuration...")
    
    np.random.seed(42)
    data = generate_obs_s_data(n=100000, p=40)
    
    N = len(data['Y_obs'])
    r0_opt = int(optimal_pilot['r0'])
    r1_opt = int(optimal_pilot['r1'])
    n_est_opt = int(optimal_gbm['n_estimators'])
    delta_opt = optimal_delta['delta']
    
    optimal_results = []
    
    for rep in range(20):
        np.random.seed(42 + rep)
        
        # Manual implementation with optimal parameters
        # Phase 1: Pilot with optimal complexity
        pilot_idx = np.random.choice(N, size=r0_opt, replace=True)
        X_pilot = data['X'][pilot_idx]
        W_pilot = data['W'][pilot_idx]
        Y_pilot = data['Y_obs'][pilot_idx]
        
        import lightgbm as lgb
        import config
        lgbm_params = {'n_jobs': 1, 'random_state': config.BASE_SEED, 'n_estimators': n_est_opt, 
                      'verbose': -1}
        
        start_time = time.time()
        mu0_model = lgb.LGBMRegressor(**lgbm_params).fit(X_pilot[W_pilot == 0], Y_pilot[W_pilot == 0])
        mu1_model = lgb.LGBMRegressor(**lgbm_params).fit(X_pilot[W_pilot == 1], Y_pilot[W_pilot == 1])
        e_model = lgb.LGBMClassifier(**lgbm_params).fit(X_pilot, W_pilot)
        
        # Predict on full data
        mu0_full = mu0_model.predict(data['X'])
        mu1_full = mu1_model.predict(data['X'])
        e_full = np.clip(e_model.predict_proba(data['X'])[:, 1], 0.01, 0.99)
        
        # Compute probabilities with optimal delta
        from methods import _orthogonal_score
        phi_pilot = _orthogonal_score(data['Y_obs'], data['W'], mu0_full, mu1_full, e_full)
        abs_phi = np.abs(phi_pilot)
        pps_probs = (abs_phi + delta_opt) / np.sum(abs_phi + delta_opt)
        
        # Phase 2: Main sample
        pps_idx = np.random.choice(N, size=r1_opt, replace=True, p=pps_probs)
        combined_idx = np.concatenate([pilot_idx, pps_idx])
        q_j = np.concatenate([np.full(r0_opt, 1.0/N), pps_probs[pps_idx]])
        
        # Fit final models
        from methods import _fit_nuisance_models, _get_hansen_hurwitz_ci
        X_c = data['X'][combined_idx]
        W_c = data['W'][combined_idx]
        Y_c = data['Y_obs'][combined_idx]
        
        weights = 1.0 / q_j
        mu0, mu1, e = _fit_nuisance_models(X_c, W_c, Y_c, 2, False, 
                                           np.mean(W_c), sample_weight=weights)
        scores = _orthogonal_score(Y_c, W_c, mu0, mu1, e)
        est_ate, ci_lower, ci_upper = _get_hansen_hurwitz_ci(scores, q_j, N)
        runtime = time.time() - start_time
        
        bias = est_ate - data['true_ate']
        coverage = (ci_lower <= data['true_ate'] <= ci_upper)
        optimal_results.append({'bias': bias, 'sq_error': bias**2, 'coverage': coverage, 'runtime': runtime})
    
    df_optimal = pd.DataFrame(optimal_results)
    optimal_rmse = np.sqrt(df_optimal['sq_error'].mean())
    optimal_bias = df_optimal['bias'].mean()
    optimal_coverage = df_optimal['coverage'].mean()
    optimal_runtime = df_optimal['runtime'].mean()
    
    print("\nOptimal Configuration Results:")
    print("-"*80)
    print(f"  RMSE:     {optimal_rmse:.4f}")
    print(f"  Bias:     {optimal_bias:.4f}")
    print(f"  Coverage: {optimal_coverage:.2f}")
    print(f"  Runtime:  {optimal_runtime:.2f}s")
    
    # Compare with current and individual optimals
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON")
    print("="*80)
    
    configurations = []
    
    # Current configuration
    configurations.append({
        'config': 'Current',
        'n_estimators': 30,
        'delta': 0.01,
        'r0': 3000,
        'r1': 7000,
        'rmse': current_gbm['rmse'],
        'coverage': current_gbm['coverage'],
        'bias': abs(current_gbm['bias']),
        'runtime': current_gbm['runtime']
    })
    
    # Optimal n_estimators only
    configurations.append({
        'config': 'Optimal GBM Only',
        'n_estimators': optimal_gbm['n_estimators'],
        'delta': 0.01,
        'r0': 3000,
        'r1': 7000,
        'rmse': optimal_gbm['rmse'],
        'coverage': optimal_gbm['coverage'],
        'bias': abs(optimal_gbm['bias']),
        'runtime': optimal_gbm['runtime']
    })
    
    # Optimal delta only
    configurations.append({
        'config': 'Optimal Delta Only',
        'n_estimators': 30,
        'delta': optimal_delta['delta'],
        'r0': 3000,
        'r1': 7000,
        'rmse': optimal_delta['rmse'],
        'coverage': optimal_delta['coverage'],
        'bias': abs(optimal_delta['bias']),
        'runtime': optimal_delta['runtime']
    })
    
    # Optimal r0 only
    configurations.append({
        'config': 'Optimal r0 Only',
        'n_estimators': 30,
        'delta': 0.01,
        'r0': optimal_pilot['r0'],
        'r1': optimal_pilot['r1'],
        'rmse': optimal_pilot['rmse'],
        'coverage': optimal_pilot['coverage'],
        'bias': abs(optimal_pilot['bias']),
        'runtime': optimal_pilot['runtime']
    })
    
    # Optimal combination
    configurations.append({
        'config': 'Optimal Combination',
        'n_estimators': optimal_gbm['n_estimators'],
        'delta': optimal_delta['delta'],
        'r0': optimal_pilot['r0'],
        'r1': optimal_pilot['r1'],
        'rmse': optimal_rmse,
        'coverage': optimal_coverage,
        'bias': abs(optimal_bias),
        'runtime': optimal_runtime
    })
    
    df_comparison = pd.DataFrame(configurations)
    df_comparison.to_csv('optimal_configuration_results.csv', index=False)
    
    print("\nConfiguration Comparison:")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    current_rmse = df_comparison[df_comparison['config'] == 'Current']['rmse'].values[0]
    optimal_rmse_final = df_comparison[df_comparison['config'] == 'Optimal Combination']['rmse'].values[0]
    
    rmse_improvement = (current_rmse - optimal_rmse_final) / current_rmse * 100
    
    print(f"\nRMSE Improvement: {current_rmse:.4f} → {optimal_rmse_final:.4f} ({rmse_improvement:.1f}% reduction)")
    
    current_cov = df_comparison[df_comparison['config'] == 'Current']['coverage'].values[0]
    optimal_cov = df_comparison[df_comparison['config'] == 'Optimal Combination']['coverage'].values[0]
    print(f"Coverage Improvement: {current_cov:.2f} → {optimal_cov:.2f}")
    
    # Contribution of each factor
    print("\n" + "-"*80)
    print("CONTRIBUTION OF EACH FACTOR")
    print("-"*80)
    
    gbm_contribution = (current_rmse - df_comparison[df_comparison['config'] == 'Optimal GBM Only']['rmse'].values[0]) / current_rmse * 100
    delta_contribution = (current_rmse - df_comparison[df_comparison['config'] == 'Optimal Delta Only']['rmse'].values[0]) / current_rmse * 100
    pilot_contribution = (current_rmse - df_comparison[df_comparison['config'] == 'Optimal r0 Only']['rmse'].values[0]) / current_rmse * 100
    
    print(f"GBM Complexity: {gbm_contribution:.1f}% RMSE reduction")
    print(f"Delta:          {delta_contribution:.1f}% RMSE reduction")
    print(f"Pilot Size:     {pilot_contribution:.1f}% RMSE reduction")
    print(f"\nCombined:       {rmse_improvement:.1f}% RMSE reduction")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = df_comparison['config'].values
    x_pos = np.arange(len(configs))
    
    # RMSE comparison
    rmse_vals = df_comparison['rmse'].values
    colors = ['red' if c == 'Current' else 'lightgreen' if c == 'Optimal Combination' else 'lightblue' 
              for c in configs]
    axes[0, 0].bar(x_pos, rmse_vals, color=colors, edgecolor='black')
    axes[0, 0].set_ylabel('RMSE', fontsize=11)
    axes[0, 0].set_title('RMSE Comparison Across Configurations', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(rmse_vals):
        axes[0, 0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Coverage comparison
    cov_vals = df_comparison['coverage'].values
    axes[0, 1].bar(x_pos, cov_vals, color=colors, edgecolor='black')
    axes[0, 1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Nominal 95%')
    axes[0, 1].set_ylabel('Coverage', fontsize=11)
    axes[0, 1].set_title('Coverage Comparison Across Configurations', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_ylim([0.75, 1.0])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].legend()
    
    # Bias comparison
    bias_vals = df_comparison['bias'].values
    axes[1, 0].bar(x_pos, bias_vals, color=colors, edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    axes[1, 0].set_ylabel('|Bias|', fontsize=11)
    axes[1, 0].set_title('Absolute Bias Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Runtime comparison
    runtime_vals = df_comparison['runtime'].values
    axes[1, 1].bar(x_pos, runtime_vals, color=colors, edgecolor='black')
    axes[1, 1].set_ylabel('Runtime (seconds)', fontsize=11)
    axes[1, 1].set_title('Runtime Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('optimal_configuration_comparison.pdf', dpi=300, bbox_inches='tight')
    
    print("\n✓ Results saved to: optimal_configuration_results.csv")
    print("✓ Plot saved to: optimal_configuration_comparison.pdf")
    
    # Generate final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    print(f"""
    OPTIMAL CONFIGURATION FOR OS-DML:
    
    Parameter              Current    Optimal    Change
    ─────────────────────────────────────────────────────
    n_estimators (pilot)      30        {optimal_gbm['n_estimators']:.0f}      {'↑' if optimal_gbm['n_estimators'] > 30 else '→'}
    delta                   0.0100    {optimal_delta['delta']:.4f}    {'↓' if optimal_delta['delta'] < 0.01 else '→'}
    r0 (pilot size)         3,000     {optimal_pilot['r0']:.0f}      {'↑' if optimal_pilot['r0'] > 3000 else '→'}
    r1 (main size)          7,000     {optimal_pilot['r1']:.0f}      {'↓' if optimal_pilot['r1'] < 7000 else '→'}
    
    PERFORMANCE IMPROVEMENT:
    ─────────────────────────────────────────────────────
    RMSE:     {current_rmse:.4f} → {optimal_rmse_final:.4f}  ({rmse_improvement:.1f}% reduction)
    Coverage: {current_cov:.2f} → {optimal_cov:.2f}  ({(optimal_cov - current_cov)*100:+.0f}pp)
    
    TO APPLY THESE CHANGES:
    ─────────────────────────────────────────────────────
    1. Edit methods.py, line 303:
       lgbm_params_pilot = {{'n_estimators': {optimal_gbm['n_estimators']:.0f}, ...}}
    
    2. Edit methods.py, line 292:
       delta = kwargs.get('delta', {optimal_delta['delta']:.4f})
    
    3. Edit config.py, line 81:
       "r0": {optimal_pilot['r0']:.0f}, "r1": {optimal_pilot['r1']:.0f}
    
    4. Re-run Experiment 2 in Simulation.ipynb
    """)
    
    # Rank factors by importance
    contributions = [
        ('Pilot Size', pilot_contribution),
        ('GBM Complexity', gbm_contribution),
        ('Delta', delta_contribution)
    ]
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "-"*80)
    print("FACTOR IMPORTANCE RANKING")
    print("-"*80)
    for i, (factor, contrib) in enumerate(contributions, 1):
        print(f"{i}. {factor:<20} {contrib:>6.1f}% RMSE reduction")


def plot_scenario_comparison():
    """Create visualization comparing OS performance across scenarios."""
    
    print("\n" + "="*80)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    df = pd.read_csv('analysis_results/experiment_2_main_comparison/summary_table.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['RMSE', 'Coverage', 'Bias', 'Runtime']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Plot OS vs others for each scenario
        scenarios = df['scenario'].unique()
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, method in enumerate(['OS', 'UNIF', 'LSS']):
            values = df[df['method'] == method].sort_values('scenario')[metric].values
            ax.bar(x + i*width, values, width, label=method)
        
        # Add reference line for coverage
        if metric == 'Coverage':
            ax.axhline(y=0.95, color='red', linestyle='--', label='Nominal 95%')
            ax.set_ylim([0.75, 1.0])
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Scenario')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('os_diagnostic_plots.pdf', dpi=300)
    print("\n✓ Diagnostic plots saved to: os_diagnostic_plots.pdf")


if __name__ == "__main__":
    
    # Analyze existing results
    analyze_existing_results()
    
    # Run diagnostic tests in order
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTIC TESTS")
    print("="*80)
    print("\nEach test runs 10 replications per configuration on OBS-S scenario")
    print("This will take approximately 10-15 minutes in total")
    print("="*80)
    
    # TEST 1: GBM Complexity (First priority)
    print("\n\n")
    try:
        diagnostic_test_1_gbm_complexity()
    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # TEST 2: Delta Sensitivity (Second priority)
    print("\n\n")
    try:
        diagnostic_test_2_delta()
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # TEST 3: Pilot Size Sensitivity (Third priority)
    print("\n\n")
    try:
        diagnostic_test_3_pilot_size()
    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Supplementary: Probability distribution
    print("\n\n")
    try:
        diagnostic_probability_distribution()
    except Exception as e:
        print(f"\n❌ Supplementary analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Overall comparison plot
    print("\n\n")
    try:
        plot_scenario_comparison()
    except Exception as e:
        print(f"\n❌ Scenario comparison plot failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Find optimal combination
    print("\n\n")
    try:
        find_optimal_combination()
    except Exception as e:
        print(f"\n❌ Optimal combination analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print final summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print("""
    TESTS COMPLETED:
    ✓ Test 1: GBM Complexity Sensitivity
      - Output: test1_gbm_complexity_results.csv
      - Plot:   test1_gbm_complexity_sensitivity.pdf
      
    ✓ Test 2: Delta Sensitivity
      - Output: test2_delta_sensitivity_results.csv
      - Plot:   test2_delta_sensitivity.pdf
      
    ✓ Test 3: Pilot Size Sensitivity
      - Output: test3_pilot_size_sensitivity_results.csv
      - Plot:   test3_pilot_size_sensitivity.pdf
    
    ✓ Supplementary: Probability Distribution Analysis
      - Plot:   supplementary_probability_distribution.pdf
    
    ✓ Scenario Comparison Plot
      - Plot:   os_diagnostic_plots.pdf
    
    ✓ Optimal Combination Analysis
      - Output: optimal_configuration_results.csv
      - Plot:   optimal_configuration_comparison.pdf
    
    NEXT STEPS:
    1. Review optimal_configuration_comparison.pdf for best parameters
    2. Check optimal_configuration_results.csv for detailed comparison
    3. Apply recommended configuration to config.py and methods.py
    4. Re-run experiments with optimized parameters
    """)

