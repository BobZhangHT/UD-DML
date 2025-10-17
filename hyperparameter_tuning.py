#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hyperparameter_tuning.py

Hyperparameter tuning for OS-DML using Optuna (Bayesian Optimization).
Tunes three key parameters:
  1. n_estimators: GBM complexity in pilot stage
  2. delta: Stabilization constant
  3. r0: Pilot sample size (r1 = r_total - r0)

Features:
  - Efficient Bayesian optimization (TPE algorithm)
  - Checkpoint mechanism: Auto-saves progress to SQLite database
  - Multi-CPU support: Parallel trial execution
  - Resume capability: Can interrupt (Ctrl+C) and resume later
  - Server-friendly: Uses matplotlib (no browser needed) for visualizations

Usage:
  python hyperparameter_tuning.py
  
  To resume after interruption:
  - Simply run the script again with RESUME=True (default)
  - Progress is automatically loaded from checkpoint database
  
  To start fresh:
  - Set RESUME=False or delete the .db file
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Suppress LightGBM output
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
        plot_parallel_coordinate
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not installed. Installing...")
    os.system('pip install optuna plotly kaleido -q')
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_contour,
        plot_parallel_coordinate
    )
    OPTUNA_AVAILABLE = True

from data_generation import generate_obs_s_data, generate_obs_c_data, generate_rct_s_data, generate_rct_c_data
import lightgbm as lgb
import config
from methods import _orthogonal_score, _fit_nuisance_models, _get_hansen_hurwitz_ci


class OSHyperparameterOptimizer:
    """
    Hyperparameter optimizer for OS-DML using Optuna.
    Supports both single-scenario and multi-scenario optimization.
    """
    
    def __init__(self, scenarios='all', n_replications=10, r_total=10000, 
                 aggregation='mean', n_jobs=1):
        """
        Initialize the optimizer.
        
        Args:
            scenarios: 'all' for multi-scenario, or specific scenario name
            n_replications: Number of Monte Carlo replications per trial per scenario
            r_total: Total sample budget (r0 + r1)
            aggregation: How to aggregate across scenarios ('mean', 'max', 'weighted')
            n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
        """
        self.n_replications = n_replications
        self.r_total = r_total
        self.N = 100000  # Population size
        self.aggregation = aggregation
        self.n_jobs = n_jobs
        
        # Configure scenarios
        all_scenarios = {
            'OBS-S': {
                'func': generate_obs_s_data,
                'params': {'n': self.N, 'p': 40},
                'is_rct': False
            },
            'OBS-C': {
                'func': generate_obs_c_data,
                'params': {'n': self.N, 'p': 100},
                'is_rct': False
            },
            'RCT-S': {
                'func': generate_rct_s_data,
                'params': {'n': self.N, 'p': 30},
                'is_rct': True
            },
            'RCT-C': {
                'func': generate_rct_c_data,
                'params': {'n': self.N, 'p': 120},
                'is_rct': True
            }
        }
        
        if scenarios == 'all':
            self.scenarios = all_scenarios
            self.multi_scenario = True
            self.scenario_name = 'all_scenarios'
            print(f"Initializing multi-scenario optimizer (all 4 scenarios)")
        else:
            if scenarios not in all_scenarios:
                raise ValueError(f"Unknown scenario: {scenarios}")
            self.scenarios = {scenarios: all_scenarios[scenarios]}
            self.multi_scenario = False
            self.scenario_name = scenarios
            print(f"Initializing single-scenario optimizer ({scenarios})")
        
        print(f"  N_replications: {n_replications} per scenario per trial")
        print(f"  Aggregation:    {aggregation}")
        print(f"  Total budget:   r0 + r1 = {r_total}")
    
    def objective(self, trial):
        """
        Objective function for Optuna to minimize.
        
        For multi-scenario: Aggregates performance across all scenarios.
        Returns aggregated RMSE (lower is better).
        """
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 20, 150, step=10)
        # Sample log10(delta) uniformly in [-4, -1], then convert: 10^-4, 10^-3.5, 10^-3, etc.
        log10_delta = trial.suggest_float('log10_delta', -4, -1)
        delta = 10 ** log10_delta
        pilot_ratio = trial.suggest_float('pilot_ratio', 0.1, 0.9, step=0.05)
        
        r0 = int(self.r_total * pilot_ratio)
        r1 = self.r_total - r0
        
        # Dictionary to store results by scenario
        scenario_metrics = {}
        
        # Run on each scenario
        for scenario_name, scenario_config in self.scenarios.items():
            results = []
            
            for rep in range(self.n_replications):
                try:
                    # Generate data with different seed (thread-safe)
                    # Use trial number and rep to ensure unique seeds across parallel workers
                    seed = config.BASE_SEED + trial.number * 10000 + rep
                    np.random.seed(seed)
                    data = scenario_config['func'](**scenario_config['params'])
                    true_ate = data['true_ate']
                    
                    # Run OS-DML with specified hyperparameters
                    result = self._run_os_with_params(
                        data, n_estimators, delta, r0, r1, scenario_config['is_rct']
                    )
                    
                    # Calculate metrics
                    bias = result['est_ate'] - true_ate
                    sq_error = bias**2
                    coverage = (result['ci_lower'] <= true_ate <= result['ci_upper'])
                    ci_width = result['ci_upper'] - result['ci_lower']
                    
                    results.append({
                        'sq_error': sq_error,
                        'bias': bias,
                        'coverage': coverage,
                        'ci_width': ci_width,
                        'runtime': result['runtime']
                    })
                    
                except Exception as e:
                    # If trial fails, return high penalty
                    print(f"Trial failed on {scenario_name} with error: {e}")
                    return 1.0  # High RMSE penalty
            
            # Aggregate metrics for this scenario
            df_results = pd.DataFrame(results)
            rmse = np.sqrt(df_results['sq_error'].mean())
            coverage = df_results['coverage'].mean()
            bias_mean = df_results['bias'].mean()
            
            scenario_metrics[scenario_name] = {
                'rmse': rmse,
                'coverage': coverage,
                'bias': bias_mean
            }
        
        # Store per-scenario metrics as user attributes
        for scenario_name, metrics in scenario_metrics.items():
            trial.set_user_attr(f'rmse_{scenario_name}', metrics['rmse'])
            trial.set_user_attr(f'coverage_{scenario_name}', metrics['coverage'])
            trial.set_user_attr(f'bias_{scenario_name}', metrics['bias'])
        
        trial.set_user_attr('r0', r0)
        trial.set_user_attr('r1', r1)
        
        # Aggregate across scenarios
        rmse_values = [m['rmse'] for m in scenario_metrics.values()]
        coverage_values = [m['coverage'] for m in scenario_metrics.values()]
        
        if self.aggregation == 'mean':
            # Average RMSE across scenarios
            agg_rmse = np.mean(rmse_values)
            agg_coverage = np.mean(coverage_values)
        elif self.aggregation == 'max':
            # Worst-case RMSE (minimax)
            agg_rmse = np.max(rmse_values)
            agg_coverage = np.min(coverage_values)
        elif self.aggregation == 'weighted':
            # Weighted by scenario difficulty (OBS-S and OBS-C get more weight)
            weights = {'OBS-S': 0.3, 'OBS-C': 0.3, 'RCT-S': 0.2, 'RCT-C': 0.2}
            agg_rmse = sum(scenario_metrics[s]['rmse'] * weights.get(s, 0.25) 
                          for s in scenario_metrics.keys())
            agg_coverage = sum(scenario_metrics[s]['coverage'] * weights.get(s, 0.25) 
                              for s in scenario_metrics.keys())
        else:
            agg_rmse = np.mean(rmse_values)
            agg_coverage = np.mean(coverage_values)
        
        # Store aggregated metrics
        trial.set_user_attr('agg_rmse', agg_rmse)
        trial.set_user_attr('agg_coverage', agg_coverage)
        
        # Optuna minimizes, so return RMSE + coverage penalty
        coverage_penalty = abs(agg_coverage - 0.95) * 0.5
        
        return agg_rmse + coverage_penalty
    
    def _run_os_with_params(self, data, n_estimators, delta, r0, r1, is_rct):
        """
        Run OS-DML with specified hyperparameters.
        Manually implements to allow parameter control.
        
        Args:
            data: Generated data dictionary
            n_estimators: Number of trees for pilot GBM
            delta: Stabilization constant
            r0: Pilot sample size
            r1: Main sample size
            is_rct: Whether this is RCT data
        """
        N = len(data['Y_obs'])
        
        # Phase 1: Pilot with specified complexity
        pilot_idx = np.random.choice(N, size=r0, replace=True)
        X_pilot = data['X'][pilot_idx]
        W_pilot = data['W'][pilot_idx]
        Y_pilot = data['Y_obs'][pilot_idx]
        
        lgbm_params = {
            'n_jobs': 1, 
            'random_state': config.BASE_SEED, 
            'n_estimators': n_estimators,
            'num_leaves': 31,
            'verbose': -1
        }
        
        start_time = time.time()
        
        mu0_model = lgb.LGBMRegressor(**lgbm_params).fit(
            X_pilot[W_pilot == 0], Y_pilot[W_pilot == 0])
        mu1_model = lgb.LGBMRegressor(**lgbm_params).fit(
            X_pilot[W_pilot == 1], Y_pilot[W_pilot == 1])
        
        if is_rct:
            e_full = np.full(N, np.mean(W_pilot))
        else:
            e_model = lgb.LGBMClassifier(**lgbm_params).fit(X_pilot, W_pilot)
            e_full = np.clip(e_model.predict_proba(data['X'])[:, 1], 0.01, 0.99)
        
        # Predict on full data
        mu0_full = mu0_model.predict(data['X'])
        mu1_full = mu1_model.predict(data['X'])
        
        # Compute probabilities with specified delta
        phi_pilot = _orthogonal_score(data['Y_obs'], data['W'], mu0_full, mu1_full, e_full)
        abs_phi = np.abs(phi_pilot)
        pps_probs = (abs_phi + delta) / np.sum(abs_phi + delta)
        
        # Phase 2: Main sample
        pps_idx = np.random.choice(N, size=r1, replace=True, p=pps_probs)
        combined_idx = np.concatenate([pilot_idx, pps_idx])
        q_j = np.concatenate([np.full(r0, 1.0/N), pps_probs[pps_idx]])
        
        # Fit final models
        X_c = data['X'][combined_idx]
        W_c = data['W'][combined_idx]
        Y_c = data['Y_obs'][combined_idx]
        
        weights = 1.0 / q_j
        mu0, mu1, e = _fit_nuisance_models(
            X_c, W_c, Y_c, 2, is_rct, 
            np.mean(W_c), sample_weight=weights
        )
        scores = _orthogonal_score(Y_c, W_c, mu0, mu1, e)
        est_ate, ci_lower, ci_upper = _get_hansen_hurwitz_ci(scores, q_j, N)
        
        runtime = time.time() - start_time
        
        return {
            'est_ate': est_ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'runtime': runtime
        }
    
    def optimize(self, n_trials=100, storage_name=None, resume=True):
        """
        Run Bayesian optimization to find optimal hyperparameters.
        
        Args:
            n_trials: Number of trials for Optuna
            storage_name: Database file name for checkpointing (None for in-memory)
            resume: Whether to resume from existing study if available
        
        Returns:
            study: Optuna study object with results
        """
        print("\n" + "="*80)
        mode_str = "Multi-Scenario" if self.multi_scenario else self.scenario_name
        print(f"HYPERPARAMETER OPTIMIZATION FOR OS-DML ({mode_str})")
        print("="*80)
        print(f"\nSearch space:")
        print(f"  n_estimators: [20, 150] (step 10)")
        print(f"  delta:        [10^-4, 10^-1] = [0.0001, 0.1] (log10 scale)")
        print(f"                Sampled as: 10^x where x ~ Uniform[-4, -1]")
        print(f"  pilot_ratio:  [0.1, 0.9] (step 0.05)")
        print(f"\nOptimization:")
        print(f"  Algorithm:     Bayesian (TPE)")
        print(f"  Trials:        {n_trials}")
        n_scenarios = len(self.scenarios)
        total_runs = n_trials * self.n_replications * n_scenarios
        print(f"  Replications:  {self.n_replications} per trial per scenario")
        print(f"  Scenarios:     {n_scenarios}")
        print(f"  Total runs:    ~{total_runs}")
        
        # Checkpoint configuration
        if storage_name is None:
            storage_name = f'optuna_{self.scenario_name}.db'
        storage_url = f'sqlite:///{storage_name}'
        
        print(f"\nCheckpoint:")
        print(f"  Storage:       {storage_name}")
        print(f"  Resume:        {resume}")
        
        # Parallel execution configuration
        n_jobs = self.n_jobs
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        
        print(f"\nParallel Execution:")
        print(f"  CPUs:          {n_jobs if n_jobs > 1 else '1 (sequential)'}")
        
        if n_jobs > 1:
            total_time_estimate = total_runs * 2 / 60 / n_jobs
        else:
            total_time_estimate = total_runs * 2 / 60
        print(f"\nEstimated time: ~{total_time_estimate:.0f} minutes")
        
        # Create or load study with checkpoint support
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f'os_dml_{self.scenario_name}',
            storage=storage_url,
            load_if_exists=resume
        )
        
        # Check if resuming
        if len(study.trials) > 0 and resume:
            print(f"\n✓ Resuming from existing study with {len(study.trials)} completed trials")
            remaining_trials = max(0, n_trials - len(study.trials))
            if remaining_trials == 0:
                print(f"  Study already has {len(study.trials)} trials (requested {n_trials})")
                print(f"  To run more trials, increase n_trials parameter")
                return study
            else:
                print(f"  Running {remaining_trials} additional trials...")
                n_trials = remaining_trials
        
        # Optimize
        print("\nStarting optimization...")
        print("-"*80)
        
        study.optimize(
            self.objective, 
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=n_jobs,
            catch=(Exception,)  # Continue on errors in parallel mode
        )
        
        print(f"\n✓ Study saved to {storage_name}")
        print(f"  Total trials completed: {len(study.trials)}")
        
        return study


def analyze_optimization_results(study, scenarios, multi_scenario=False):
    """
    Analyze and visualize optimization results.
    
    Args:
        study: Optuna study object
        scenarios: Dict of scenarios or scenario name
        multi_scenario: Whether this is multi-scenario optimization
    """
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    # Best trial
    best_trial = study.best_trial
    
    print(f"\nBest Trial (Trial #{best_trial.number}):")
    print("-"*80)
    print(f"  Objective Value (Agg. RMSE + penalty): {best_trial.value:.4f}")
    print(f"\n  Optimal Hyperparameters:")
    print(f"    n_estimators:  {best_trial.params['n_estimators']}")
    # delta was sampled via log10_delta
    best_delta = 10 ** best_trial.params['log10_delta']
    print(f"    log10(delta):  {best_trial.params['log10_delta']:.2f}")
    print(f"    delta:         {best_delta:.6f}")
    print(f"    pilot_ratio:   {best_trial.params['pilot_ratio']:.2f}")
    print(f"    r0:            {best_trial.user_attrs['r0']}")
    print(f"    r1:            {best_trial.user_attrs['r1']}")
    
    # Display per-scenario performance if multi-scenario
    if multi_scenario:
        print(f"\n  Aggregated Performance:")
        print(f"    Agg. RMSE:     {best_trial.user_attrs['agg_rmse']:.4f}")
        print(f"    Agg. Coverage: {best_trial.user_attrs['agg_coverage']:.4f}")
        
        print(f"\n  Per-Scenario Performance:")
        print(f"  {'Scenario':<10} {'RMSE':<10} {'Coverage':<10} {'Bias':<10}")
        print(f"  {'-'*40}")
        
        for scenario in ['OBS-S', 'OBS-C', 'RCT-S', 'RCT-C']:
            if f'rmse_{scenario}' in best_trial.user_attrs:
                rmse_s = best_trial.user_attrs[f'rmse_{scenario}']
                cov_s = best_trial.user_attrs[f'coverage_{scenario}']
                bias_s = best_trial.user_attrs[f'bias_{scenario}']
                print(f"  {scenario:<10} {rmse_s:<10.4f} {cov_s:<10.4f} {bias_s:<10.4f}")
    else:
        print(f"\n  Performance Metrics:")
        # For single scenario, look for coverage and bias in user_attrs
        if 'coverage_' in str(best_trial.user_attrs.keys()):
            # Multi-scenario mode
            scenario_name = list(scenarios.keys())[0] if isinstance(scenarios, dict) else scenarios
            print(f"    Coverage:      {best_trial.user_attrs.get(f'coverage_{scenario_name}', 'N/A'):.4f}")
            print(f"    Bias:          {best_trial.user_attrs.get(f'bias_{scenario_name}', 'N/A'):.4f}")
    
    # Save detailed results
    trials_df = study.trials_dataframe()
    suffix = 'all_scenarios' if multi_scenario else (list(scenarios.keys())[0] if isinstance(scenarios, dict) else scenarios)
    trials_df.to_csv(f'optuna_trials_{suffix}.csv', index=False)
    print(f"\n✓ All trials saved to: optuna_trials_{suffix}.csv")
    
    # Extract best parameters
    best_delta = 10 ** best_trial.params['log10_delta']
    best_params = {
        'n_estimators': best_trial.params['n_estimators'],
        'log10_delta': best_trial.params['log10_delta'],
        'delta': best_delta,
        'pilot_ratio': best_trial.params['pilot_ratio'],
        'r0': best_trial.user_attrs['r0'],
        'r1': best_trial.user_attrs['r1'],
        'agg_rmse': best_trial.user_attrs.get('agg_rmse', best_trial.value),
        'agg_coverage': best_trial.user_attrs.get('agg_coverage', 0.95),
        'objective': best_trial.value
    }
    
    # Add per-scenario metrics if available
    for scenario in ['OBS-S', 'OBS-C', 'RCT-S', 'RCT-C']:
        if f'rmse_{scenario}' in best_trial.user_attrs:
            best_params[f'rmse_{scenario}'] = best_trial.user_attrs[f'rmse_{scenario}']
            best_params[f'coverage_{scenario}'] = best_trial.user_attrs[f'coverage_{scenario}']
            best_params[f'bias_{scenario}'] = best_trial.user_attrs[f'bias_{scenario}']
    
    return trials_df, best_params


def create_matplotlib_visualizations(study, scenario):
    """
    Create comprehensive visualizations using matplotlib (server-friendly, no browser needed).
    """
    print("\n" + "="*80)
    print("CREATING MATPLOTLIB VISUALIZATIONS (Server-friendly)")
    print("="*80)
    
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    
    if len(completed_trials) == 0:
        print("  No completed trials to visualize")
        return
    
    # 1. Optimization History
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(completed_trials['number'], completed_trials['value'], 'o-', alpha=0.6, label='Objective Value')
        # Plot best value so far
        best_values = completed_trials['value'].cummin()
        ax.plot(completed_trials['number'], best_values, 'r-', linewidth=2, label='Best Value')
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Objective Value (RMSE + penalty)', fontsize=12)
        ax.set_title('Optimization History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'optuna_history_{scenario}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Optimization history: optuna_history_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create optimization history: {e}")
    
    # 2. Parameter Importances (using random forest importance from optuna)
    try:
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        params = list(importances.keys())
        values = list(importances.values())
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
        
        bars = ax.barh(params, values, color=colors)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Parameters', fontsize=12)
        ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'optuna_importance_{scenario}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Parameter importances: optuna_importance_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create parameter importance plot: {e}")
    
    # 3. Parallel Coordinate Plot
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Normalize parameters to [0, 1]
        param_cols = ['params_n_estimators', 'params_log10_delta', 'params_pilot_ratio']
        normalized = completed_trials[param_cols].copy()
        for col in param_cols:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        
        # Color by objective value
        colors = completed_trials['value']
        norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())
        cmap = plt.cm.RdYlGn_r
        
        # Plot each trial
        for idx, row in normalized.iterrows():
            trial_idx = completed_trials.index.get_loc(idx)
            color = cmap(norm(colors.iloc[trial_idx]))
            ax.plot(range(len(param_cols)), row[param_cols], 'o-', alpha=0.3, color=color, linewidth=1)
        
        ax.set_xticks(range(len(param_cols)))
        ax.set_xticklabels(['n_estimators', 'log10_delta', 'pilot_ratio'], fontsize=11)
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_title('Parallel Coordinate Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Objective Value', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'optuna_parallel_{scenario}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Parallel coordinate plot: optuna_parallel_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create parallel coordinate plot: {e}")
    
    # 4. Contour plots for parameter pairs
    _create_contour_plot(completed_trials, 'params_n_estimators', 'params_log10_delta', 
                        'n_estimators', 'log10_delta', scenario, 'gbm_delta')
    _create_contour_plot(completed_trials, 'params_n_estimators', 'params_pilot_ratio',
                        'n_estimators', 'pilot_ratio', scenario, 'gbm_pilot')
    _create_contour_plot(completed_trials, 'params_log10_delta', 'params_pilot_ratio',
                        'log10_delta', 'pilot_ratio', scenario, 'delta_pilot')


def _create_contour_plot(trials_df, x_col, y_col, x_label, y_label, scenario, suffix):
    """Helper function to create 2D contour plots"""
    try:
        from scipy.interpolate import griddata
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        x = trials_df[x_col].values
        y = trials_df[y_col].values
        z = trials_df['value'].values
        
        # Create grid
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate
        Zi = griddata((x, y), z, (Xi, Yi), method='cubic', fill_value=z.mean())
        
        # Contour plot
        contour = ax.contourf(Xi, Yi, Zi, levels=20, cmap='RdYlGn_r', alpha=0.8)
        ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Scatter actual trials
        scatter = ax.scatter(x, y, c=z, s=50, cmap='RdYlGn_r', edgecolors='black', 
                           linewidth=1, alpha=0.9, zorder=5)
        
        # Mark best trial
        best_idx = z.argmin()
        ax.scatter(x[best_idx], y[best_idx], s=300, c='red', marker='*', 
                  edgecolors='white', linewidth=2, label='Best Trial', zorder=10)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'2D Contour: {x_label} vs {y_label}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Objective Value', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'optuna_contour_{suffix}_{scenario}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Contour ({x_label} vs {y_label}): optuna_contour_{suffix}_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create {suffix} contour: {e}")


def create_optuna_visualizations(study, scenario):
    """
    Create comprehensive visualizations of optimization results.
    Uses matplotlib by default (server-friendly), falls back to Plotly if needed.
    """
    # Try matplotlib version first (no browser needed)
    try:
        create_matplotlib_visualizations(study, scenario)
        return
    except Exception as e:
        print(f"\nMatplotlib visualizations failed: {e}")
        print("Falling back to Plotly (may require browser/kaleido)...\n")
    
    # Fallback to Plotly (original implementation)
    print("\n" + "="*80)
    print("CREATING OPTUNA VISUALIZATIONS (Plotly)")
    print("="*80)
    
    # 1. Optimization History
    try:
        fig = plot_optimization_history(study)
        fig.write_image(f'optuna_history_{scenario}.pdf')
        print(f"✓ Optimization history: optuna_history_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create optimization history: {e}")
    
    # 2. Parameter Importances
    try:
        fig = plot_param_importances(study)
        fig.write_image(f'optuna_importance_{scenario}.pdf')
        print(f"✓ Parameter importances: optuna_importance_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create parameter importance plot: {e}")
    
    # 3. Parallel Coordinate Plot
    try:
        fig = plot_parallel_coordinate(study)
        fig.write_image(f'optuna_parallel_{scenario}.pdf')
        print(f"✓ Parallel coordinate plot: optuna_parallel_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create parallel coordinate plot: {e}")
    
    # 4. Contour plots for parameter pairs
    try:
        fig = plot_contour(study, params=['n_estimators', 'log10_delta'])
        fig.write_image(f'optuna_contour_gbm_delta_{scenario}.pdf')
        print(f"✓ Contour (n_estimators vs log10_delta): optuna_contour_gbm_delta_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create GBM-delta contour: {e}")
    
    try:
        fig = plot_contour(study, params=['n_estimators', 'pilot_ratio'])
        fig.write_image(f'optuna_contour_gbm_pilot_{scenario}.pdf')
        print(f"✓ Contour (n_estimators vs pilot_ratio): optuna_contour_gbm_pilot_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create GBM-pilot contour: {e}")
    
    try:
        fig = plot_contour(study, params=['log10_delta', 'pilot_ratio'])
        fig.write_image(f'optuna_contour_delta_pilot_{scenario}.pdf')
        print(f"✓ Contour (log10_delta vs pilot_ratio): optuna_contour_delta_pilot_{scenario}.pdf")
    except Exception as e:
        print(f"  Could not create delta-pilot contour: {e}")


def create_custom_analysis_plots(trials_df, scenario, best_params):
    """
    Create custom analysis plots showing parameter relationships.
    """
    print("\n" + "="*80)
    print("CREATING CUSTOM ANALYSIS PLOTS")
    print("="*80)
    
    # Extract parameters and metrics
    # Handle both single-scenario and multi-scenario modes
    coverage_col = 'user_attrs_agg_coverage' if 'user_attrs_agg_coverage' in trials_df.columns else 'user_attrs_coverage'
    trials_df['rmse_only'] = trials_df['value'] - abs(trials_df[coverage_col] - 0.95) * 0.5
    trials_df['coverage'] = trials_df[coverage_col]
    # Convert log10_delta to delta for plotting
    trials_df['params_delta'] = 10 ** trials_df['params_log10_delta']
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    # Row 1: Each parameter vs RMSE
    # n_estimators vs RMSE
    axes[0, 0].scatter(trials_df['params_n_estimators'], trials_df['rmse_only'], 
                      alpha=0.5, s=30, c=trials_df['coverage'], cmap='RdYlGn', vmin=0.85, vmax=1.0)
    axes[0, 0].axvline(x=best_params['n_estimators'], color='red', linestyle='--', label='Optimal')
    axes[0, 0].axvline(x=30, color='orange', linestyle=':', label='Current')
    axes[0, 0].set_xlabel('n_estimators (Pilot GBM Complexity)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('GBM Complexity vs RMSE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # delta vs RMSE (on log10 scale)
    axes[0, 1].scatter(trials_df['params_delta'], trials_df['rmse_only'], 
                      alpha=0.5, s=30, c=trials_df['coverage'], cmap='RdYlGn', vmin=0.85, vmax=1.0)
    axes[0, 1].axvline(x=best_params['delta'], color='red', linestyle='--', label='Optimal')
    axes[0, 1].axvline(x=0.01, color='orange', linestyle=':', label='Current')
    axes[0, 1].set_xlabel('Delta (Stabilization Constant)')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Delta vs RMSE (log10 scale)')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # pilot_ratio vs RMSE
    axes[0, 2].scatter(trials_df['params_pilot_ratio'], trials_df['rmse_only'], 
                      alpha=0.5, s=30, c=trials_df['coverage'], cmap='RdYlGn', vmin=0.85, vmax=1.0)
    axes[0, 2].axvline(x=best_params['pilot_ratio'], color='red', linestyle='--', label='Optimal')
    axes[0, 2].axvline(x=0.3, color='orange', linestyle=':', label='Current')
    axes[0, 2].set_xlabel('Pilot Ratio (r0 / r_total)')
    axes[0, 2].set_ylabel('RMSE')
    axes[0, 2].set_title('Pilot Size vs RMSE')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Each parameter vs Coverage
    # n_estimators vs Coverage
    axes[1, 0].scatter(trials_df['params_n_estimators'], trials_df['coverage'], 
                      alpha=0.5, s=30, c=trials_df['rmse_only'], cmap='viridis_r')
    axes[1, 0].axhline(y=0.95, color='red', linestyle='--', label='Nominal 95%')
    axes[1, 0].axvline(x=best_params['n_estimators'], color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('n_estimators')
    axes[1, 0].set_ylabel('Coverage')
    axes[1, 0].set_title('GBM Complexity vs Coverage')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.7, 1.0])
    
    # delta vs Coverage
    axes[1, 1].scatter(trials_df['params_delta'], trials_df['coverage'], 
                      alpha=0.5, s=30, c=trials_df['rmse_only'], cmap='viridis_r')
    axes[1, 1].axhline(y=0.95, color='red', linestyle='--', label='Nominal 95%')
    axes[1, 1].axvline(x=best_params['delta'], color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Delta')
    axes[1, 1].set_ylabel('Coverage')
    axes[1, 1].set_title('Delta vs Coverage')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0.7, 1.0])
    
    # pilot_ratio vs Coverage
    axes[1, 2].scatter(trials_df['params_pilot_ratio'], trials_df['coverage'], 
                      alpha=0.5, s=30, c=trials_df['rmse_only'], cmap='viridis_r')
    axes[1, 2].axhline(y=0.95, color='red', linestyle='--', label='Nominal 95%')
    axes[1, 2].axvline(x=best_params['pilot_ratio'], color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Pilot Ratio')
    axes[1, 2].set_ylabel('Coverage')
    axes[1, 2].set_title('Pilot Size vs Coverage')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0.7, 1.0])
    
    # Row 3: Parameter interactions (2D heatmaps)
    # Create pivot tables for heatmaps
    
    # n_estimators vs delta
    pivot_1 = trials_df.pivot_table(
        values='rmse_only', 
        index='params_n_estimators', 
        columns=pd.cut(trials_df['params_delta'], bins=5),
        aggfunc='mean'
    )
    if not pivot_1.empty:
        sns.heatmap(pivot_1, ax=axes[2, 0], cmap='RdYlGn_r', annot=False, cbar_kws={'label': 'RMSE'})
        axes[2, 0].set_xlabel('Delta (binned)')
        axes[2, 0].set_ylabel('n_estimators')
        axes[2, 0].set_title('Interaction: GBM × Delta')
    
    # n_estimators vs pilot_ratio
    pivot_2 = trials_df.pivot_table(
        values='rmse_only',
        index='params_n_estimators',
        columns=pd.cut(trials_df['params_pilot_ratio'], bins=5),
        aggfunc='mean'
    )
    if not pivot_2.empty:
        sns.heatmap(pivot_2, ax=axes[2, 1], cmap='RdYlGn_r', annot=False, cbar_kws={'label': 'RMSE'})
        axes[2, 1].set_xlabel('Pilot Ratio (binned)')
        axes[2, 1].set_ylabel('n_estimators')
        axes[2, 1].set_title('Interaction: GBM × Pilot Size')
    
    # delta vs pilot_ratio
    pivot_3 = trials_df.pivot_table(
        values='rmse_only',
        index=pd.cut(trials_df['params_delta'], bins=5),
        columns=pd.cut(trials_df['params_pilot_ratio'], bins=5),
        aggfunc='mean'
    )
    if not pivot_3.empty:
        sns.heatmap(pivot_3, ax=axes[2, 2], cmap='RdYlGn_r', annot=False, cbar_kws={'label': 'RMSE'})
        axes[2, 2].set_xlabel('Pilot Ratio (binned)')
        axes[2, 2].set_ylabel('Delta (binned)')
        axes[2, 2].set_title('Interaction: Delta × Pilot Size')
    
    plt.tight_layout()
    plt.savefig(f'hyperparameter_analysis_{scenario}.pdf', dpi=300, bbox_inches='tight')
    print(f"\n✓ Custom analysis plot: hyperparameter_analysis_{scenario}.pdf")


def compare_with_baseline(best_params, scenario):
    """
    Compare optimal configuration with current baseline.
    """
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    
    # Current configuration
    current = {
        'n_estimators': 30,
        'delta': 0.01,
        'pilot_ratio': 0.3,
        'r0': 3000,
        'r1': 7000
    }
    
    print("\n" + "-"*80)
    print("CONFIGURATION COMPARISON")
    print("-"*80)
    print(f"\n{'Parameter':<20} {'Current':<15} {'Optimal':<15} {'Change':<10}")
    print("-"*80)
    print(f"{'n_estimators':<20} {current['n_estimators']:<15} {best_params['n_estimators']:<15} "
          f"{'↑' if best_params['n_estimators'] > current['n_estimators'] else '→' if best_params['n_estimators'] == current['n_estimators'] else '↓'}")
    print(f"{'delta':<20} {current['delta']:<15.4f} {best_params['delta']:<15.6f} "
          f"{'↓' if best_params['delta'] < current['delta'] else '→' if best_params['delta'] == current['delta'] else '↑'}")
    print(f"{'pilot_ratio':<20} {current['pilot_ratio']:<15.2f} {best_params['pilot_ratio']:<15.2f} "
          f"{'↑' if best_params['pilot_ratio'] > current['pilot_ratio'] else '→' if best_params['pilot_ratio'] == current['pilot_ratio'] else '↓'}")
    print(f"{'r0':<20} {current['r0']:<15} {best_params['r0']:<15} "
          f"{'↑' if best_params['r0'] > current['r0'] else '→' if best_params['r0'] == current['r0'] else '↓'}")
    print(f"{'r1':<20} {current['r1']:<15} {best_params['r1']:<15} "
          f"{'↓' if best_params['r1'] < current['r1'] else '→' if best_params['r1'] == current['r1'] else '↑'}")
    
    print("\n" + "-"*80)
    print("CODE CHANGES NEEDED")
    print("-"*80)
    
    changes_needed = []
    
    if best_params['n_estimators'] != current['n_estimators']:
        changes_needed.append(
            f"1. methods.py, line 304:\n"
            f"   'n_estimators': {best_params['n_estimators']}  # was {current['n_estimators']}"
        )
    
    if abs(best_params['delta'] - current['delta']) > 0.001:
        changes_needed.append(
            f"2. methods.py, line 292:\n"
            f"   delta = kwargs.get('delta', {best_params['delta']:.6f})  # was {current['delta']}"
        )
    
    if best_params['r0'] != current['r0']:
        changes_needed.append(
            f"3. config.py, line 81:\n"
            f"   \"r0\": {best_params['r0']}, \"r1\": {best_params['r1']}  # was r0={current['r0']}, r1={current['r1']}"
        )
    
    if changes_needed:
        print("\nApply these changes:")
        for change in changes_needed:
            print(change)
    else:
        print("\n✓ Current configuration is already optimal!")


def test_optimal_configuration(best_params, scenarios_dict, n_test_reps=30, multi_scenario=False):
    """
    Test the optimal configuration with more replications for robust evaluation.
    Tests on all scenarios to show comprehensive performance.
    """
    print("\n" + "="*80)
    print("TESTING OPTIMAL CONFIGURATION")
    print("="*80)
    print(f"\nRunning {n_test_reps} replications per scenario with optimal parameters...")
    
    all_test_results = {}
    
    # Test on each scenario
    for scenario_name, scenario_config in scenarios_dict.items():
        print(f"\n  Testing on {scenario_name}...")
        
        test_results = []
        
        for rep in range(n_test_reps):
            np.random.seed(config.BASE_SEED + 1000 + rep)  # Different seeds
            data = scenario_config['func'](**scenario_config['params'])
            true_ate = data['true_ate']
            
            # Create a temporary optimizer for this scenario
            temp_opt = OSHyperparameterOptimizer(scenarios=scenario_name)
            result = temp_opt._run_os_with_params(
                data,
                n_estimators=best_params['n_estimators'],
                delta=best_params['delta'],
                r0=best_params['r0'],
                r1=best_params['r1'],
                is_rct=scenario_config['is_rct']
            )
            
            bias = result['est_ate'] - true_ate
            sq_error = bias**2
            coverage = (result['ci_lower'] <= true_ate <= result['ci_upper'])
            ci_width = result['ci_upper'] - result['ci_lower']
            
            test_results.append({
                'scenario': scenario_name,
                'est_ate': result['est_ate'],
                'bias': bias,
                'sq_error': sq_error,
                'coverage': coverage,
                'ci_width': ci_width,
                'runtime': result['runtime']
            })
        
        df_test_scenario = pd.DataFrame(test_results)
        all_test_results[scenario_name] = df_test_scenario
    
    # Print summary table
    print(f"\n\nOptimal Configuration Performance ({n_test_reps} reps per scenario):")
    print("="*80)
    print(f"{'Scenario':<10} {'RMSE':<10} {'Bias':<10} {'Coverage':<10} {'CI Width':<10} {'Runtime':<10}")
    print("-"*80)
    
    summary_rows = []
    for scenario_name, df_test in all_test_results.items():
        rmse = np.sqrt(df_test['sq_error'].mean())
        bias = df_test['bias'].mean()
        coverage = df_test['coverage'].mean()
        ci_width = df_test['ci_width'].mean()
        runtime = df_test['runtime'].mean()
        
        print(f"{scenario_name:<10} {rmse:<10.4f} {bias:<10.4f} {coverage:<10.4f} {ci_width:<10.4f} {runtime:<10.2f}")
        
        summary_rows.append({
            'scenario': scenario_name,
            'rmse': rmse,
            'bias': bias,
            'coverage': coverage,
            'ci_width': ci_width,
            'runtime': runtime
        })
    
    # Overall statistics
    df_summary = pd.DataFrame(summary_rows)
    print("-"*80)
    print(f"{'AVERAGE':<10} {df_summary['rmse'].mean():<10.4f} {df_summary['bias'].mean():<10.4f} "
          f"{df_summary['coverage'].mean():<10.4f} {df_summary['ci_width'].mean():<10.4f} {df_summary['runtime'].mean():<10.2f}")
    
    # Compare with baseline from experiment results
    if multi_scenario:
        try:
            baseline_df = pd.read_csv('analysis_results/experiment_2_main_comparison/summary_table.csv')
            baseline_os = baseline_df[baseline_df['method'] == 'OS']
            
            print("\n" + "="*80)
            print("IMPROVEMENT vs BASELINE (Per Scenario)")
            print("="*80)
            print(f"{'Scenario':<10} {'RMSE Change':<15} {'Coverage Change':<18} {'Status':<10}")
            print("-"*80)
            
            for scenario_name in summary_rows:
                sname = scenario_name['scenario']
                baseline = baseline_os[baseline_os['scenario'] == sname]
                if not baseline.empty:
                    baseline = baseline.iloc[0]
                    opt_rmse = scenario_name['rmse']
                    opt_cov = scenario_name['coverage']
                    
                    rmse_change = (baseline['RMSE'] - opt_rmse) / baseline['RMSE'] * 100
                    cov_change = (opt_cov - baseline['Coverage']) * 100
                    
                    status = '✓✓' if rmse_change > 10 and opt_cov >= 0.93 else '✓' if rmse_change > 0 else '→'
                    
                    print(f"{sname:<10} {rmse_change:>6.1f}%         {cov_change:>+6.1f}pp           {status:<10}")
            
            print("\n  ✓✓ = Substantial improvement")
            print("  ✓  = Modest improvement")
            print("  →  = No improvement")
            
        except Exception as e:
            print(f"\nCould not load baseline for comparison: {e}")
    
    # Combine all results
    df_all = pd.concat(all_test_results.values(), ignore_index=True)
    
    return df_all, df_summary


if __name__ == "__main__":
    
    print("="*80)
    print("OS-DML HYPERPARAMETER OPTIMIZATION using Optuna")
    print("="*80)
    print("\nThis script uses Bayesian optimization (TPE) to find optimal hyperparameters")
    print("for OS-DML across ALL scenarios (OBS-S, OBS-C, RCT-S, RCT-C).")
    print("\nHyperparameters being tuned:")
    print("  1. n_estimators: GBM complexity in pilot [20, 150]")
    print("  2. delta: Stabilization constant [10^-4, 10^-1] (log10 scale)")
    print("     → Sampled as: log10(delta) ~ Uniform[-4, -1], then delta = 10^x")
    print("  3. pilot_ratio: r0/(r0+r1) [0.1, 0.9]")
    
    # Configuration
    MULTI_SCENARIO = True    # Optimize across all scenarios
    N_TRIALS = 60            # More trials for multi-scenario (60-100 recommended)
    N_REPS = 10              # MC replications per trial per scenario
    AGGREGATION = 'mean'     # 'mean', 'max' (minimax), or 'weighted'
    N_JOBS = -1              # Number of parallel CPUs (-1 = all available, 1 = sequential)
    CHECKPOINT = True        # Enable checkpoint (save progress to database)
    RESUME = True            # Resume from checkpoint if exists
    
    print(f"\nConfiguration:")
    print(f"  Mode:          {'Multi-scenario (ALL 4)' if MULTI_SCENARIO else 'Single scenario'}")
    print(f"  Aggregation:   {AGGREGATION}")
    print(f"  Trials:        {N_TRIALS}")
    print(f"  Replications:  {N_REPS} per scenario per trial")
    print(f"  Parallel CPUs: {'All available' if N_JOBS == -1 else N_JOBS}")
    print(f"  Checkpoint:    {'Enabled' if CHECKPOINT else 'Disabled'}")
    print(f"  Resume:        {'Yes' if RESUME else 'No'}")
    
    if MULTI_SCENARIO:
        total_runs = N_TRIALS * N_REPS * 4
        print(f"  Total runs:    {total_runs} (4 scenarios)")
        import multiprocessing
        n_cpus = multiprocessing.cpu_count() if N_JOBS == -1 else max(1, N_JOBS)
        est_time = total_runs * 2 / 60 / n_cpus if N_JOBS != 1 else total_runs * 2 / 60
        print(f"  Est. time:     ~{est_time:.0f} minutes (with {n_cpus} CPUs)")
    else:
        print(f"  Total runs:    {N_TRIALS * N_REPS}")
    
    print(f"\nNote: Optuna intelligently explores the space.")
    print(f"      {AGGREGATION.capitalize()} aggregation finds robust parameters across scenarios.")
    if CHECKPOINT:
        print(f"      Progress is auto-saved. Safe to interrupt (Ctrl+C) and resume later.")
    
    # Create optimizer
    optimizer = OSHyperparameterOptimizer(
        scenarios='all' if MULTI_SCENARIO else 'OBS-S',
        n_replications=N_REPS,
        r_total=10000,
        aggregation=AGGREGATION,
        n_jobs=N_JOBS
    )
    
    # Run optimization with checkpoint support
    start_time = time.time()
    storage_file = f'optuna_{optimizer.scenario_name}.db' if CHECKPOINT else None
    study = optimizer.optimize(
        n_trials=N_TRIALS,
        storage_name=storage_file,
        resume=RESUME
    )
    elapsed = time.time() - start_time
    
    print(f"\n✓ Optimization complete in {elapsed/60:.1f} minutes")
    
    # Analyze results
    suffix = 'all_scenarios' if MULTI_SCENARIO else 'OBS-S'
    trials_df, best_params = analyze_optimization_results(study, optimizer.scenarios, MULTI_SCENARIO)
    
    # Create visualizations
    create_optuna_visualizations(study, suffix)
    create_custom_analysis_plots(trials_df, suffix, best_params)
    
    # Compare with baseline
    compare_with_baseline(best_params, suffix)
    
    # Test optimal configuration on all scenarios
    df_test, df_summary = test_optimal_configuration(
        best_params, optimizer.scenarios, n_test_reps=30, multi_scenario=MULTI_SCENARIO
    )
    
    # Save final recommendation
    recommendation = {
        'optimization_mode': 'multi_scenario' if MULTI_SCENARIO else 'single_scenario',
        'aggregation_method': AGGREGATION,
        'n_estimators_optimal': best_params['n_estimators'],
        'delta_optimal': best_params['delta'],
        'r0_optimal': best_params['r0'],
        'r1_optimal': best_params['r1'],
        'pilot_ratio_optimal': best_params['pilot_ratio'],
    }
    
    # Add per-scenario metrics
    for _, row in df_summary.iterrows():
        scenario = row['scenario']
        recommendation[f'rmse_{scenario}'] = row['rmse']
        recommendation[f'coverage_{scenario}'] = row['coverage']
        recommendation[f'bias_{scenario}'] = row['bias']
    
    # Add aggregate metrics
    recommendation['rmse_mean'] = df_summary['rmse'].mean()
    recommendation['coverage_mean'] = df_summary['coverage'].mean()
    recommendation['bias_mean_abs'] = df_summary['bias'].abs().mean()
    recommendation['rmse_max'] = df_summary['rmse'].max()
    recommendation['coverage_min'] = df_summary['coverage'].min()
    
    pd.DataFrame([recommendation]).to_csv(f'optimal_parameters_{suffix}.csv', index=False)
    df_summary.to_csv(f'optimal_performance_by_scenario_{suffix}.csv', index=False)
    
    print("\n" + "="*80)
    print("ALL OUTPUTS SAVED")
    print("="*80)
    
    checkpoint_info = ""
    if CHECKPOINT:
        checkpoint_info = f"""
    Checkpoint Database:
      • {storage_file} - Study progress (can resume from this)
      • To resume: Run script again with RESUME=True
      • To restart: Delete this file or set RESUME=False
    """
    
    print(f"""
    CSV Files:
      • optuna_trials_{suffix}.csv - All trials data ({len(trials_df)} trials)
      • optimal_parameters_{suffix}.csv - Final recommendation
      • optimal_performance_by_scenario_{suffix}.csv - Per-scenario performance
    {checkpoint_info}
    Optimization Analysis Plots (matplotlib, server-friendly):
      • optuna_history_{suffix}.pdf - Optimization convergence
      • optuna_importance_{suffix}.pdf - Parameter importance ⭐ KEY
      • optuna_parallel_{suffix}.pdf - Parallel coordinate plot
      • optuna_contour_gbm_delta_{suffix}.pdf - 2D contour
      • optuna_contour_gbm_pilot_{suffix}.pdf - 2D contour
      • optuna_contour_delta_pilot_{suffix}.pdf - 2D contour
    
    Custom Analysis:
      • hyperparameter_analysis_{suffix}.pdf - 9-panel comprehensive plot ⭐ KEY
    
    Next Steps:
      1. Review hyperparameter_analysis_{suffix}.pdf for parameter relationships
      2. Check optuna_importance_{suffix}.pdf for which parameter matters most
      3. Review optimal_performance_by_scenario_{suffix}.csv for per-scenario results
      4. Apply recommended changes from console output above
      5. Re-run experiments with optimal configuration
    """)
    
    # Print final summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print("\nOptimal Parameters (Robust Across All Scenarios):")
    print(f"  n_estimators = {best_params['n_estimators']}")
    print(f"  delta = {best_params['delta']:.6f}")
    print(f"  r0 = {best_params['r0']}, r1 = {best_params['r1']}")
    
    print("\nPerformance on Each Scenario:")
    print(df_summary.to_string(index=False))
    
    print("\n\nAggregate Performance:")
    print(f"  Average RMSE:      {df_summary['rmse'].mean():.4f}")
    print(f"  Average Coverage:  {df_summary['coverage'].mean():.4f}")
    print(f"  Worst RMSE:        {df_summary['rmse'].max():.4f}")
    print(f"  Worst Coverage:    {df_summary['coverage'].min():.4f}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

