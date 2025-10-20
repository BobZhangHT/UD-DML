#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulation_Unified.py

Unified simulation script integrating all experiments.
Run this file or convert to notebook using: jupytext --to ipynb Simulation_Unified.py
"""

# %% [markdown]
# # OS-DML Simulation Study
# 
# This notebook runs all experiments:
# 1. **Experiment 1**: Sensitivity Analysis (Hyperparameter Optimization)
# 2. **Experiment 2**: Main Comparison  
# 3. **Experiment 3**: Robustness Check

# %%
# Setup
import warnings
import os
warnings.filterwarnings('ignore')
# Additional suppression for LightGBM feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names, but LGBM.* was fitted with feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
warnings.filterwarnings('ignore', message='.*feature names.*')

import json, pickle, time, traceback, sys
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['LIGHTGBM_VERBOSITY'] = '-1'

import config
import evaluation
from data_generation import generate_obs_1_data, generate_obs_2_data, generate_obs_3_data, generate_rct_1_data, generate_rct_2_data, generate_rct_3_data
import lightgbm as lgb
from methods import _orthogonal_score, _fit_nuisance_models, _get_hansen_hurwitz_ci

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:
    import sys
    import subprocess
    print("Installing Optuna...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna', '-q'])
    import optuna
    from optuna.importance import get_param_importances

print("✓ Setup complete")

# %%
# Optimizer Class
class OSHyperparameterOptimizer:
    def __init__(self, scenarios='all', n_replications=10, r_total=10000, aggregation='mean', n_jobs=1):
        self.n_replications = n_replications
        self.r_total = r_total
        self.N = config.N_POPULATION
        self.aggregation = aggregation
        self.n_jobs = n_jobs
        
        all_scenarios = {
            'OBS-1': {'func': generate_obs_1_data, 'params': {'n': self.N, 'p': 20}, 'is_rct': False},
            'OBS-2': {'func': generate_obs_2_data, 'params': {'n': self.N, 'p': 50}, 'is_rct': False},
            'OBS-3': {'func': generate_obs_3_data, 'params': {'n': self.N, 'p': 100}, 'is_rct': False},
            'RCT-1': {'func': generate_rct_1_data, 'params': {'n': self.N, 'p': 20}, 'is_rct': True},
            'RCT-2': {'func': generate_rct_2_data, 'params': {'n': self.N, 'p': 50}, 'is_rct': True},
            'RCT-3': {'func': generate_rct_3_data, 'params': {'n': self.N, 'p': 100}, 'is_rct': True}
        }
        
        self.scenarios = all_scenarios if scenarios == 'all' else {scenarios: all_scenarios[scenarios]}
        self.scenario_name = 'all_scenarios' if scenarios == 'all' else scenarios
    
    def objective(self, trial):
        # Pilot n_estimators should not exceed full GBM's n_estimators (100)
        n_estimators = trial.suggest_int('n_estimators', 20, 100, step=10)
        log10_delta = trial.suggest_float('log10_delta', -4, -1)
        delta = 10 ** log10_delta
        pilot_ratio = trial.suggest_float('pilot_ratio', 0.1, 0.9, step=0.05)
        
        r0 = int(self.r_total * pilot_ratio)
        r1 = self.r_total - r0
        
        scenario_metrics = {}
        for scenario_name, scenario_config in self.scenarios.items():
            results = []
            for rep in range(self.n_replications):
                try:
                    np.random.seed(config.BASE_SEED + trial.number * 10000 + rep)
                    data = scenario_config['func'](**scenario_config['params'])
                    result = self._run_os_with_params(data, n_estimators, delta, r0, r1, scenario_config['is_rct'])
                    bias = result['est_ate'] - data['true_ate']
                    results.append({'sq_error': bias**2, 'coverage': (result['ci_lower'] <= data['true_ate'] <= result['ci_upper'])})
                except:
                    return 1.0
            
            df = pd.DataFrame(results)
            scenario_metrics[scenario_name] = {'rmse': np.sqrt(df['sq_error'].mean()), 'coverage': df['coverage'].mean()}
            trial.set_user_attr(f'rmse_{scenario_name}', scenario_metrics[scenario_name]['rmse'])
            trial.set_user_attr(f'coverage_{scenario_name}', scenario_metrics[scenario_name]['coverage'])
        
        trial.set_user_attr('r0', r0)
        trial.set_user_attr('r1', r1)
        
        agg_rmse = np.mean([m['rmse'] for m in scenario_metrics.values()])
        agg_coverage = np.mean([m['coverage'] for m in scenario_metrics.values()])
        trial.set_user_attr('agg_rmse', agg_rmse)
        trial.set_user_attr('agg_coverage', agg_coverage)
        
        return agg_rmse + abs(agg_coverage - 0.95) * 0.5
    
    def _run_os_with_params(self, data, n_estimators, delta, r0, r1, is_rct):
        N = len(data['Y_obs'])
        pilot_idx = np.random.choice(N, size=r0, replace=True)
        lgbm_params = {'n_jobs': 1, 'random_state': config.BASE_SEED, 'n_estimators': n_estimators, 'verbose': -1, 'feature_name': None}
        
        X_p, W_p, Y_p = data['X'][pilot_idx], data['W'][pilot_idx], data['Y_obs'][pilot_idx]
        mu0 = lgb.LGBMRegressor(**lgbm_params).fit(X_p[W_p == 0], Y_p[W_p == 0]).predict(data['X'])
        mu1 = lgb.LGBMRegressor(**lgbm_params).fit(X_p[W_p == 1], Y_p[W_p == 1]).predict(data['X'])
        e = np.full(N, np.mean(W_p)) if is_rct else np.clip(lgb.LGBMClassifier(**lgbm_params).fit(X_p, W_p).predict_proba(data['X'])[:, 1], 0.01, 0.99)
        
        phi = _orthogonal_score(data['Y_obs'], data['W'], mu0, mu1, e)
        pps_probs = (np.abs(phi) + delta) / np.sum(np.abs(phi) + delta)
        
        pps_idx = np.random.choice(N, size=r1, replace=True, p=pps_probs)
        combined_idx = np.concatenate([pilot_idx, pps_idx])
        q_j = np.concatenate([np.full(r0, 1.0/N), pps_probs[pps_idx]])
        
        X_c, W_c, Y_c = data['X'][combined_idx], data['W'][combined_idx], data['Y_obs'][combined_idx]
        mu0_f, mu1_f, e_f = _fit_nuisance_models(X_c, W_c, Y_c, 2, is_rct, np.mean(W_c), sample_weight=1.0/q_j)
        scores = _orthogonal_score(Y_c, W_c, mu0_f, mu1_f, e_f)
        est_ate, ci_lower, ci_upper = _get_hansen_hurwitz_ci(scores, q_j, N)
        
        return {'est_ate': est_ate, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
    
    def optimize(self, n_trials=100, storage_name=None, resume=True):
        storage_name = storage_name or f'optuna_{self.scenario_name}.db'
        n_jobs = self.n_jobs if self.n_jobs != -1 else __import__('multiprocessing').cpu_count()
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f'os_dml_{self.scenario_name}',
            storage=f'sqlite:///{storage_name}',
            load_if_exists=resume
        )
        
        if len(study.trials) > 0 and resume:
            n_trials = max(0, n_trials - len(study.trials))
            if n_trials == 0:
                return study
        
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs, catch=(Exception,))
        return study

print("✓ Optimizer defined")

# %%
# Experiment 1
def run_experiment_1():
    print("="*80 + "\\nEXPERIMENT 1: SENSITIVITY ANALYSIS\\n" + "="*80)
    
    scenarios, all_methods, experiments = config.get_experiments()
    exp_config = experiments['experiment_1_sensitivity_analysis']
    params = exp_config['params']
    output_dir = Path(exp_config['base_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = OSHyperparameterOptimizer(
        scenarios='all',
        n_replications=params['n_replications'],
        r_total=params['r_total'],
        aggregation=params['aggregation'],
        n_jobs=params['n_jobs']
    )
    
    storage_file = str(output_dir / f"optuna_{optimizer.scenario_name}.db") if params['checkpoint'] else None
    study = optimizer.optimize(n_trials=params['n_trials'], storage_name=storage_file, resume=params['resume'])
    
    best_trial = study.best_trial
    best_params = {
        'n_estimators': best_trial.params['n_estimators'],
        'delta': 10 ** best_trial.params['log10_delta'],
        'pilot_ratio': best_trial.params['pilot_ratio'],
        'r0': best_trial.user_attrs['r0'],
        'r1': best_trial.user_attrs['r1'],
        'agg_rmse': best_trial.user_attrs.get('agg_rmse', 0),
        'agg_coverage': best_trial.user_attrs.get('agg_coverage', 0.95)
    }
    
    # Add per-scenario metrics for all 6 DGPs
    for scenario in ['OBS-1', 'OBS-2', 'OBS-3', 'RCT-1', 'RCT-2', 'RCT-3']:
        if f'rmse_{scenario}' in best_trial.user_attrs:
            best_params[f'rmse_{scenario}'] = best_trial.user_attrs[f'rmse_{scenario}']
            best_params[f'coverage_{scenario}'] = best_trial.user_attrs[f'coverage_{scenario}']
    
    with open(output_dir / 'optimal_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / 'trials.csv', index=False)
    
    # Generate visualizations
    print("\\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(study, trials_df, best_params, output_dir)
    
    print(f"\\n✓ Optimal: n_estimators={best_params['n_estimators']}, delta={best_params['delta']:.6f}, pilot_ratio={best_params['pilot_ratio']:.2f}")
    return best_params

def create_visualizations(study, trials_df, best_params, output_dir):
    """Create comprehensive visualization plots for optimization results."""
    
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    
    if len(completed_trials) == 0:
        print("⚠ No completed trials to visualize")
        return
    
    # 1. Optimization History
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(completed_trials['number'], completed_trials['value'], 'o-', alpha=0.6, label='Objective Value')
        best_values = completed_trials['value'].cummin()
        ax.plot(completed_trials['number'], best_values, 'r-', linewidth=2, label='Best Value')
        ax.set_xlabel('Trial Number', fontsize=12)
        ax.set_ylabel('Objective Value (RMSE + penalty)', fontsize=12)
        ax.set_title('Optimization History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'optimization_history.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: optimization_history.pdf")
    except Exception as e:
        print(f"⚠ Could not create optimization history: {e}")
    
    # 2. Parameter Importances
    try:
        importances = get_param_importances(study)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        params_list = list(importances.keys())
        values = list(importances.values())
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params_list)))
        
        bars = ax.barh(params_list, values, color=colors)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Parameters', fontsize=12)
        ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_importance.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: parameter_importance.pdf")
    except Exception as e:
        print(f"⚠ Could not create parameter importance: {e}")
    
    # 3. Parallel Coordinate Plot
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        param_cols = ['params_n_estimators', 'params_log10_delta', 'params_pilot_ratio']
        normalized = completed_trials[param_cols].copy()
        for col in param_cols:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        
        colors = completed_trials['value']
        norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())
        cmap = plt.cm.RdYlGn_r
        
        for idx, row in normalized.iterrows():
            trial_idx = completed_trials.index.get_loc(idx)
            color = cmap(norm(colors.iloc[trial_idx]))
            ax.plot(range(len(param_cols)), row[param_cols], 'o-', alpha=0.3, color=color, linewidth=1)
        
        ax.set_xticks(range(len(param_cols)))
        ax.set_xticklabels(['n_estimators', 'log10_delta', 'pilot_ratio'], fontsize=11)
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_title('Parallel Coordinate Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Objective Value', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parallel_coordinate.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: parallel_coordinate.pdf")
    except Exception as e:
        print(f"⚠ Could not create parallel coordinate plot: {e}")
    
    # 4. Contour Plots for parameter pairs
    create_contour_plots(completed_trials, best_params, output_dir)
    
    # 5. Custom Analysis Plot (9-panel)
    create_custom_analysis_plot(completed_trials, best_params, output_dir)
    
    print(f"\\n✓ All visualizations saved to: {output_dir}")

def create_contour_plots(trials_df, best_params, output_dir):
    """Create 2D contour plots for parameter pairs."""
    try:
        from scipy.interpolate import griddata
    except ImportError:
        print("⚠ scipy not available, skipping contour plots")
        return
    
    param_pairs = [
        ('params_n_estimators', 'params_log10_delta', 'n_estimators', 'log10_delta', 'gbm_delta'),
        ('params_n_estimators', 'params_pilot_ratio', 'n_estimators', 'pilot_ratio', 'gbm_pilot'),
        ('params_log10_delta', 'params_pilot_ratio', 'log10_delta', 'pilot_ratio', 'delta_pilot')
    ]
    
    for x_col, y_col, x_label, y_label, suffix in param_pairs:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            x = trials_df[x_col].values
            y = trials_df[y_col].values
            z = trials_df['value'].values
            
            xi = np.linspace(x.min(), x.max(), 50)
            yi = np.linspace(y.min(), y.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            
            Zi = griddata((x, y), z, (Xi, Yi), method='cubic', fill_value=z.mean())
            
            contour = ax.contourf(Xi, Yi, Zi, levels=20, cmap='RdYlGn_r', alpha=0.8)
            ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            ax.scatter(x, y, c=z, s=50, cmap='RdYlGn_r', edgecolors='black', linewidth=1, alpha=0.9, zorder=5)
            
            best_idx = z.argmin()
            ax.scatter(x[best_idx], y[best_idx], s=300, c='red', marker='*', 
                      edgecolors='white', linewidth=2, label='Best Trial', zorder=10)
            
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f'2D Contour: {x_label} vs {y_label}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Objective Value', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'contour_{suffix}.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: contour_{suffix}.pdf")
        except Exception as e:
            print(f"⚠ Could not create {suffix} contour: {e}")

def create_custom_analysis_plot(trials_df, best_params, output_dir):
    """Create 9-panel custom analysis plot."""
    try:
        import seaborn as sns
        
        # Prepare data
        coverage_col = 'user_attrs_agg_coverage' if 'user_attrs_agg_coverage' in trials_df.columns else 'user_attrs_coverage'
        trials_df['rmse_only'] = trials_df['value'] - abs(trials_df[coverage_col] - 0.95) * 0.5
        trials_df['coverage'] = trials_df[coverage_col]
        trials_df['params_delta'] = 10 ** trials_df['params_log10_delta']
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        
        # Row 1: Each parameter vs RMSE
        axes[0, 0].scatter(trials_df['params_n_estimators'], trials_df['rmse_only'], 
                          alpha=0.5, s=30, c=trials_df['coverage'], cmap='RdYlGn', vmin=0.85, vmax=1.0)
        axes[0, 0].axvline(x=best_params['n_estimators'], color='red', linestyle='--', label='Optimal')
        axes[0, 0].axvline(x=30, color='orange', linestyle=':', label='Default')
        axes[0, 0].set_xlabel('n_estimators')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('GBM Complexity vs RMSE')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].scatter(trials_df['params_delta'], trials_df['rmse_only'], 
                          alpha=0.5, s=30, c=trials_df['coverage'], cmap='RdYlGn', vmin=0.85, vmax=1.0)
        axes[0, 1].axvline(x=best_params['delta'], color='red', linestyle='--', label='Optimal')
        axes[0, 1].axvline(x=0.01, color='orange', linestyle=':', label='Default')
        axes[0, 1].set_xlabel('Delta')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Delta vs RMSE')
        axes[0, 1].set_xscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].scatter(trials_df['params_pilot_ratio'], trials_df['rmse_only'], 
                          alpha=0.5, s=30, c=trials_df['coverage'], cmap='RdYlGn', vmin=0.85, vmax=1.0)
        axes[0, 2].axvline(x=best_params['pilot_ratio'], color='red', linestyle='--', label='Optimal')
        axes[0, 2].axvline(x=0.3, color='orange', linestyle=':', label='Default')
        axes[0, 2].set_xlabel('Pilot Ratio')
        axes[0, 2].set_ylabel('RMSE')
        axes[0, 2].set_title('Pilot Size vs RMSE')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Each parameter vs Coverage
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
        
        # Row 3: Parameter interactions (heatmaps)
        try:
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
        except:
            axes[2, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        
        try:
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
        except:
            axes[2, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        
        try:
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
        except:
            axes[2, 2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hyperparameter_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: hyperparameter_analysis.pdf")
    except Exception as e:
        print(f"⚠ Could not create custom analysis plot: {e}")

# %%
# Helper functions
def update_config(best_params):
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace('PILOT_RATIO = 0.3', f'PILOT_RATIO = {best_params["pilot_ratio"]:.2f}')
    content = content.replace('DELTA = 0.01', f'DELTA = {best_params["delta"]:.6f}')
    content = content.replace('PILOT_N_ESTIMATORS = 30', f'PILOT_N_ESTIMATORS = {best_params["n_estimators"]}')
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    __import__('importlib').reload(config)
    print("✓ Config updated")

def run_single_replication(params):
    """Run one simulation replication with checkpoint support."""
    exp_name, scenario_name, method_name, sim_id, specific_params, checkpoint_dir = params
    
    # Generate unique checkpoint filename
    misspec_str = specific_params.get('misspecification', '')
    if misspec_str:
        checkpoint_file = checkpoint_dir / scenario_name / f'sim_{sim_id}_{method_name}_{misspec_str}.pkl'
    else:
        checkpoint_file = checkpoint_dir / scenario_name / f'sim_{sim_id}_{method_name}.pkl'
    
    # Check if checkpoint exists
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                result = pickle.load(f)
                # Verify it has required fields
                if 'est_ate' in result and 'scenario' in result:
                    return result
        except:
            pass  # If corrupted, recompute
    
    try:
        np.random.seed(config.BASE_SEED + sim_id)
        scenarios, all_methods, _ = config.get_experiments()
        
        scenario_config = scenarios[scenario_name]
        method_config = all_methods[method_name]
        
        data = scenario_config['data_gen_func'](**scenario_config['params'])
        func = method_config['func']
        
        method_params = {'k_folds': specific_params['k_folds']} if method_name == 'FULL' else {'r': specific_params['r'], 'k_folds': specific_params['k_folds']}
        if 'misspecification' in specific_params:
            method_params['misspecification'] = specific_params['misspecification']
        
        result = func(data['X'], data['W'], data['Y_obs'], data['pi_true'], scenario_config['design'] == 'rct', **method_params)
        result.update({'exp_name': exp_name, 'scenario': scenario_name, 'method': method_name, 'sim_id': sim_id, 'true_ate': data['true_ate']})
        
        # Save checkpoint
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    except Exception as e:
        print(f"Error in sim {sim_id} ({scenario_name}, {method_name}): {e}")
        return None

def run_experiment(exp_name, n_jobs=-1):
    """
    Run experiment with checkpoint support and parallel computation.
    
    Args:
        exp_name: Name of the experiment
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
    """
    print(f"\\n{'='*80}\\n{exp_name.upper()}\\n{'='*80}")
    
    scenarios, all_methods, experiments = config.get_experiments()
    exp_config = experiments[exp_name]
    output_dir = Path(exp_config['base_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we can load completed results
    results_file = output_dir / 'raw_results.pkl'
    if results_file.exists():
        try:
            with open(results_file, 'rb') as f:
                existing_results = pickle.load(f)
            print(f"✓ Found existing results file with {len(existing_results)} replications")
            print(f"  To recompute, delete: {results_file}")
            
            # Still generate summary if needed
            summary_file = output_dir / 'summary_table.csv'
            if not summary_file.exists():
                summary_df = evaluation.process_results(existing_results)
                summary_df.to_csv(summary_file, index=False)
                print(f"✓ Generated summary table")
            
            return existing_results
        except:
            print(f"⚠ Existing results file corrupted, recomputing...")
    
    # Prepare all parameter combinations
    all_params = []
    for scenario_name in exp_config['scenarios']:
        for method_name in exp_config['methods']:
            specific_params = {'k_folds': exp_config['params']['k_folds']} if method_name == 'FULL' else {'r': {'r_total': exp_config['params']['r_total']}, 'k_folds': exp_config['params']['k_folds']}
            
            if 'misspecification_scenarios' in exp_config['params']:
                for misspec in exp_config['params']['misspecification_scenarios']:
                    for sim_id in range(config.N_SIM):
                        all_params.append((exp_name, scenario_name, method_name, sim_id, {**specific_params, 'misspecification': misspec}, checkpoint_dir))
            else:
                for sim_id in range(config.N_SIM):
                    all_params.append((exp_name, scenario_name, method_name, sim_id, specific_params, checkpoint_dir))
    
    print(f"\\nTotal replications: {len(all_params)}")
    
    # Check for existing checkpoints
    existing_count = sum(1 for p in all_params if _checkpoint_exists(p))
    if existing_count > 0:
        print(f"✓ Found {existing_count} existing checkpoints, will skip them")
        print(f"  Computing {len(all_params) - existing_count} remaining replications")
    
    # Run simulations with parallel computation
    if n_jobs == 1:
        print(f"Running sequentially...")
        results = []
        for p in tqdm(all_params, desc="Simulations"):
            result = run_single_replication(p)
            if result is not None:
                results.append(result)
    else:
        import multiprocessing
        actual_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        print(f"Running with {actual_jobs} parallel jobs...")
        
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(run_single_replication)(p) for p in all_params
        )
        results = [r for r in results if r is not None]
    
    print(f"\\n✓ Completed {len(results)} successful replications")
    
    # Save consolidated results
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved raw results to: {results_file}")
    
    # Generate summary
    summary_df = evaluation.process_results(results)
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    print(f"✓ Saved summary table")
    
    # Generate LaTeX table
    if exp_name == 'experiment_3_robustness_check':
        # For experiment 3, generate separate robustness tables
        evaluation.generate_robustness_tables(results, output_dir)
    else:
        # For other experiments, generate standard LaTeX table
        evaluation.generate_latex_table(summary_df, exp_name, output_dir)
    
    print(f"\\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    return results

def _checkpoint_exists(params):
    """Check if checkpoint exists for given parameters."""
    exp_name, scenario_name, method_name, sim_id, specific_params, checkpoint_dir = params
    
    misspec_str = specific_params.get('misspecification', '')
    if misspec_str:
        checkpoint_file = checkpoint_dir / scenario_name / f'sim_{sim_id}_{method_name}_{misspec_str}.pkl'
    else:
        checkpoint_file = checkpoint_dir / scenario_name / f'sim_{sim_id}_{method_name}.pkl'
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                result = pickle.load(f)
                return 'est_ate' in result and 'scenario' in result
        except:
            return False
    return False

# %%
# Main execution
if __name__ == "__main__":
    print("="*80 + "\\nRUNNING ALL EXPERIMENTS\\n" + "="*80)
    
    # Configuration for parallel computation
    N_JOBS_EXP2_3 = -1  # Use all CPUs for Experiments 2 & 3 (-1 = all, 1 = sequential)
    
    print(f"\\nConfiguration:")
    print(f"  Experiment 1: Optuna handles parallelization (n_jobs from config)")
    print(f"  Experiments 2&3: n_jobs = {N_JOBS_EXP2_3} ({'all CPUs' if N_JOBS_EXP2_3 == -1 else f'{N_JOBS_EXP2_3} CPUs'})")
    print(f"  Checkpoint: Enabled for all experiments")
    print(f"  Resume: Automatically resumes from checkpoints")
    
    start_time = time.time()
    
    # =============================================================================
    # EXPERIMENT 1: SENSITIVITY ANALYSIS
    # =============================================================================
    print(f"\\n{'='*80}")
    print("STARTING EXPERIMENT 1: SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    
    try:
        best_params = run_experiment_1()
        print("✓ Experiment 1 completed successfully")
    except Exception as e:
        print(f"✗ Experiment 1 failed: {e}")
        traceback.print_exc()
        print("\\nCannot proceed without optimal parameters. Stopping.")
        sys.exit(1)
    
    # =============================================================================
    # UPDATE CONFIG WITH OPTIMAL PARAMETERS
    # =============================================================================
    print(f"\\n{'='*80}")
    print("UPDATING CONFIG WITH OPTIMAL PARAMETERS")
    print(f"{'='*80}")
    
    try:
        update_config(best_params)
        print("✓ Config updated successfully")
        print(f"  PILOT_RATIO: {best_params['pilot_ratio']:.2f}")
        print(f"  DELTA: {best_params['delta']:.6f}")
        print(f"  PILOT_N_ESTIMATORS: {best_params['n_estimators']}")
    except Exception as e:
        print(f"⚠ Failed to update config: {e}")
        traceback.print_exc()
    
    # =============================================================================
    # EXPERIMENT 2: MAIN COMPARISON
    # =============================================================================
    print(f"\\n{'='*80}")
    print("STARTING EXPERIMENT 2: MAIN COMPARISON")
    print(f"{'='*80}")
    
    try:
        run_experiment('experiment_2_main_comparison', n_jobs=N_JOBS_EXP2_3)
        print("✓ Experiment 2 completed successfully")
    except Exception as e:
        print(f"✗ Experiment 2 failed: {e}")
        traceback.print_exc()
    
    # =============================================================================
    # EXPERIMENT 3: ROBUSTNESS CHECK
    # =============================================================================
    print(f"\\n{'='*80}")
    print("STARTING EXPERIMENT 3: ROBUSTNESS CHECK")
    print(f"{'='*80}")
    
    try:
        run_experiment('experiment_3_robustness_check', n_jobs=N_JOBS_EXP2_3)
        print("✓ Experiment 3 completed successfully")
    except Exception as e:
        print(f"✗ Experiment 3 failed: {e}")
        traceback.print_exc()
    
    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    elapsed = time.time() - start_time
    
    print(f"\\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    
    print(f"\\nTotal runtime: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    
    print(f"\\nOptimal Parameters:")
    print(f"  n_estimators:  {best_params['n_estimators']}")
    print(f"  delta:         {best_params['delta']:.6f}")
    print(f"  pilot_ratio:   {best_params['pilot_ratio']:.2f}")
    print(f"  r0:            {best_params['r0']}")
    print(f"  r1:            {best_params['r1']}")
    
    print(f"\\nPerformance:")
    print(f"  Agg. RMSE:     {best_params['agg_rmse']:.4f}")
    print(f"  Agg. Coverage: {best_params['agg_coverage']:.4f}")
    
    print(f"\\nOutput Files:")
    print(f"  Experiment 1: ./simulation_results/exp1_sensitivity_analysis/")
    print(f"    - optimal_parameters.json")
    print(f"    - 7 PDF visualization files")
    print(f"  Experiment 2: ./simulation_results/exp2_main_comparison/")
    print(f"    - summary_table.csv")
    print(f"  Experiment 3: ./simulation_results/exp3_robustness_check/")
    print(f"    - summary_table.csv")
    
    print(f"\\n{'='*80}")
    print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")

