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
warnings.filterwarnings('ignore')

import os, json, pickle, time, traceback
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['LIGHTGBM_VERBOSITY'] = '-1'

import config
import evaluation
from data_generation import generate_obs_s_data, generate_obs_c_data, generate_rct_s_data, generate_rct_c_data
import lightgbm as lgb
from methods import _orthogonal_score, _fit_nuisance_models, _get_hansen_hurwitz_ci

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:
    import sys
    !{sys.executable} -m pip install optuna -q
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
            'OBS-S': {'func': generate_obs_s_data, 'params': {'n': self.N, 'p': 40}, 'is_rct': False},
            'OBS-C': {'func': generate_obs_c_data, 'params': {'n': self.N, 'p': 100}, 'is_rct': False},
            'RCT-S': {'func': generate_rct_s_data, 'params': {'n': self.N, 'p': 30}, 'is_rct': True},
            'RCT-C': {'func': generate_rct_c_data, 'params': {'n': self.N, 'p': 120}, 'is_rct': True}
        }
        
        self.scenarios = all_scenarios if scenarios == 'all' else {scenarios: all_scenarios[scenarios]}
        self.scenario_name = 'all_scenarios' if scenarios == 'all' else scenarios
    
    def objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 20, 150, step=10)
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
        lgbm_params = {'n_jobs': 1, 'random_state': config.BASE_SEED, 'n_estimators': n_estimators, 'verbose': -1}
        
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
    
    with open(output_dir / 'optimal_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    study.trials_dataframe().to_csv(output_dir / 'trials.csv', index=False)
    
    print(f"\\n✓ Optimal: n_estimators={best_params['n_estimators']}, delta={best_params['delta']:.6f}, pilot_ratio={best_params['pilot_ratio']:.2f}")
    return best_params

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
    exp_name, scenario_name, method_name, sim_id, specific_params = params
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
        return result
    except:
        return None

def run_experiment(exp_name):
    print(f"\\n{'='*80}\\n{exp_name.upper()}\\n{'='*80}")
    
    scenarios, all_methods, experiments = config.get_experiments()
    exp_config = experiments[exp_name]
    output_dir = Path(exp_config['base_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_params = []
    for scenario_name in exp_config['scenarios']:
        for method_name in exp_config['methods']:
            specific_params = {'k_folds': exp_config['params']['k_folds']} if method_name == 'FULL' else {'r': {'r_total': exp_config['params']['r_total']}, 'k_folds': exp_config['params']['k_folds']}
            
            if 'misspecification_scenarios' in exp_config['params']:
                for misspec in exp_config['params']['misspecification_scenarios']:
                    for sim_id in range(config.N_SIM):
                        all_params.append((exp_name, scenario_name, method_name, sim_id, {**specific_params, 'misspecification': misspec}))
            else:
                for sim_id in range(config.N_SIM):
                    all_params.append((exp_name, scenario_name, method_name, sim_id, specific_params))
    
    results = [r for r in [run_single_replication(p) for p in tqdm(all_params, desc="Simulations")] if r]
    
    with open(output_dir / 'raw_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    summary_df = evaluation.process_results(results)
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    print(f"\\n✓ {len(results)} replications\\n{summary_df.to_string(index=False)}")
    return results

# %%
# Main execution
print("="*80 + "\\nRUNNING ALL EXPERIMENTS\\n" + "="*80)

start_time = time.time()

# Experiment 1
best_params = run_experiment_1()
update_config(best_params)

# Experiment 2
run_experiment('experiment_2_main_comparison')

# Experiment 3
run_experiment('experiment_3_robustness_check')

print(f"\\n{'='*80}\\nALL COMPLETE in {(time.time()-start_time)/60:.1f} minutes\\n{'='*80}")

