# -*- coding: utf-8 -*-
"""
config.py

Central configuration for the redesigned OS-DML simulation suite.
Defines data-generating processes (DGPs), estimator settings, and experiment grids.
All content is in English.
"""
import numpy as np

from data_generation import (
    generate_rct_1_data, generate_rct_2_data, generate_rct_3_data,
    generate_obs_1_data, generate_obs_2_data, generate_obs_3_data,
)
import methods

# =============================================================================
# 1. GENERAL SIMULATION PARAMETERS
# =============================================================================
BASE_SEED = 20250919
DEFAULT_REPLICATIONS = 200
N_POPULATION = 100_000
K_FOLDS = 2

# =============================================================================
# 1.1 ESTIMATOR CONFIGURATION
# =============================================================================
ESTIMATOR_TYPE = 'hajek'
DELTA = 0.001  # Stabilisation constant for sampling probabilities
DEFAULT_PILOT_RATIO = 0.20
LGBM_N_ESTIMATORS = 100  # Shared across pilot and full nuisance models

# =============================================================================
# 1.2 GLOBAL GRIDS
# =============================================================================
PILOT_RATIO_GRID = [round(x, 2) for x in np.linspace(0.10, 0.90, 9)]
SUBSAMPLE_TOTALS = [250, 500, 1000, 2000, 4000]
POPULATION_SIZE_GRID = [25_000, 50_000, 100_000, 200_000, 400_000]
ROBUSTNESS_MISSPECIFICATIONS = [
    'correct_correct',
    'correct_wrong',
    'wrong_correct',
    'wrong_wrong',
]
LARGE_POPULATION_THRESHOLD = 200_000
LARGE_POPULATION_REPLICATIONS = 150

# =============================================================================
# 2. EXPERIMENT DEFINITIONS
# =============================================================================

def get_experiments():
    """Return scenario definitions, method catalog, and redesigned experiments."""

    scenarios = {
        'RCT-1': {
            'data_gen_func': generate_rct_1_data,
            'params': {'n': N_POPULATION, 'p': 20},
            'design': 'rct',
            'heterogeneity': 'low',
        },
        'RCT-2': {
            'data_gen_func': generate_rct_2_data,
            'params': {'n': N_POPULATION, 'p': 50},
            'design': 'rct',
            'heterogeneity': 'moderate',
        },
        'RCT-3': {
            'data_gen_func': generate_rct_3_data,
            'params': {'n': N_POPULATION, 'p': 100},
            'design': 'rct',
            'heterogeneity': 'high',
        },
        'OBS-1': {
            'data_gen_func': generate_obs_1_data,
            'params': {'n': N_POPULATION, 'p': 20},
            'design': 'obs',
            'heterogeneity': 'low',
        },
        'OBS-2': {
            'data_gen_func': generate_obs_2_data,
            'params': {'n': N_POPULATION, 'p': 50},
            'design': 'obs',
            'heterogeneity': 'moderate',
        },
        'OBS-3': {
            'data_gen_func': generate_obs_3_data,
            'params': {'n': N_POPULATION, 'p': 100},
            'design': 'obs',
            'heterogeneity': 'high',
        },
    }

    all_methods = {
        'OS': {'func': methods.run_os},
        'UNIF': {'func': methods.run_unif},
        'LSS': {'func': methods.run_lss},
        'FULL': {'func': methods.run_full, 'params': {'k_folds': K_FOLDS}},
    }

    experiments = {
        'experiment_pilot_ratio_sweep': {
            'description': (
                'Pilot subsample ratio sweep for OS-DML across all six DGPs; '
                'delta fixed small, pilot/final LightGBM share n_estimators.'
            ),
            'scenarios': list(scenarios.keys()),
            'methods': ['OS'],
            'base_dir': './simulation_results/pilot_ratio_sweep',
            'params': {
                'pilot_ratios': PILOT_RATIO_GRID,
                'r_total': 1000,
                'delta': DELTA,
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': DEFAULT_REPLICATIONS,
                'population_size': N_POPULATION,
                'coverage_threshold': 0.93,
            },
        },
        'experiment_subsample_budget': {
            'description': (
                'Total subsample budget comparison for OS-DML and benchmarks '
                'under empirical pilot ratio selection.'
            ),
            'scenarios': list(scenarios.keys()),
            'methods': ['OS', 'UNIF', 'LSS', 'FULL'],
            'base_dir': './simulation_results/subsample_budget',
            'params': {
                'r_totals': SUBSAMPLE_TOTALS,
                'pilot_ratio_strategy': 'empirical',
                'delta': DELTA,
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': DEFAULT_REPLICATIONS,
                'population_size': N_POPULATION,
            },
        },
        'experiment_population_scaling': {
            'description': (
                'Full population size scaling study with fixed subsample budget '
                'and empirical pilot ratio.'
            ),
            'scenarios': list(scenarios.keys()),
            'methods': ['OS', 'UNIF', 'LSS', 'FULL'],
            'base_dir': './simulation_results/population_scaling',
            'params': {
                'population_sizes': POPULATION_SIZE_GRID,
                'r_total': 1000,
                'pilot_ratio_strategy': 'empirical',
                'delta': DELTA,
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': DEFAULT_REPLICATIONS,
                'large_population_threshold': LARGE_POPULATION_THRESHOLD,
                'n_replications_large': LARGE_POPULATION_REPLICATIONS,
            },
        },
        'experiment_double_robustness': {
            'description': (
                'Double-robustness stress test with four nuisance specifications '
                'extended to all DGPs.'
            ),
            'scenarios': list(scenarios.keys()),
            'methods': ['OS'],
            'base_dir': './simulation_results/double_robustness',
            'params': {
                'misspecification_scenarios': ROBUSTNESS_MISSPECIFICATIONS,
                'r_total': 1000,
                'pilot_ratio_strategy': 'empirical',
                'delta': DELTA,
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': DEFAULT_REPLICATIONS,
            },
        },
    }

    return scenarios, all_methods, experiments
