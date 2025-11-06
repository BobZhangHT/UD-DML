# -*- coding: utf-8 -*-
"""
config.py

Central configuration for the UD-DML simulation suite.
Defines data-generating processes (DGPs), estimator settings, and experiment grids.
All content is in English.
"""
import numpy as np

from data_generation import (
    generate_rct_1_data, generate_rct_2_data, generate_rct_3_data,
    generate_obs_1_data, generate_obs_2_data, generate_obs_3_data,
)
import methods
MAX_PARALLEL_JOBS = 16  # Max cores for outer parallelism

# =============================================================================
# 1. GENERAL SIMULATION PARAMETERS
# =============================================================================
BASE_SEED = 20250919
DEFAULT_REPLICATIONS = 500
N_POPULATION = 500_000
K_FOLDS = 2

# =============================================================================
# 1.1 UD-DML SUBSAMPLING CONFIGURATION
# =============================================================================

# Uniform Design sampling parameters
UD_NEAREST_NEIGHBORS = 8  # Number of neighbors to probe when deduplicating matches
UD_DEDUP_STRATEGY = 'random_remaining'  # How to fill slots when nearest neighbors repeat
UD_CDF_CLIP = 1e-6  # Clip bounds for PIT transformed values
UD_ENSURE_UNIQUE = True  # Enforce unique subsample indices
# Future extensions can hook into these parameters (e.g., MMD kernels)

# LightGBM defaults
LGBM_N_ESTIMATORS = 100  # Number of boosting iterations
LGBM_MAX_DEPTH = 5  # Maximum depth of trees
LGBM_LEARNING_RATE = 0.1  # Learning rate for gradient boosting
LGBM_NUM_LEAVES = 31  # Number of leaves in each tree
DEFAULT_NUISANCE_LEARNER = 'lgbm'
RF_N_TREES = 100
RF_N_JOBS = 1
LASSO_CV_FOLDS = 5
LASSO_CV_MAX_ITER = 5000
LOGIT_CV_MAX_ITER = 5000
LOGIT_CV_SCORING = 'neg_log_loss'
LOGIT_CV_CS = np.logspace(-2, 2, 5)

# =============================================================================
# 1.2 GLOBAL GRIDS
# =============================================================================
SUBSAMPLE_TOTALS = [1000, 2_500, 5_000, 7_500, 10_000]
POPULATION_SIZE_GRID = [100_000, 500_000]
ROBUSTNESS_MISSPECIFICATIONS = [
    'correct_correct',
    'correct_wrong',
    'wrong_correct',
    'wrong_wrong',
]

# =============================================================================
# 2. EXPERIMENT DEFINITIONS
# =============================================================================

def get_experiments():
    """Return scenario definitions, method catalog, and redesigned experiments."""

    scenarios = {
        'RCT-1': {
            'data_gen_func': generate_rct_1_data,
            'params': {'n': N_POPULATION, 'p': 10},
            'design': 'rct',
            'covariates': 'x1',
        },
        'RCT-2': {
            'data_gen_func': generate_rct_2_data,
            'params': {'n': N_POPULATION, 'p': 10},
            'design': 'rct',
            'covariates': 'x2',
        },
        'RCT-3': {
            'data_gen_func': generate_rct_3_data,
            'params': {'n': N_POPULATION, 'p': 10},
            'design': 'rct',
            'covariates': 'x3',
        },
        'OBS-1': {
            'data_gen_func': generate_obs_1_data,
            'params': {'n': N_POPULATION, 'p': 10},
            'design': 'obs',
            'covariates': 'x1',
        },
        'OBS-2': {
            'data_gen_func': generate_obs_2_data,
            'params': {'n': N_POPULATION, 'p': 10},
            'design': 'obs',
            'covariates': 'x2',
        },
        'OBS-3': {
            'data_gen_func': generate_obs_3_data,
            'params': {'n': N_POPULATION, 'p': 10},
            'design': 'obs',
            'covariates': 'x3',
        },
    }

    all_methods = {
        'UD': {'func': methods.run_ud},
        'UNIF': {'func': methods.run_unif},
        'FULL': {'func': methods.run_full, 'params': {'k_folds': K_FOLDS}},
    }

    experiments = {
        'experiment_visualization': {
            'description': '2D covariate coverage diagnostics for UD-DML versus UNIF.',
            'scenarios': list(scenarios.keys()),
            'methods': ['UD', 'UNIF'],
            'base_dir': './simulation_results/visualization',
            'params': {
                'r_total': 500,
                'population_size': 100_000,
                'store_sample': True,
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': 1,
            },
        },
        'experiment_subsample_size': {
            'description': 'RMSE/coverage across subsample sizes for all six DGPs.',
            'scenarios': list(scenarios.keys()),
            'methods': ['UD', 'UNIF'],
            'base_dir': './simulation_results/subsample_size_grid',
            'params': {
                'r_totals': SUBSAMPLE_TOTALS,
                'population_size': N_POPULATION,
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': DEFAULT_REPLICATIONS,
            },
        },
        'experiment_population_size': {
            'description': (
                'Effect of full-data size on estimation accuracy for low/high UD budgets.'
            ),
            'scenarios': list(scenarios.keys()),
            'methods': ['UD', 'UNIF', 'FULL'],
            'base_dir': './simulation_results/population_scaling',
            'params': {
                'population_sizes': POPULATION_SIZE_GRID,
                'r_totals': [1_000, 5_000],
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': DEFAULT_REPLICATIONS,
            },
        },
        'experiment_double_robust': {
            'description': 'Double-robustness stress test for observational scenarios.',
            'scenarios': ['OBS-1', 'OBS-2', 'OBS-3'],
            'methods': ['UD', 'UNIF'],
            'base_dir': './simulation_results/double_robustness',
            'params': {
                'misspecification_scenarios': ROBUSTNESS_MISSPECIFICATIONS,
                'r_total': 5_000,
                'population_size': N_POPULATION,
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': DEFAULT_REPLICATIONS,
            },
        },
        'experiment_nuisance_sensitivity': {
            'description': (
                'Comparison of nuisance learners (LGBM, RF, LASSO) on OBS-3 across budgets.'
            ),
            'scenarios': ['OBS-3'],
            'methods': ['UD', 'UNIF'],
            'base_dir': './simulation_results/nuisance_sensitivity',
            'params': {
                'r_totals': SUBSAMPLE_TOTALS,
                'population_size': N_POPULATION,
                'nuisance_learners': ['lasso_cv', 'rf', 'lgbm'],
                'n_estimators': LGBM_N_ESTIMATORS,
                'k_folds': K_FOLDS,
                'n_replications': DEFAULT_REPLICATIONS,
            },
        },
    }

    return scenarios, all_methods, experiments
