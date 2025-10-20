# -*- coding: utf-8 -*-
"""
config.py

Defines all parameters and experimental setups for the latest OS-DML 
simulation study, structured into three distinct experiments. 
All content is in English.
"""
import numpy as np
from data_generation import (
    generate_rct_1_data, generate_rct_2_data, generate_rct_3_data,
    generate_obs_1_data, generate_obs_2_data, generate_obs_3_data
)
import methods

# =============================================================================
# 1. GENERAL SIMULATION PARAMETERS
# =============================================================================
BASE_SEED = 20250919
N_SIM = 50  # Monte Carlo replications (use 1000 for a full study)
N_POPULATION = 100000 # Population size N
K_FOLDS = 2 # Number of folds for cross-fitting

# =============================================================================
# 1.1 ESTIMATOR CONFIGURATION
# =============================================================================
# Choose estimator for point estimation and variance estimation
# Options: 'hajek' (default), 'hh' (Hansen-Hurwitz)
ESTIMATOR_TYPE = 'hajek'  # Default to Hájek estimator

# =============================================================================
# 1.2 OS-DML ALGORITHM PARAMETERS
# =============================================================================
# Global parameters for OS-DML algorithm
PILOT_RATIO = 0.80          # Pilot sample ratio: r0 / (r0 + r1)
DELTA = 0.005466               # Stabilization constant for probability construction
PILOT_N_ESTIMATORS = 50    # Number of estimators for pilot stage LightGBM models

# =============================================================================
# 2. EXPERIMENT DEFINITIONS
# =============================================================================

def get_experiments():
    """Defines all experiments to be run in the simulation study."""
    
    # --- Data Generating Processes (DGPs) ---
    # 6 scenarios: RCT-1, RCT-2, RCT-3, OBS-1, OBS-2, OBS-3
    scenarios = {
        'RCT-1': {
            "data_gen_func": generate_rct_1_data,
            "params": { "n": N_POPULATION, "p": 20 },
            "design": "rct",
            "heterogeneity": "low"
        },
        'RCT-2': {
            "data_gen_func": generate_rct_2_data,
            "params": { "n": N_POPULATION, "p": 50 },
            "design": "rct",
            "heterogeneity": "moderate"
        },
        'RCT-3': {
            "data_gen_func": generate_rct_3_data,
            "params": { "n": N_POPULATION, "p": 100 },
            "design": "rct",
            "heterogeneity": "high"
        },
        'OBS-1': {
            "data_gen_func": generate_obs_1_data,
            "params": { "n": N_POPULATION, "p": 20 },
            "design": "obs",
            "heterogeneity": "low"
        },
        'OBS-2': {
            "data_gen_func": generate_obs_2_data,
            "params": { "n": N_POPULATION, "p": 50 },
            "design": "obs",
            "heterogeneity": "moderate"
        },
        'OBS-3': {
            "data_gen_func": generate_obs_3_data,
            "params": { "n": N_POPULATION, "p": 100 },
            "design": "obs",
            "heterogeneity": "high"
        }
    }

    # --- Method Definitions ---
    all_methods = {
        "OS": { "func": methods.run_os },
        "UNIF": { "func": methods.run_unif },
        "LSS": { "func": methods.run_lss },
        "FULL": { "func": methods.run_full, "params": { "k_folds": K_FOLDS } }
    }
    
    # --- Experiment Construction ---
    experiments = {
        "experiment_1_sensitivity_analysis": {
            "description": "Hyperparameter sensitivity analysis using Bayesian optimization to find optimal parameters across all DGPs.",
            "scenarios": ['RCT-1', 'RCT-2', 'RCT-3', 'OBS-1', 'OBS-2', 'OBS-3'],  # All 6 DGPs
            "methods": ['OS'],
            "base_dir": "./simulation_results/exp1_sensitivity_analysis",
            "params": {
                "r_total": 1000,
                "n_trials": 60,
                "n_replications": 10,
                "aggregation": "mean",
                "n_jobs": -1,
                "checkpoint": True,
                "resume": True
            }
        },
        "experiment_2_main_comparison": {
            "description": "Comprehensive performance comparison of all methods across all DGPs.",
            "scenarios": ['RCT-1', 'RCT-2', 'RCT-3', 'OBS-1', 'OBS-2', 'OBS-3'],  # All 6 DGPs
            "methods": ['OS', 'UNIF', 'LSS', 'FULL'],
            "base_dir": "./simulation_results/exp2_main_comparison",
            "params": {
                "r_total": 1000,  # Use r_total instead of fixed r0, r1
                "k_folds": K_FOLDS
            }
        },
        "experiment_3_robustness_check": {
            "description": "Verifies bias reduction and double robustness of OS-DML under moderate heterogeneity DGPs with four nuisance model specifications.",
            "scenarios": ['RCT-2', 'OBS-2'],  # Only moderate heterogeneity scenarios
            "methods": ['OS'],
            "base_dir": "./simulation_results/exp3_robustness_check",
            "params": {
                "r_total": 1000,  # Use r_total instead of fixed r0, r1
                "k_folds": K_FOLDS,
                "misspecification_scenarios": ['correct_correct', 'correct_wrong', 'wrong_correct', 'wrong_wrong']
            }
        }
    }
    
    return scenarios, all_methods, experiments

