# -*- coding: utf-8 -*-
"""
config.py

Defines all parameters and experimental setups for the latest OS-DML 
simulation study, structured into three distinct experiments. 
All content is in English.
"""
import numpy as np
from data_generation import (
    generate_rct_s_data, generate_rct_c_data,
    generate_obs_s_data, generate_obs_c_data
)
import methods

# =============================================================================
# 1. GENERAL SIMULATION PARAMETERS
# =============================================================================
BASE_SEED = 20250919
N_SIM = 50  # Monte Carlo replications (use 1000 for a full study)
N_POPULATION = 1000000 # Population size N
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
PILOT_RATIO = 0.3          # Pilot sample ratio: r0 / (r0 + r1)
DELTA = 0.01               # Stabilization constant for probability construction
PILOT_N_ESTIMATORS = 30    # Number of estimators for pilot stage LightGBM models

# =============================================================================
# 2. EXPERIMENT DEFINITIONS
# =============================================================================

def get_experiments():
    """Defines all experiments to be run in the simulation study."""
    
    # --- Data Generating Processes (DGPs) ---
    scenarios = {
        'RCT-S': {
            "data_gen_func": generate_rct_s_data,
            "params": { "n": N_POPULATION, "p": 30 },
            "design": "rct"
        },
        'RCT-C': {
            "data_gen_func": generate_rct_c_data,
            "params": { "n": N_POPULATION, "p": 120 },
            "design": "rct"
        },
        'OBS-S': {
            "data_gen_func": generate_obs_s_data,
            "params": { "n": N_POPULATION, "p": 40 },
            "design": "obs"
        },
        'OBS-C': {
            "data_gen_func": generate_obs_c_data,
            "params": { "n": N_POPULATION, "p": 100 },
            "design": "obs"
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
            "description": "Hyperparameter sensitivity analysis using Bayesian optimization to find optimal parameters.",
            "scenarios": ['RCT-S', 'RCT-C', 'OBS-S', 'OBS-C'],
            "methods": ['OS'],
            "base_dir": "./simulation_results/exp1_sensitivity_analysis",
            "params": {
                "r_total": 10000,
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
            "scenarios": ['RCT-S', 'RCT-C', 'OBS-S', 'OBS-C'],
            "methods": ['OS', 'UNIF', 'LSS', 'FULL'],
            "base_dir": "./simulation_results/exp2_main_comparison",
            "params": {
                "r_total": 10000,  # Use r_total instead of fixed r0, r1
                "k_folds": K_FOLDS
            }
        },
        "experiment_3_robustness_check": {
            "description": "Verifies bias reduction and double robustness of OS-DML under the OBS-C DGP.",
            "scenarios": ['OBS-C'],
            "methods": ['OS'],
            "base_dir": "./simulation_results/exp3_robustness_check",
            "params": {
                "r_total": 10000,  # Use r_total instead of fixed r0, r1
                "k_folds": K_FOLDS,
                "misspecification_scenarios": ['correct_correct', 'correct_wrong', 'wrong_correct', 'wrong_wrong']
            }
        }
    }
    
    return scenarios, all_methods, experiments

