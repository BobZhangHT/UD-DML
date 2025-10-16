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
N_POPULATION = 100000 # Population size N
K_FOLDS = 2 # Number of folds for cross-fitting

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
        "experiment_1_pilot_allocation": {
            "description": "Investigates the effect of pilot sample allocation on OS-DML performance.",
            "scenarios": ['RCT-S', 'OBS-S'],
            "methods": ['OS'],
            "base_dir": "./simulation_results/exp1_pilot_allocation",
            "params": {
                "r_total": 10000,
                "pilot_ratios": np.arange(0.1, 1.0, 0.1) # Pilot ratios from 0.1 to 0.9
            }
        },
        "experiment_2_main_comparison": {
            "description": "Comprehensive performance comparison of all methods across all DGPs.",
            "scenarios": ['RCT-S', 'RCT-C', 'OBS-S', 'OBS-C'],
            "methods": ['OS', 'UNIF', 'LSS', 'FULL'],
            "base_dir": "./simulation_results/exp2_main_comparison",
            "params": {
                "r0": 3000, "r1": 7000, "k_folds": K_FOLDS
            }
        },
        "experiment_3_robustness_check": {
            "description": "Verifies bias reduction and double robustness of OS-DML under the OBS-C DGP.",
            "scenarios": ['OBS-C'],
            "methods": ['OS'],
            "base_dir": "./simulation_results/exp3_robustness_check",
            "params": {
                "r0": 3000, "r1": 7000, "k_folds": K_FOLDS,
                "misspecification_scenarios": ['correct_correct', 'correct_wrong', 'wrong_correct', 'wrong_wrong']
            }
        }
    }
    
    return scenarios, all_methods, experiments

