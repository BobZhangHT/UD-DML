# config.py
#
# This script contains all experimental configurations for the simulation study.
# It defines the parameters for data generation, methods, and evaluation.
# -----------------------------------------------------------------------------

import numpy as np
from data_generation import (generate_linear_data, 
                           generate_logistic_data, 
                           generate_cox_data)
from methods import (run_abbs, run_os_repeated, run_blbb, run_bmh)

# --- General Simulation Parameters ---
BASE_SEED = 20250907  # For reproducibility
N_SIM = 100         # Number of simulation runs for each scenario
N_FULL = int(1e5)   # Full dataset size
P_DIM = 50          # Number of covariates
R_SUB = 1000        # Expected subsample size
T_POS = 2000        # Number of posterior samples / repetitions
BURN_IN = 1000      # Burn-in for MCMC methods

# --- Directory Setup ---
SIM_RESULTS_DIR = "./simulation_results"
ANALYSIS_RESULTS_DIR = "./analysis_results"

# --- Model Configurations ---

# True beta coefficients (sparse, with p=50)
true_beta = np.zeros(P_DIM)
true_beta[:10] = np.arange(0.5, 1.5, 0.1)
true_beta[0] = 2.0  # Make the first parameter more prominent for CI analysis

# --- Scenario Definitions ---
# A list of scenarios to run.
scenarios = {
    "linear_normal": {
        "model": "linear",
        "data_gen_func": generate_linear_data,
        "params": {
            "N": N_FULL,
            "p": P_DIM,
            "beta": true_beta,
            "error_dist": "t",
            "df": 3,
            "sigma": 1.0,
            "true_beta": true_beta
        }
    },
    "logistic_balanced": {
        "model": "logistic",
        "data_gen_func": generate_logistic_data,
        "params": {
            "N": N_FULL,
            "p": P_DIM,
            "beta": true_beta,
            "imbalance_offset": 0.0,
            "true_beta": true_beta
        }
    }
    # "logistic_imbalanced": {
    #     "model": "logistic",
    #     "data_gen_func": generate_logistic_data,
    #     "params": {
    #         "N": N_FULL,
    #         "p": P_DIM,
    #         "beta": true_beta,
    #         "imbalance_offset": -2.0,
    #         "true_beta": true_beta
    #     }
    # },
    # "cox_ph_low_censoring": {
    #     "model": "coxph",
    #     "data_gen_func": generate_cox_data,
    #     "params": {
    #         "N": N_FULL,
    #         "p": P_DIM,
    #         "beta": true_beta / 4,
    #         "lambda0": 0.01,
    #         "tau_max": 50,
    #         "true_beta": true_beta / 4
    #     }
    # },
    # "cox_ph_high_censoring": {
    #     "model": "coxph",
    #     "data_gen_func": generate_cox_data,
    #     "params": {
    #         "N": N_FULL,
    #         "p": P_DIM,
    #         "beta": true_beta / 4,
    #         "lambda0": 0.01,
    #         "tau_max": 10,
    #         "true_beta": true_beta / 4
    #     }
    # }
}

# --- Method Definitions ---
# A dictionary of methods to run for each scenario.
methods_to_run = {
    "ABBS": {
        "func": run_abbs,
        "params": {
            "r": R_SUB,
            "T": T_POS + BURN_IN,
            "burn_in": BURN_IN,
            "c_init": 1.0, "a0": 1.0, "b0": 1.0
        }
    },
    "OS": {
        "func": run_os_repeated,
        "params": {
            "r": R_SUB,
            "reps": T_POS
        },
        "models": ["linear", "logistic"]
    },
    "BLBB": {
        "func": run_blbb,
        "params": {
            "s": 50,
            "b": N_FULL // 50,
            "T_inner": T_POS // 50
        }
    },
    "BMH": {
        "func": run_bmh,
        "params": {
            "m": R_SUB,
            "k": 10,
            "T": T_POS + BURN_IN,
            "burn_in": BURN_IN
        }
    }
}
