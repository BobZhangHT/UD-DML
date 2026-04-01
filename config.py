# -*- coding: utf-8 -*-
"""
config.py — Central configuration for the UD-DML simulation suite.

Defines:
    * Global simulation parameters (seed, replications, population size).
    * UD-DML subsampling hyper-parameters (PCA variance threshold ρ₀,
      GLP generator search budget, nearest-neighbour query size).
    * Nuisance learner defaults (LightGBM, Random Forest, LASSO-CV).
    * Experiment-level grids and scenario definitions.

All parameters referenced by ``methods.py``, ``data_generation.py``,
``simulations.py``, and ``evaluation.py`` are consolidated here so that
a single edit propagates globally.
"""

from __future__ import annotations

import numpy as np

from data_generation import (
    generate_obs_1_data,
    generate_obs_2_data,
    generate_obs_3_data,
    generate_rct_1_data,
    generate_rct_2_data,
    generate_rct_3_data,
)
import methods

# ═══════════════════════════════════════════════════════════════════════════
# 1. Global Simulation Parameters
# ═══════════════════════════════════════════════════════════════════════════

BASE_SEED: int = 20250919
"""Master random seed for reproducibility across all experiments."""

DEFAULT_REPLICATIONS: int = 1000
"""Number of Monte Carlo replications per experiment variant."""

N_POPULATION: int = 500_000
"""Default full-data population size *n*."""

K_FOLDS: int = 2
"""Number of cross-fitting folds for DML (Section 3.2)."""

MAX_PARALLEL_JOBS: int = 16
"""Upper bound on the number of cores used for outer parallelism."""

# ═══════════════════════════════════════════════════════════════════════════
# 2. UD-DML Subsampling Configuration (Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════════

UD_VARIANCE_THRESHOLD: float = 0.85
"""Cumulative variance threshold ρ₀ for PCA dimension retention (Step 3).

Following Zhang et al. (2023) and Zhou et al. (2024), the default is 0.85.
When the covariates are already uncorrelated and equi-variant, q = p.
"""

UD_MAX_GENERATOR_CANDIDATES: int = 200
"""Maximum number of admissible power generators to evaluate (Step 6).

If the total number of admissible α exceeds this value, a random subset
is drawn and searched (Section 2.2, paragraph following the definition
of α̂).
"""

UD_NEAREST_NEIGHBORS: int = 5
"""Initial k for the adaptive nearest-neighbour query (Steps 16–17).

The query expands adaptively (k ← min(n_arm, 2k)) until an unused unit
is found.  A small initial k keeps the typical cost close to O(log n).
"""

# ═══════════════════════════════════════════════════════════════════════════
# 3. Nuisance Learner Defaults
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_NUISANCE_LEARNER: str = "lgbm"
"""Default ML learner for nuisance estimation (Section 3.2)."""

# -- LightGBM --
LGBM_N_ESTIMATORS: int = 100
"""Number of boosting rounds (Section 3.2: '100 boosting rounds')."""

LGBM_MAX_DEPTH: int = 5
"""Maximum tree depth."""

LGBM_LEARNING_RATE: float = 0.1
"""Gradient boosting learning rate."""

LGBM_NUM_LEAVES: int = 31
"""Maximum number of leaves per tree."""

# -- Random Forest --
RF_N_TREES: int = 100
"""Number of trees (Section 3.3, Experiment 4: '100 trees')."""

RF_N_JOBS: int = 1
"""Number of parallel jobs for RF fitting."""

# -- LASSO / Logistic CV --
LASSO_CV_FOLDS: int = 5
"""Inner CV folds for LASSO-CV (Section 3.3, Experiment 4)."""

LASSO_CV_MAX_ITER: int = 5000
"""Maximum iterations for LASSO-CV solver."""

LOGIT_CV_MAX_ITER: int = 5000
"""Maximum iterations for logistic regression CV solver."""

LOGIT_CV_SCORING: str = "neg_log_loss"
"""Scoring metric for LogisticRegressionCV."""

LOGIT_CV_CS: np.ndarray = np.logspace(-2, 2, 5)
"""Regularisation grid for LogisticRegressionCV."""

# ═══════════════════════════════════════════════════════════════════════════
# 4. Experiment Grids
# ═══════════════════════════════════════════════════════════════════════════

SUBSAMPLE_TOTALS: list[int] = [1_000, 2_500, 5_000, 7_500, 10_000]
"""Subsample budget grid r ∈ {1000, 2500, 5000, 7500, 10000}."""

POPULATION_SIZE_GRID: list[int] = [100_000, 500_000]
"""Full-data sizes for the scalability experiment (Section 3.3, Exp 2)."""

ROBUSTNESS_MISSPECIFICATIONS: list[str] = [
    "correct_correct",
    "correct_wrong",
    "wrong_correct",
    "wrong_wrong",
]
"""Nuisance model specification scenarios (Section 3.3, Experiment 3)."""


# ═══════════════════════════════════════════════════════════════════════════
# 5. Scenario and Experiment Definitions
# ═══════════════════════════════════════════════════════════════════════════


def get_experiments():
    """Return scenario definitions, method catalogue, and experiments.

    Returns
    -------
    scenarios : dict
        Keys are scenario names (e.g. ``'RCT-1'``); values contain
        ``data_gen_func``, ``params``, ``design``, and ``covariates``.
    all_methods : dict
        Keys are method names (``'UD'``, ``'UNIF'``, ``'FULL'``); values
        contain ``func`` and optional default ``params``.
    experiments : dict
        Keys are experiment identifiers; values describe the scenarios,
        methods, output directory, and parameter grid for each experiment.
    """

    scenarios = {
        "RCT-1": {
            "data_gen_func": generate_rct_1_data,
            "params": {"n": N_POPULATION, "p": 10},
            "design": "rct",
            "covariates": "x1",
        },
        "RCT-2": {
            "data_gen_func": generate_rct_2_data,
            "params": {"n": N_POPULATION, "p": 10},
            "design": "rct",
            "covariates": "x2",
        },
        "RCT-3": {
            "data_gen_func": generate_rct_3_data,
            "params": {"n": N_POPULATION, "p": 10},
            "design": "rct",
            "covariates": "x3",
        },
        "OBS-1": {
            "data_gen_func": generate_obs_1_data,
            "params": {"n": N_POPULATION, "p": 10},
            "design": "obs",
            "covariates": "x1",
        },
        "OBS-2": {
            "data_gen_func": generate_obs_2_data,
            "params": {"n": N_POPULATION, "p": 10},
            "design": "obs",
            "covariates": "x2",
        },
        "OBS-3": {
            "data_gen_func": generate_obs_3_data,
            "params": {"n": N_POPULATION, "p": 10},
            "design": "obs",
            "covariates": "x3",
        },
    }

    all_methods = {
        "UD": {"func": methods.run_ud},
        "UNIF": {"func": methods.run_unif},
        "FULL": {"func": methods.run_full, "params": {"k_folds": K_FOLDS}},
    }

    experiments = {
        "experiment_visualization": {
            "description": "2-D covariate coverage diagnostics for UD-DML vs UNIF.",
            "scenarios": list(scenarios.keys()),
            "methods": ["UD", "UNIF"],
            "base_dir": "./simulation_results/visualization",
            "params": {
                "r_total": 500,
                "population_size": 100_000,
                "store_sample": True,
                "n_estimators": LGBM_N_ESTIMATORS,
                "k_folds": K_FOLDS,
                "n_replications": 1,
            },
        },
        "experiment_subsample_size": {
            "description": (
                "RMSE / coverage across subsample sizes for all six DGPs "
                "(Section 3.3, Experiment 1)."
            ),
            "scenarios": list(scenarios.keys()),
            "methods": ["UD", "UNIF"],
            "base_dir": "./simulation_results/subsample_size_grid",
            "params": {
                "r_totals": SUBSAMPLE_TOTALS,
                "population_size": N_POPULATION,
                "n_estimators": LGBM_N_ESTIMATORS,
                "k_folds": K_FOLDS,
                "n_replications": DEFAULT_REPLICATIONS,
            },
        },
        "experiment_population_size": {
            "description": (
                "Scalability: effect of full-data size n on estimation accuracy "
                "for low / high UD budgets (Section 3.3, Experiment 2)."
            ),
            "scenarios": list(scenarios.keys()),
            "methods": ["UD", "UNIF", "FULL"],
            "base_dir": "./simulation_results/population_scaling",
            "params": {
                "population_sizes": POPULATION_SIZE_GRID,
                "r_totals": [1_000, 5_000],
                "n_estimators": LGBM_N_ESTIMATORS,
                "k_folds": K_FOLDS,
                "n_replications": DEFAULT_REPLICATIONS,
            },
        },
        "experiment_double_robust": {
            "description": (
                "Double-robustness stress test for observational scenarios "
                "(Section 3.3, Experiment 3)."
            ),
            "scenarios": ["OBS-1", "OBS-2", "OBS-3"],
            "methods": ["UD", "UNIF"],
            "base_dir": "./simulation_results/double_robustness",
            "params": {
                "misspecification_scenarios": ROBUSTNESS_MISSPECIFICATIONS,
                "r_total": 5_000,
                "population_size": N_POPULATION,
                "n_estimators": LGBM_N_ESTIMATORS,
                "k_folds": K_FOLDS,
                "n_replications": DEFAULT_REPLICATIONS,
            },
        },
        "experiment_nuisance_sensitivity": {
            "description": (
                "Learner sensitivity: LGBM vs RF vs LASSO on OBS-3 "
                "(Section 3.3, Experiment 4)."
            ),
            "scenarios": ["OBS-3"],
            "methods": ["UD", "UNIF"],
            "base_dir": "./simulation_results/nuisance_sensitivity",
            "params": {
                "r_totals": SUBSAMPLE_TOTALS,
                "population_size": N_POPULATION,
                "nuisance_learners": ["lasso_cv", "rf", "lgbm"],
                "n_estimators": LGBM_N_ESTIMATORS,
                "k_folds": K_FOLDS,
                "n_replications": DEFAULT_REPLICATIONS,
            },
        },
    }

    return scenarios, all_methods, experiments
