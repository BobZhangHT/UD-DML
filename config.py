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

from pathlib import Path

import numpy as np

from data_generation import (
    generate_obs_1_data,
    generate_obs_2_data,
    generate_obs_3_data,
    generate_obs_3_overlap_data,
)
import methods

# ═══════════════════════════════════════════════════════════════════════════
# 1. Global Simulation Parameters
# ═══════════════════════════════════════════════════════════════════════════

BASE_SEED: int = 20250919
"""Master random seed for reproducibility across all experiments."""

DEFAULT_REPLICATIONS: int = 500
"""Number of Monte Carlo replications per experiment variant."""

N_POPULATION: int = 500_000
"""Default full-data population size *n*."""

K_FOLDS: int = 2
"""Number of cross-fitting folds for DML (Section 3.2)."""

MAX_PARALLEL_JOBS: int = 32
"""Upper bound on the number of cores used for outer parallelism.

Tuned for the 32 vCPU AMD EPYC 9654 / 60 GB RAM server.  Override at
runtime via env ``OS_DML_MAX_JOBS`` (e.g. when running on a laptop).
Each worker uses ~1-2 GB for n=5e5 replications, so 32 workers fit
comfortably into 60 GB with LGBM/DML overhead.
"""

# ═══════════════════════════════════════════════════════════════════════════
# 2. UD-DML Subsampling Configuration (Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════════

UD_VARIANCE_THRESHOLD: float = 0.85
"""Cumulative variance threshold ρ₀ for PCA dimension retention (Step 3).

Following Zhang et al. (2023) and Zhou et al. (2024), the default is 0.85.
When the covariates are already uncorrelated and equi-variant, q = p.
"""

UD_MAX_GENERATOR_CANDIDATES: int = 30
"""Budget B_γ: maximum number of admissible power generators to evaluate (Algorithm 1).

If the total number of admissible α exceeds B_γ, a random subset of size
B_γ is drawn and searched (quasi-optimal / budgeted search, Section 2.2).
"""

UD_NEAREST_NEIGHBORS: int = 5
"""Legacy parameter kept for config compatibility.

The main UD-DML path uses exact 1-NN matching *with* replacement in Z-space
(Algorithm 1); this key is not used by ``methods.run_ud``.
"""

UD_SKELETON_DISK_CACHE_DIR: Path | None = Path("ud_skeleton_cache")
"""Persist GLP uniform-design skeletons ``U*`` for reuse across processes/runs.

Files are keyed by ``(r_p, q, B_gamma, cache_seed)`` and are bitwise-identical
to the in-memory cache in ``methods``. Set to ``None`` to disable unless the
environment variable ``UD_SKELETON_DISK_CACHE`` overrides (see ``methods``).

The first run populates this directory; later runs and parallel workers load
from disk when the in-process cache misses (typical under ``joblib``).
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

OVERLAP_STRENGTH_GRID: list[float] = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
"""Propensity coefficient multiplier c for the overlap gradient experiment.

logit(e(X)) = c · (0.3·X₁ + 0.3·X₂ − 0.5·X₆).  Higher c → worse overlap.
c=0.5 reproduces the default OBS-3; c=1.5 creates near-zero overlap.
"""


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

    # Only observational scenarios — RCTs removed per paper scope.
    scenarios = {
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
        "OBS-3-overlap": {
            "data_gen_func": generate_obs_3_overlap_data,
            "params": {"n": N_POPULATION, "p": 10, "overlap_strength": 1.0},
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
        # ── Exp 1: Statistical efficiency vs subsample budget r ──
        "experiment_subsample_size": {
            "description": (
                "RMSE / CI coverage / CI width across subsample sizes r "
                "for all three OBS scenarios (Paper Figure 1)."
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
        # ── Exp 2: Overlap gradient — how advantage scales with overlap severity ──
        "experiment_overlap_gradient": {
            "description": (
                "Overlap gradient: RMSE and CI metrics as a function of "
                "propensity coefficient strength c (Paper Figure 2 + Table 1)."
            ),
            "scenarios": ["OBS-3-overlap"],
            "methods": ["UD", "UNIF"],
            "base_dir": "./simulation_results/overlap_gradient",
            "params": {
                "overlap_strengths": OVERLAP_STRENGTH_GRID,
                "r_total": 5_000,
                "population_size": N_POPULATION,
                "n_estimators": LGBM_N_ESTIMATORS,
                "k_folds": K_FOLDS,
                "n_replications": DEFAULT_REPLICATIONS,
            },
        },
        # ── Exp 3: Double-robustness stress test ──
        "experiment_double_robust": {
            "description": (
                "Double-robustness stress test under nuisance misspecification "
                "(Paper Table 2)."
            ),
            "scenarios": list(scenarios.keys()),
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
        # ── Exp 4: Scalability with population n + bias-variance decomposition ──
        "experiment_population_size": {
            "description": (
                "Scalability: effect of n on estimation accuracy for low/high "
                "subsample budgets, with bias-variance decomposition (Paper Table 3)."
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
        # ── Exp 5: Asymptotic normality + covariate balance (post-process) ──
        # Uses raw results from Exp 1 (r=5000, OBS-3). No extra simulation needed.
        # Generates: Q-Q normality plot (Figure 3), SMD love plot (Figure 4).
        "experiment_visualization": {
            "description": (
                "Propensity density, Q-Q normality, and SMD love plot "
                "diagnostics (Paper Figures 3-4). Single rep for propensity viz."
            ),
            "scenarios": list(scenarios.keys()),
            "methods": ["UD", "UNIF"],
            "base_dir": "./simulation_results/visualization",
            "params": {
                "r_total": 5_000,
                "population_size": N_POPULATION,
                "store_sample": True,
                "n_estimators": LGBM_N_ESTIMATORS,
                "k_folds": K_FOLDS,
                "n_replications": 1,
            },
        },
    }

    return scenarios, all_methods, experiments
