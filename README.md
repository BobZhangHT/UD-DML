# UD-DML: Uniform Design Double Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the simulation suite described in **UD_DML.pdf**. The repository focuses on
uniform-design subsampling paired with double/debiased machine learning, benchmarking the proposed
UD-DML estimator against uniform subsampling (UNIF-DML) across controlled randomised trials and
observational study designs.

---

## Overview

UD-DML first constructs a space-filling subsample in the covariate domain, then runs cross-fitted
Neyman-orthogonal scores on the selected points. In this code base:

- Covariates are generated from three canonical designs (independent uniform, mixed marginals,
  correlated block) with dimensionality fixed at **p = 10**.
- Average treatment effects are estimated with K-fold cross-fitting using nuisance models. The default
  learner is **LightGBM**, with **LassoCV** and **Random Forest** available as alternatives.
- Subsampling weights employ stratified Latin-hypercube draws with fast `cKDTree` lookup and JIT-compiled
  transformations via **Numba** for efficiency.
- Five experiment families reproduce the analyses in the paper: visualisation, subsample-size grids,
  population-size scaling, double-robustness checks, and nuisance-learner sensitivity. Each experiment
  also writes publication-ready CSV/LaTeX summaries (for example `population_size_publication_table.*`
  and `double_robust_publication_table.*`) alongside plots.

---

## Installation

```bash
pip install -r requirements.txt
```

The main dependencies are `numpy`, `scipy`, `numba`, `scikit-learn`, `lightgbm`, `pandas`, `joblib`,
`tqdm`, and `matplotlib`.

---

## Quick Start

```python
import numpy as np
from data_generation import generate_obs_1_data
from methods import run_ud, run_unif
import config

# Generate a synthetic observational dataset (p = 10 by default)
data = generate_obs_1_data(n=100_000, p=10)

# UD-DML estimate with a subsample of 5,000 points
ud_result = run_ud(
    data["X"], data["W"], data["Y_obs"], data["pi_true"],
    is_rct=False,
    r={"r_total": 5_000},
    k_folds=config.K_FOLDS,
)

# Uniform subsampling baseline (same r_total)
unif_result = run_unif(
    data["X"], data["W"], data["Y_obs"], data["pi_true"],
    is_rct=False,
    r={"r_total": 5_000},
    k_folds=config.K_FOLDS,
)

print("UD-DML ATE:", ud_result["est_ate"])
print("UNIF ATE:", unif_result["est_ate"])
```

Both estimators return point estimates, Wald-style 95% confidence intervals, runtime, and the actual
subsample size used.

---

## Simulation Experiments

All experiments are orchestrated by **`simulations.py`**. Every run checkpoints individual Monte
Carlo replications under `simulation_results/` and writes processed tables/figures to
`analysis_results/`.

```bash
# Run the full catalogue with automatic core detection
python simulations.py

# Fast smoke test (10 replications, trimmed grids)
python simulations.py --fast-demo --jobs 1

# Select a subset of experiments
python simulations.py --experiments experiment_population_size experiment_double_robust --jobs 4
```

### Experiment catalogue

1. **`experiment_visualization`** – draws 2D projections of UD vs. UNIF subsamples for every DGP.
2. **`experiment_subsample_size`** – compares UD and UNIF across `r_total` in
   {1k, 2.5k, 5k, 7.5k, 10k} for all six scenarios.
3. **`experiment_population_size`** – studies scalability over population sizes
   {100k, 500k} at low/high subsample budgets, benchmarking UD, UNIF, and the FULL-data estimator.
4. **`experiment_double_robust`** – checks the four nuisance specifications
   (correct/correct → wrong/wrong) on observational scenarios, with UD/UNIF outputs and boxplots.
5. **`experiment_nuisance_sensitivity`** – benchmarks UD/UNIF with alternative nuisance learners
   (`lasso_cv`, random forest, LightGBM) on the most challenging DGP (OBS-3).

Each experiment description, method list, output directory, and parameter grid is declared in
`config.get_experiments()`.

### Outputs & reporting

For every experiment the driver writes:

- Raw replication checkpoints under `simulation_results/<experiment>/checkpoints/`.
- Aggregated plots to `analysis_results/<short_name>/figures/` (where `<short_name>` is the experiment
  name without the `experiment_` prefix, e.g., `visualization`, `population_size`).
- Publication-ready tables (CSV + LaTeX) to `analysis_results/<short_name>/tables/`, e.g.:
  - `population_size_publication_table.csv/.tex` – Scenario × Population × Subsample with RMSE / CI coverage / CI width / runtime blocks (UD, UNIF, FULL).
  - `double_robust_publication_table.csv/.tex` – OBS scenarios with outcome/propensity misspecification grids for UD vs. UNIF.

Figures are rendered with matplotlib in `Agg` mode, so no GUI backend is required.

---

## Configuration Highlights

`config.py` centralises global settings:

```python
BASE_SEED = 20250919
DEFAULT_REPLICATIONS = 500
N_POPULATION = 500_000
K_FOLDS = 2

# Nuisance learners
DEFAULT_NUISANCE_LEARNER = 'lgbm'  # default: LightGBM
LGBM_N_ESTIMATORS = 100
LGBM_MAX_DEPTH = 5
LASSO_CV_FOLDS = 5
LASSO_CV_MAX_ITER = 5000
RF_N_JOBS = 1  # keep tree models single-threaded inside outer parallel loops

# Subsample grids
SUBSAMPLE_TOTALS = [1_000, 2_500, 5_000, 7_500, 10_000]
POPULATION_SIZE_GRID = [100_000, 500_000]
```

All six scenarios now pass `p=10` into the data generators. Adjust `SUBSAMPLE_TOTALS`,
`POPULATION_SIZE_GRID`, and experiment-specific dictionaries to explore different budgets or
population scales.

Parallel execution is controlled via:

- `config.MAX_PARALLEL_JOBS` – default upper bound on outer parallelism.
- `OS_DML_MAX_JOBS` – environment override evaluated at runtime.
- `RF_N_JOBS = 1` – tree-based nuisance learners remain single-threaded to avoid oversubscription.

---

## Project Structure

```
UD-DML/
├── config.py             # Global configuration & experiment catalogue
├── data_generation.py    # DGP definitions (p = 10)
├── evaluation.py         # Post-processing, tables, and figures
├── methods.py            # UD-DML and UNIF estimators
├── simulations.py        # Unified driver for all experiments
├── requirements.txt
├── README.md
├── simulation_results/   # Raw outputs (created at runtime)
└── analysis_results/     # Aggregated tables/plots (created at runtime)
```

---

## Citation

If you use this codebase, please cite the accompanying UD-DML manuscript when available.

---

## License

Released under the [MIT License](LICENSE).
