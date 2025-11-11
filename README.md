# UD-DML: Uniform Design Double Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of Uniform Design Double Machine Learning (UD-DML) for efficient causal inference on large datasets. This repository reproduces the simulation studies from **UD_DML.pdf**.

---

## What is UD-DML?

UD-DML estimates average treatment effects (ATE) when working with massive datasets. It works in two steps:

1. **Subsampling**: Selects a space-filling subsample using uniform design (Latin-hypercube points with nearest-neighbor matching)
2. **Estimation**: Applies cross-fitted double machine learning on the subsample

The method is compared against uniform random subsampling (UNIF-DML) and full-data DML (FULL-DML).

---

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies**: numpy, scipy, numba, scikit-learn, lightgbm, pandas, joblib, tqdm, matplotlib

---

## Quick Start

```python
from data_generation import generate_obs_1_data
from methods import run_ud, run_unif
import config

# Generate synthetic data (10 covariates, 100k observations)
data = generate_obs_1_data(n=100_000, p=10)

# UD-DML estimate with 5,000 subsample points
ud_result = run_ud(
    data["X"], data["W"], data["Y_obs"], data["pi_true"],
    is_rct=False,
    r={"r_total": 5_000},
    k_folds=config.K_FOLDS,
)

# Uniform subsampling baseline
unif_result = run_unif(
    data["X"], data["W"], data["Y_obs"], data["pi_true"],
    is_rct=False,
    r={"r_total": 5_000},
    k_folds=config.K_FOLDS,
)

print("UD-DML ATE:", ud_result["est_ate"])
print("UNIF ATE:", unif_result["est_ate"])
```

Each estimator returns: point estimate, 95% confidence interval, runtime, and actual subsample size.

---

## Data Scenarios

The codebase includes **6 scenarios** (3 RCT + 3 Observational):

- **RCT-1, OBS-1**: Simple linear models, high overlap
- **RCT-2, OBS-2**: Moderate non-linearity, moderate overlap  
- **RCT-3, OBS-3**: Complex non-linear models, low overlap

All scenarios use **10 covariates** (`p=10`) with different distributions:
- **X1**: Independent uniform
- **X2**: Mixed marginals (uniform + normal)
- **X3**: Gaussian mixture + standard normal

---

## Running Experiments

Run all experiments:

```bash
python simulations.py
```

Run specific experiments:

```bash
python simulations.py --experiments experiment_subsample_size experiment_population_size --jobs 4
```

Quick test (fewer replications):

```bash
python simulations.py --fast-demo --jobs 1
```

### Available Experiments

1. **`experiment_visualization`** - Visual comparison of UD vs UNIF subsamples
2. **`experiment_subsample_size`** - Performance across subsample sizes {1k, 2.5k, 5k, 7.5k, 10k}
3. **`experiment_population_size`** - Scalability with population sizes {100k, 500k}
4. **`experiment_double_robust`** - Double-robustness tests with misspecified models
5. **`experiment_nuisance_sensitivity`** - Comparison across learners (LightGBM, Lasso, Random Forest)

### Output Files

Results are saved to:
- **Raw data**: `simulation_results/<experiment>/checkpoints/`
- **Plots**: `analysis_results/<experiment>/figures/`
- **Tables**: `analysis_results/<experiment>/tables/` (CSV and LaTeX formats)

---

## Configuration

Edit `config.py` to adjust settings:

```python
# Simulation parameters
BASE_SEED = 20250919
DEFAULT_REPLICATIONS = 500
N_POPULATION = 500_000
K_FOLDS = 2

# Nuisance learners (default: LightGBM)
DEFAULT_NUISANCE_LEARNER = 'lgbm'
LGBM_N_ESTIMATORS = 100
LGBM_MAX_DEPTH = 5

# Subsample sizes
SUBSAMPLE_TOTALS = [1_000, 2_500, 5_000, 7_500, 10_000]
POPULATION_SIZE_GRID = [100_000, 500_000]
```

---

## Project Structure

```
UD-DML/
├── config.py             # Configuration and experiment definitions
├── data_generation.py    # Data-generating processes (6 scenarios)
├── methods.py            # UD-DML, UNIF-DML, and FULL-DML estimators
├── evaluation.py         # Post-processing and visualization
├── simulations.py        # Main experiment driver
├── requirements.txt
└── README.md
```

---

## Citation

If you use this code, please cite the UD-DML manuscript.

---

## License

MIT License
